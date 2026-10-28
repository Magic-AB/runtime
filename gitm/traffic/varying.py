"""Synthetic traffic that varies: quiet stretches, bursts, and mixed prompt lengths (ENG-178).

Steady load never exercises traffic matching or the revert guard, so the on/off window
runner (ENG-183) cannot be tested against it. This builds a :class:`~gitm.traffic.schema.Trace`
out of *phases*, each with its own arrival rate, clustering and length distributions, so the
shape of the traffic changes over the run in ways the traffic-distance metric (ENG-98,
:mod:`gitm.traffic.shift`) can score.

The result is an ordinary canonical trace: ``write_timed_trace`` turns it into the JSONL the
harness replays. Nothing here talks to an engine, and nothing has been replayed on K2.6 / H200;
what is checked (``tests/test_traffic_varying.py``) is that the metric calls the right stretches
different and the matching stretches the same.

Everything is seeded; the same ``seed`` and phases give the same trace, and
``TraceMeta.sha256`` pins it to the generating parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from gitm.traffic.schema import CanonicalRequest, Trace, TraceMeta

__all__ = ["Phase", "DEFAULT_PHASES", "varying_trace", "slice_trace"]


@dataclass(frozen=True)
class Phase:
    """One stretch of traffic.

    ``rate_rps`` is the mean arrival rate. ``burst_size`` > 1 makes arrivals come in clumps:
    clump starts are Poisson at ``rate_rps / burst_size`` and every clump holds
    ``burst_size`` requests spread over ``burst_spread_s`` seconds, so the mean rate is
    unchanged but the arrivals are far more clustered than Poisson. Lengths are lognormal,
    given by their median and the log-space sigma.
    """

    name: str
    duration_s: float
    rate_rps: float
    input_median: float
    input_sigma: float = 0.5
    output_median: float = 128.0
    output_sigma: float = 0.5
    burst_size: int = 1
    burst_spread_s: float = 0.5


#: Quiet, then a burst, then long mixed prompts, then back to quiet. The two quiet phases
#: match each other on purpose: a metric that fires on the second one is wrong.
DEFAULT_PHASES: tuple[Phase, ...] = (
    Phase("quiet", 120.0, rate_rps=4.0, input_median=600, output_median=128),
    Phase("burst", 120.0, rate_rps=12.0, input_median=600, output_median=128,
          burst_size=24, burst_spread_s=1.0),
    Phase("mixed_long", 120.0, rate_rps=4.0, input_median=2400, input_sigma=0.9,
          output_median=384, output_sigma=0.7),
    Phase("quiet_again", 120.0, rate_rps=4.0, input_median=600, output_median=128),
)

_MIN_TOKENS = 1
_MAX_INPUT = 32_768
_MAX_OUTPUT = 8_192


def _lengths(rng: np.random.Generator, n: int, median: float, sigma: float, cap: int) -> list[int]:
    raw = rng.lognormal(math.log(median), sigma, n)
    return [int(min(max(round(x), _MIN_TOKENS), cap)) for x in raw]


def _arrivals(rng: np.random.Generator, p: Phase) -> list[float]:
    """Arrival offsets within the phase, sorted, all in ``[0, duration_s)``."""
    if p.burst_size <= 1:
        n = rng.poisson(p.rate_rps * p.duration_s)
        return sorted(float(t) for t in rng.uniform(0.0, p.duration_s, n))
    clumps = rng.poisson(p.rate_rps * p.duration_s / p.burst_size)
    starts = rng.uniform(0.0, p.duration_s, clumps)
    out: list[float] = []
    for s in starts:
        offs = rng.uniform(0.0, p.burst_spread_s, p.burst_size)
        out.extend(float(s + o) for o in offs if s + o < p.duration_s)
    return sorted(out)


def varying_trace(
    *, seed: int = 0, phases: tuple[Phase, ...] = DEFAULT_PHASES,
) -> Trace:
    """A canonical trace whose rate, clustering and lengths change phase by phase."""
    if not phases:
        raise ValueError("need at least one phase")
    for p in phases:
        if p.duration_s <= 0 or p.rate_rps <= 0:
            raise ValueError(f"phase {p.name!r}: duration_s and rate_rps must be positive")
        if p.input_median <= 0 or p.output_median <= 0:
            raise ValueError(f"phase {p.name!r}: length medians must be positive")
        if p.burst_size < 1 or p.burst_spread_s <= 0:
            raise ValueError(f"phase {p.name!r}: burst_size >= 1 and burst_spread_s > 0")
    rng = np.random.default_rng(seed)
    reqs: list[CanonicalRequest] = []
    t0 = 0.0
    notes = []
    for p in phases:
        arr = _arrivals(rng, p)
        ins = _lengths(rng, len(arr), p.input_median, p.input_sigma, _MAX_INPUT)
        outs = _lengths(rng, len(arr), p.output_median, p.output_sigma, _MAX_OUTPUT)
        reqs.extend(CanonicalRequest(arrival_s=t0 + a, input_tokens=i, output_tokens=o)
                    for a, i, o in zip(arr, ins, outs, strict=True))
        notes.append(f"{p.name}: {t0:.0f}-{t0 + p.duration_s:.0f}s, {len(arr)} requests")
        t0 += p.duration_s
    reqs.sort(key=lambda r: r.arrival_s)
    if not reqs:
        raise ValueError("the phases produced no requests")
    base = reqs[0].arrival_s  # arrival_s is seconds from the first row
    reqs = [CanonicalRequest(arrival_s=r.arrival_s - base, input_tokens=r.input_tokens,
                             output_tokens=r.output_tokens) for r in reqs]
    spec = json.dumps({"seed": seed, "phases": [asdict(p) for p in phases]}, sort_keys=True)
    meta = TraceMeta(
        source="synthetic-varying", path=f"synthetic:varying/seed={seed}",
        sha256=hashlib.sha256(spec.encode()).hexdigest(),
        rows_read=len(reqs), rows_emitted=len(reqs), span_s=reqs[-1].arrival_s,
        notes=["synthetic: generated, not captured; no prefix or session identity", *notes])
    return Trace(meta=meta, requests=reqs)


def slice_trace(trace: Trace, start_s: float, end_s: float) -> Trace:
    """The requests that arrive in ``[start_s, end_s)``, re-based to start at 0.

    The window's span is its nominal length ``end_s - start_s`` (not first-to-last arrival), so
    the rate it reports is requests per second of window.
    """
    if end_s <= start_s:
        raise ValueError("end_s must be after start_s")
    kept = [CanonicalRequest(arrival_s=r.arrival_s - start_s, input_tokens=r.input_tokens,
                             output_tokens=r.output_tokens)
            for r in trace.requests if start_s <= r.arrival_s < end_s]
    meta = TraceMeta(
        source=trace.meta.source, path=trace.meta.path, sha256=trace.meta.sha256,
        rows_read=len(kept), rows_emitted=len(kept), span_s=end_s - start_s,
        notes=[f"window {start_s:.0f}-{end_s:.0f}s of {trace.meta.path}"])
    return Trace(meta=meta, requests=kept)
