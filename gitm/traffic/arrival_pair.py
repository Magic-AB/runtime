"""Smooth and bursty versions of the same traffic, for the arrival-pattern experiment (ENG-153).

The question is how much throughput and tail latency change between bursty and smooth
arrivals at the same mean rate. To answer it the two traces must differ in *when* requests
arrive and in nothing else, so this builds them from one set of requests: the same count (so the
same mean rate, exactly), the same input and output lengths, and different arrival times.

* smooth: arrivals uniform over the run, i.e. a Poisson process conditioned on its count;
* bursty: arrivals in clumps of ``burst_size`` requests, each clump spread over
  ``burst_spread_s`` seconds, with the clump starts spread uniformly over the run.

Both are ordinary canonical traces; ``python -m gitm.traffic.arrival_pair`` writes them as timed
traces and prints the ``vllm bench serve`` commands. It launches nothing, and nothing has been
replayed on an engine. What is checked is that the pair differs only in arrival clustering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math

import numpy as np

from gitm.traffic.schema import CanonicalRequest, Trace, TraceMeta
from gitm.traffic.varying import _MAX_INPUT, _MAX_OUTPUT, _lengths

__all__ = ["arrival_pair", "main"]


def _pin(arrivals: list[float], duration_s: float) -> list[float]:
    """Stretch the arrivals so the first is at 0 and the last at ``duration_s`` exactly.

    The mean rate of a replay is its request count over the span between its first and last
    arrival, so two traces with the same count are only at the same rate if they span the same
    time. Random placement leaves the spans a few percent apart; this removes the difference.
    """
    lo, hi = arrivals[0], arrivals[-1]
    scale = duration_s / (hi - lo)
    return [(a - lo) * scale for a in arrivals]


def _trace(name: str, arrivals: list[float], ins: list[int], outs: list[int],
           spec: dict) -> Trace:
    base = arrivals[0]
    # Rounded to the timed-trace file's precision, so writing and re-reading changes nothing.
    # Unrounded, a request sitting on a 1 s bin edge can flip bins after the round trip.
    reqs = [CanonicalRequest(arrival_s=round(a - base, 6), input_tokens=i, output_tokens=o)
            for a, i, o in zip(arrivals, ins, outs, strict=True)]
    meta = TraceMeta(
        source=f"synthetic-{name}", path=f"synthetic:arrival-pair/{name}",
        sha256=hashlib.sha256(json.dumps({**spec, "pattern": name}, sort_keys=True).encode()
                              ).hexdigest(),
        rows_read=len(reqs), rows_emitted=len(reqs), span_s=reqs[-1].arrival_s,
        notes=["synthetic: generated, not captured; no prefix or session identity",
               f"arrival pattern: {name}; same requests as its pair, different arrival times"])
    return Trace(meta=meta, requests=reqs)


def arrival_pair(
    *, rate_rps: float, duration_s: float, seed: int = 0, burst_size: int = 24,
    burst_spread_s: float = 1.0, input_median: float = 600.0, input_sigma: float = 0.5,
    output_median: float = 128.0, output_sigma: float = 0.5,
) -> tuple[Trace, Trace]:
    """``(smooth, bursty)``: the same requests, arriving in two different patterns."""
    if rate_rps <= 0 or duration_s <= 0:
        raise ValueError("rate_rps and duration_s must be positive")
    if burst_size < 2:
        raise ValueError("burst_size must be at least 2 for a bursty trace")
    if not 0 < burst_spread_s < duration_s:
        raise ValueError("burst_spread_s must be positive and shorter than the run")
    if input_median <= 0 or output_median <= 0:
        raise ValueError("length medians must be positive")
    n = round(rate_rps * duration_s)
    if n < 2 * burst_size:
        raise ValueError(f"{n} requests is too few for bursts of {burst_size}; "
                         "raise rate_rps or duration_s")
    rng = np.random.default_rng(seed)
    ins = _lengths(rng, n, input_median, input_sigma, _MAX_INPUT)
    outs = _lengths(rng, n, output_median, output_sigma, _MAX_OUTPUT)
    smooth = sorted(float(t) for t in rng.uniform(0.0, duration_s, n))
    clumps = math.ceil(n / burst_size)
    starts = rng.uniform(0.0, duration_s - burst_spread_s, clumps)
    bursty: list[float] = []
    for k, s in enumerate(starts):
        size = min(burst_size, n - k * burst_size)
        bursty.extend(float(s + o) for o in rng.uniform(0.0, burst_spread_s, size))
    bursty.sort()
    spec = {"rate_rps": rate_rps, "duration_s": duration_s, "seed": seed,
            "burst_size": burst_size, "burst_spread_s": burst_spread_s,
            "input_median": input_median, "input_sigma": input_sigma,
            "output_median": output_median, "output_sigma": output_sigma}
    return (_trace("smooth", _pin(smooth, duration_s), ins, outs, spec),
            _trace("bursty", _pin(bursty, duration_s), ins, outs, spec))


def main(argv: list[str] | None = None) -> int:
    """Write the pair as timed traces and print the commands that replay them."""
    from gitm.traffic.replay import read_timed_trace, write_timed_trace
    from gitm.traffic.validate import REPLAY_THRESHOLDS, compare

    ap = argparse.ArgumentParser(prog="python -m gitm.traffic.arrival_pair")
    ap.add_argument("--rate", type=float, required=True, help="mean requests per second")
    ap.add_argument("--duration", type=float, default=300.0, help="seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--burst-size", type=int, default=24)
    ap.add_argument("--burst-spread", type=float, default=1.0, help="seconds per clump")
    ap.add_argument("--out-prefix", default="pair")
    ap.add_argument("--model", default="MODEL")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    a = ap.parse_args(argv)

    smooth, bursty = arrival_pair(rate_rps=a.rate, duration_s=a.duration, seed=a.seed,
                                  burst_size=a.burst_size, burst_spread_s=a.burst_spread)
    ok = True
    for tr in (smooth, bursty):
        out = f"{a.out_prefix}_{tr.meta.source.removeprefix('synthetic-')}.jsonl"
        print(tr.meta.summary())
        plan = write_timed_trace(tr, out)
        report = compare(tr, read_timed_trace(out), thresholds=REPLAY_THRESHOLDS)
        ok = ok and report.passed
        print(f"  wrote {plan.path} ({plan.requests} requests); round trip "
              f"{'PASS' if report.passed else 'FAIL'}")
        print("  " + " ".join(plan.bench_serve_argv(
            model=a.model, base_url=a.base_url, tokenizer=a.tokenizer)))
    print("\n(not run: this launches nothing; replay each on a box with vLLM and a server)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
