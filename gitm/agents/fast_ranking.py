"""Fast-loop ranking from the telemetry store's metrics table (ENG-97).

The fast loop decides without a kernel trace: its evidence is workload-wide counters,
and the store keeps them in the metrics table, one row per sample (ENG-82). This module
turns those rows into the scheduler causes the loop already knows how to rank on, then
ranks the library and attaches the counters that put each lever where it is.

    metrics rows -> SchedulerSample -> summarize -> scheduler_causes -> motivated knobs
                                                  -> select_interventions -> rank_for_loop

It reuses ``summarize``, ``scheduler_causes`` and ``select_interventions``; the new part is
only the translation from rows, and the library prior.

Library prior. ``select_interventions`` prices a lever from the kernel trace, so with no
trace every predicted gain is 0.0 and only the motivated precedence orders anything. Here a
lever whose gain came back as an unmeasured 0.0 prior takes the library's
``expected_delta_mean`` instead, marked ``library_prior``, and the list is re-sorted in the
same tier order (not rejected, not demoted, motivated first, larger gain first). The
trace-based "aimed at these ops" tier does not exist without a trace.

What the documented metric keys cannot show (``FastEvidence.missing`` says so each time):

* preemptions: ``vllm.num_preemptions`` (one delta row per window, as the collector stores
  it) feeds ``kv_cache_preemption``; SGLang has no equivalent key in the store;
* KV occupancy: only ``vllm.kv_cache_usage_perc`` is listed. SGLang's table has none (its
  ``radix_cache_hit_rate`` is a different quantity), so ``kv_cache_pressure`` cannot be
  detected on SGLang, which is what the K2.6 box serves;
* batch occupancy needs the engine's ``max_num_seqs``, which the rows do not carry.

Provisional: the store schema is still under review. Row shape assumed here (the metrics
table in ENG-82): ``window_id``, ``host_name``, ``sample_ts`` (datetime or epoch seconds),
``key``, ``value``. Health filtering belongs to the caller (``window_ids``), at read time.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from gitm.agents.policy import Policy, RankedCandidate, select_interventions
from gitm.agents.ranked_list import LeverMeta, RankedList, rank_for_loop
from gitm.optimizer.preconditions import GateContext
from gitm.optimizer.scheduler_attribution import SchedulerCause, scheduler_causes
from gitm.tracer.schema import Trace
from gitm.tracer.vllm_stats import SchedulerSample, SchedulerStatsSummary, summarize

__all__ = ["FastEvidence", "FastRanking", "KEY_MAP", "fast_evidence", "rank_fast"]

#: Documented metric keys (ENG-82, vLLM and SGLang tables) for each scheduler field.
KEY_MAP: dict[str, tuple[str, ...]] = {
    "num_running": ("vllm.num_requests_running", "sglang.num_running_reqs"),
    "num_waiting": ("vllm.num_requests_waiting", "sglang.num_queue_reqs"),
    "gpu_cache_usage": ("vllm.kv_cache_usage_perc",),
    # A counter: the collector stores one row per window holding that window's delta.
    "preemptions_delta": ("vllm.num_preemptions",),
}
_FIELD_OF = {k: f for f, keys in KEY_MAP.items() for k in keys}


@dataclass
class FastEvidence:
    summary: SchedulerStatsSummary | None
    causes: list[SchedulerCause]
    #: knob -> the cause that argues for it, first cause wins (the loop's own rule).
    motivated: dict[str, str]
    #: The raw numbers the causes were computed from.
    counters: dict[str, Any]
    #: Signals the rows could not show, each with why. Never empty-by-accident.
    missing: list[str] = field(default_factory=list)
    #: Rows dropped as unusable, each with why.
    problems: list[str] = field(default_factory=list)


def _epoch(ts: Any) -> float | None:
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:  # a naive time is read as UTC, not the host's local zone
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.timestamp()
    if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
        return None
    return float(ts)


def fast_evidence(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_num_seqs: int | None = None,
    window_ids: Iterable[str] | None = None,
    host_name: str | None = None,
) -> FastEvidence:
    """Scheduler causes and the counters behind them, from metrics-table rows.

    ``window_ids`` keeps only those windows: pass the windows whose node was healthy, the
    filter the design applies first, at read time. Rows from more than one host must be
    narrowed with ``host_name``; mixing hosts would blend two engines into one series.
    """
    keep = set(window_ids) if window_ids is not None else None
    problems: list[str] = []
    by_ts: dict[tuple[str, float], dict[str, float]] = {}
    preempt: dict[float, float] = {}  # window-end time -> preemptions in that window
    windows: set[str] = set()
    hosts: set[str] = set()
    engines: set[str] = set()
    for i, r in enumerate(rows):
        fld = _FIELD_OF.get(r.get("key"))
        if fld is None:
            continue
        if keep is not None and r.get("window_id") not in keep:
            continue
        host = r.get("host_name")
        if host_name is not None and host != host_name:
            continue
        hosts.add(host)
        engines.add(str(r.get("key")).split(".", 1)[0])
        ts = _epoch(r.get("sample_ts"))
        v = r.get("value")
        if ts is None:
            problems.append(f"row {i} {r.get('key')}: bad sample_ts {r.get('sample_ts')!r}")
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            problems.append(f"row {i} {r.get('key')}: value {v!r} is not a finite number")
            continue
        if v < 0 or (fld == "gpu_cache_usage" and v > 1.0):
            problems.append(f"row {i} {r.get('key')}: value {v!r} outside its range"
                            + (" (a 0..1 fraction)" if fld == "gpu_cache_usage" else ""))
            continue
        windows.add(r.get("window_id"))
        if fld == "preemptions_delta":
            preempt[ts] = preempt.get(ts, 0.0) + float(v)
            continue
        by_ts.setdefault((host, ts), {})[fld] = float(v)
    if len(hosts) > 1:
        raise ValueError(f"rows from {len(hosts)} hosts; pass host_name to pick one engine")

    missing: list[str] = []
    if not by_ts:
        return FastEvidence(None, [], {}, {"n_samples": 0}, ["no usable rows for the scheduler keys"],
                            problems)
    t0 = min(ts for _, ts in by_ts)
    samples = []
    ordered = sorted(by_ts.items(), key=lambda kv: kv[0][1])
    for i, ((_, ts), f) in enumerate(ordered):
        run, wait = f.get("num_running"), f.get("num_waiting")
        # The summary wants a running total. The first sample is the zero point and the last
        # carries every window's delta, so the summed deltas are what the summary reports.
        cum = None
        if preempt:
            cum = 0 if i == 0 else int(sum(d for t, d in preempt.items() if t <= ts))
            if i == len(ordered) - 1:
                cum = int(sum(preempt.values()))
        samples.append(SchedulerSample(
            t_ns=int((ts - t0) * 1e9),
            num_running=None if run is None else int(run),
            num_waiting=None if wait is None else int(wait),
            num_unfinished=int(run + wait) if run is not None and wait is not None else None,
            gpu_cache_usage=f.get("gpu_cache_usage"),
            preemptions_cumulative=cum,
            batch_occupancy=(min(run / max_num_seqs, 1.0)
                             if run is not None and max_num_seqs else None)))
    summary = summarize(samples, t0_wall_ns=int(t0 * 1e9), max_num_seqs=max_num_seqs)
    causes = scheduler_causes(summary)

    motivated: dict[str, str] = {}
    for c in causes:
        for knob in c.motivates_knobs:
            motivated.setdefault(knob, c.signal)

    if not preempt:
        missing.append("preemptions: no vllm.num_preemptions rows (SGLang has no equivalent key), "
                       "so kv_cache_preemption cannot be detected")
    if summary.peak_gpu_cache_usage is None:
        missing.append("kv_cache_usage: no KV-occupancy rows (SGLang's table has none), so "
                       "kv_cache_pressure cannot be detected")
    if max_num_seqs is None:
        missing.append("batch_occupancy: max_num_seqs not given, so under_filled_batch cannot "
                       "be detected")
    counters = {
        "n_samples": summary.n_samples, "n_windows": len(windows),
        "duration_s": summary.duration_s, "peak_queue_depth": summary.peak_queue_depth,
        "peak_running": summary.peak_running, "mean_running": summary.mean_running,
        "mean_batch_occupancy": summary.mean_batch_occupancy,
        "peak_gpu_cache_usage": summary.peak_gpu_cache_usage,
        "total_preemptions": summary.total_preemptions, "engines": sorted(engines)}
    return FastEvidence(summary, causes, motivated, counters, missing, problems)


@dataclass
class FastRanking:
    ranked: RankedList
    evidence: FastEvidence


def _tier(c: RankedCandidate) -> tuple[Any, ...]:
    return (c.rejected_reason is not None, c.demoted, c.motivated_by is None, -c.predicted_delta)


def rank_fast(
    rows: Iterable[Mapping[str, Any]],
    library: Sequence[Any],
    meta: Mapping[str, LeverMeta],
    *,
    policy: Policy | None = None,
    max_num_seqs: int | None = None,
    window_ids: Iterable[str] | None = None,
    host_name: str | None = None,
    gpu_sku: str | None = None,
    fingerprint: str | None = None,
    history: Any = None,
    current_values: Mapping[str, Any] | None = None,
    ctx: GateContext | None = None,
    top_n: int | None = None,
    strict: bool = True,
    mvp_classes: Iterable[int] = (1,),
    cost_weighted: bool = False,
) -> FastRanking:
    """The ranked list for the fast loop, from metrics rows and the library.

    Ranks the whole library (``select_interventions`` with no trace and the telemetry's
    motivated knobs), fills the library prior where the trace-free gain came back as an
    unmeasured 0.0, and hands the result to :func:`rank_for_loop` with the causes, so each
    entry says which counters put it there.
    """
    ev = fast_evidence(rows, max_num_seqs=max_num_seqs, window_ids=window_ids,
                       host_name=host_name)
    if ctx is None and ev.counters.get("engines") == ["vllm"]:
        # Without a gate the whole library is ranked, edge and HFT levers included. vLLM rows
        # say what the box is running, so gate on the vLLM workload. dtype is unknown here,
        # so levers that need a dtype are excluded rather than guessed; pass ctx to say more.
        ctx = GateContext(workload="vllm-decode", hardware=gpu_sku)
    empty = Trace(workload_id="fast-loop", fingerprint=fingerprint or "none", run_id="none",
                  device_count=1, vendor="none", captured_at_ns=0, duration_ns=0, events=[],
                  source="none")
    ranked = select_interventions(
        empty, list(library), policy or Policy(), top_n=max(len(library), 1),
        history=history, gpu_sku=gpu_sku, fingerprint=fingerprint,
        motivated=ev.motivated, current_values=current_values, ctx=ctx)
    filled = []
    for c in ranked:
        if c.delta_source == "prior" and c.predicted_delta == 0.0 and c.rejected_reason is None:
            c = replace(c, predicted_delta=float(c.spec.expected_delta_mean),
                        delta_source="library_prior")
        filled.append(c)
    filled.sort(key=_tier)
    out = rank_for_loop(filled, meta, top_n=top_n, mvp_classes=mvp_classes, strict=strict,
                        cost_weighted=cost_weighted, causes=ev.causes)
    return FastRanking(out, ev)
