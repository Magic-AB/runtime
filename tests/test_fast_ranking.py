"""Fast-loop ranking from metrics-table rows (gitm.agents.fast_ranking, ENG-97).

Synthetic rows in the shape ENG-82's metrics table describes. Nothing here has seen a
real collector, and the store schema is still under review.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from gitm.agents.fast_ranking import fast_evidence, rank_fast
from gitm.agents.ranked_list import LeverMeta, read_lever_meta
from gitm.kernels.library import load_library
from gitm.optimizer.preconditions import GateContext

LIBRARY_YAML = Path(__file__).resolve().parents[1] / "gitm/kernels/library.yaml"
T0 = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()


def rows(running, waiting=None, cache=None, *, engine="vllm", window="w1", host="h1", t0=T0):
    """One row per key per second."""
    names = {"vllm": ("vllm.num_requests_running", "vllm.num_requests_waiting"),
             "sglang": ("sglang.num_running_reqs", "sglang.num_queue_reqs")}[engine]
    out = []
    for i, r in enumerate(running):
        base = {"window_id": window, "host_name": host, "sample_ts": t0 + i}
        out.append({**base, "key": names[0], "value": r})
        if waiting is not None:
            out.append({**base, "key": names[1], "value": waiting[i]})
        if cache is not None:
            out.append({**base, "key": "vllm.kv_cache_usage_perc", "value": cache[i]})
    return out


@pytest.fixture(scope="module")
def library():
    return load_library()


@pytest.fixture(scope="module")
def meta(library):
    return {s.name: LeverMeta(s.name, lever_class=1, restart_cost="none") for s in library}


def test_a_backlog_motivates_the_admission_knobs_and_the_entry_shows_the_counters(library, meta):
    r = rank_fast(rows([8] * 10, [30] * 10), library, meta)
    ev = r.evidence
    assert [c.signal for c in ev.causes] == ["admission_backlog"]
    assert ev.counters["peak_queue_depth"] == 30 and ev.counters["peak_running"] == 8
    top = r.ranked.entries[:2]
    assert {e.name for e in top} == {"max_num_seqs_dynamic", "max_num_batched_tokens_dynamic"}
    assert all(e.motivated_by == "admission_backlog" for e in top)
    why = " | ".join(top[0].why)
    assert "peak queue depth 30" in why and "peak running 8" in why


def test_quiet_traffic_ranks_by_the_library_prior(library, meta):
    r = rank_fast(rows([40] * 10, [0] * 10, [0.3] * 10), library, meta, max_num_seqs=64)
    assert r.evidence.causes == []
    entries = r.ranked.entries
    assert entries and all(e.delta_source == "library_prior" for e in entries)
    deltas = [e.predicted_delta for e in entries]
    assert deltas == sorted(deltas, reverse=True)
    spec = {s.name: s for s in library}
    assert all(e.predicted_delta == spec[e.name].expected_delta_mean for e in entries)


def test_motivated_levers_rank_ahead_of_a_larger_prior(library, meta):
    r = rank_fast(rows([8] * 10, [30] * 10), library, meta)
    entries = r.ranked.entries
    assert entries[0].motivated_by == "admission_backlog"
    # Something ranked below it has a larger prior: the precedence, not the gain, put it first.
    assert any(e.predicted_delta > entries[0].predicted_delta for e in entries[1:])


def test_kv_pressure_without_a_preemption_counter_says_so(library, meta):
    ev = fast_evidence(rows([20] * 10, [0] * 10, [0.97] * 10))
    assert [c.signal for c in ev.causes] == ["kv_cache_pressure"]
    assert any(m.startswith("preemptions:") for m in ev.missing)


def _preemption_rows(deltas, *, window_len=10):
    """Scheduler rows plus the collector's one-delta-per-window preemption counter."""
    out = rows([20] * (window_len * len(deltas)), [0] * (window_len * len(deltas)))
    for i, d in enumerate(deltas):
        out.append({"window_id": f"w{i}", "host_name": "h1", "key": "vllm.num_preemptions",
                    "sample_ts": T0 + (i + 1) * window_len, "value": d})
    return out


def test_preemption_deltas_from_every_window_are_summed():
    ev = fast_evidence(_preemption_rows([0, 4, 3]))
    assert ev.counters["total_preemptions"] == 7
    assert "kv_cache_preemption" in [c.signal for c in ev.causes]
    assert not any(m.startswith("preemptions:") for m in ev.missing)


def test_zero_preemption_deltas_raise_no_cause_and_are_not_reported_missing():
    ev = fast_evidence(_preemption_rows([0, 0, 0]))
    assert ev.counters["total_preemptions"] == 0
    assert "kv_cache_preemption" not in [c.signal for c in ev.causes]
    assert not any(m.startswith("preemptions:") for m in ev.missing)


def test_timestamps_may_be_iso_strings_and_naive_times_are_utc():
    base = rows([5, 5], [0, 0])
    iso = [{**r, "sample_ts": datetime.fromtimestamp(r["sample_ts"], timezone.utc).isoformat()}
           for r in base]
    naive = [{**r, "sample_ts": datetime.fromtimestamp(r["sample_ts"], timezone.utc)
              .replace(tzinfo=None)} for r in base]
    want = fast_evidence(base).counters["duration_s"]
    assert fast_evidence(iso).counters["duration_s"] == want
    assert fast_evidence(naive).counters["duration_s"] == want
    assert fast_evidence(iso).problems == []


def test_sglang_rows_work_but_kv_pressure_is_listed_as_undetectable():
    ev = fast_evidence(rows([8] * 10, [30] * 10, engine="sglang"))
    assert [c.signal for c in ev.causes] == ["admission_backlog"]
    assert any(m.startswith("kv_cache_usage:") for m in ev.missing)


def test_under_filled_batches_need_max_num_seqs():
    with_cap = fast_evidence(rows([4] * 10, [0] * 10), max_num_seqs=64)
    assert "under_filled_batch" in [c.signal for c in with_cap.causes]
    without = fast_evidence(rows([4] * 10, [0] * 10))
    assert "under_filled_batch" not in [c.signal for c in without.causes]
    assert any(m.startswith("batch_occupancy:") for m in without.missing)


def test_only_the_named_windows_count():
    healthy = rows([20] * 10, [0] * 10, window="good")
    sick = rows([5] * 10, [90] * 10, window="sick", t0=T0 + 100)
    both = fast_evidence(healthy + sick)
    only_good = fast_evidence(healthy + sick, window_ids=["good"])
    assert "admission_backlog" in [c.signal for c in both.causes]
    assert only_good.causes == [] and only_good.counters["n_windows"] == 1


def test_unusable_rows_are_dropped_and_reported():
    bad = rows([8] * 3, [30] * 3, [0.5] * 3)
    bad[0]["value"] = float("nan")
    bad[1]["value"] = -4
    bad[5]["value"] = 1.4                       # a percent, not a 0..1 fraction
    bad[2]["sample_ts"] = "yesterday"
    ev = fast_evidence(bad)
    assert len(ev.problems) == 4
    assert any("0..1 fraction" in p for p in ev.problems)
    assert ev.summary is not None


def test_two_hosts_are_refused_not_blended():
    mixed = rows([8] * 3, [30] * 3, host="a") + rows([8] * 3, [30] * 3, host="b")
    with pytest.raises(ValueError, match="host_name"):
        fast_evidence(mixed)
    assert fast_evidence(mixed, host_name="a").summary.n_samples == 3


def test_no_rows_says_so_and_ranking_still_returns_the_prior_list(library, meta):
    ev = fast_evidence([])
    assert ev.summary is None and ev.causes == [] and ev.missing
    assert rank_fast([], library, meta).ranked.entries


def test_unrelated_keys_are_ignored():
    noise = [{"window_id": "w", "host_name": "h", "sample_ts": T0, "key": "hw.gpu.power",
              "value": 500.0}]
    assert fast_evidence(noise).summary is None


def test_strict_mode_with_the_real_library_returns_nothing_until_levers_are_classed(library):
    real_meta = read_lever_meta(LIBRARY_YAML)
    strict = rank_fast(rows([8] * 5, [30] * 5), library, real_meta)
    assert strict.ranked.entries == []
    loose = rank_fast(rows([8] * 5, [30] * 5), library, real_meta, strict=False, top_n=3)
    assert len(loose.ranked.entries) == 3
    assert loose.ranked.entries[0].motivated_by == "admission_backlog"


def test_a_lever_that_changes_nothing_is_left_out(library, meta):
    name = "kv_cache_block_size_16"            # sets block_size to 16
    ctx = GateContext(workload="vllm-decode", dtype="bf16")  # this lever needs a known dtype
    before = rank_fast(rows([40] * 5, [0] * 5), library, meta, ctx=ctx)
    assert name in {e.name for e in before.ranked.entries}
    after = rank_fast(rows([40] * 5, [0] * 5), library, meta, ctx=ctx,
                      current_values={"block_size": 16})
    assert name not in {e.name for e in after.ranked.entries}
    assert dict(after.ranked.excluded)[name].startswith("baseline_noop")


def test_vllm_rows_gate_out_levers_for_other_workloads(library, meta):
    r = rank_fast(rows([20] * 10, [0] * 10), library, meta, strict=False)
    names = [e.name for e in r.ranked.entries]
    assert names, "the vLLM levers should still be ranked"
    assert not any(n.startswith(("edge_", "hft_")) for n in names)
    why = dict(r.ranked.excluded)
    assert "not_applicable" in why["edge_fp16_autocast"]


def test_a_lever_that_needs_a_dtype_is_excluded_until_the_dtype_is_known(library, meta):
    r = rank_fast(rows([20] * 10, [0] * 10), library, meta, strict=False)
    why = dict(r.ranked.excluded)
    assert "dtype is unknown" in why["kv_cache_block_size_16"]
