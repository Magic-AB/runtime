"""Baseline 3 for the distance metric: the loop's own ranking, without history."""

from __future__ import annotations

import random

from gitm.agents.policy import Policy, select_interventions
from gitm.kernels.library import load_library
from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate
from gitm.optimizer.history import History, LeverRecord
from gitm.optimizer.preconditions import GateContext
from gitm.research.distance_metric.baselines import catalog_predict_delta_ranking, random_ranking
from gitm.tracer.schema import KernelEvent, Trace

SKU = "AMD Instinct MI355X"
CTX = GateContext(workload="vllm-decode", dtype="bf16", hardware=SKU, kv_cache_len=8192, num_gpus=8,
                  has_collective=True, has_interconnect=True)


def _trace() -> Trace:
    events = [
        KernelEvent(name="fused_moe_kernel", start_ns=0, end_ns=600, stream_id=7,
                    device_id=0, correlation_id=1),
        KernelEvent(name="void gemm_kernel", start_ns=600, end_ns=900, stream_id=7,
                    device_id=0, correlation_id=2),
        KernelEvent(name="flash_attn_fwd", start_ns=900, end_ns=1000, stream_id=7,
                    device_id=0, correlation_id=3),
    ]
    return Trace(
        workload_id="vllm-decode", fingerprint="fp", run_id="r", device_count=1,
        vendor="amd", captured_at_ns=0, duration_ns=1000, events=events,
    )


def _spec(name, kernels=(), *, mean=0.05, whole_step=False, tier="moderate",
          commit=False, hardware=None) -> InterventionSpec:
    return InterventionSpec(
        name=name, summary="s", knob=name, value=1,
        expected_delta_mean=mean, expected_delta_lo=min(mean, 0.0), expected_delta_hi=max(mean, 0.1),
        source="t", applies_to_kernels=list(kernels), whole_step=whole_step,
        applicability=Applicability(workloads=["vllm-decode"], requires_hardware=hardware),
        safety=SafetyGate(tier=tier, requires_qualification_commit=commit),
    )


def _library() -> list[InterventionSpec]:
    """Every branch of the sort key: magnitude, a name tie-break, a non-positive
    estimate, an empty scope, and each rejection reason."""
    return [
        _spec("moe_lever", ["fused_moe_kernel"]),
        _spec("gemm_lever", ["gemm"]),
        _spec("gemm_twin", ["gemm"]),
        _spec("whole_step_lever", whole_step=True, mean=0.02),
        _spec("negative_lever", ["fused_moe_kernel"], mean=-0.03),
        _spec("empty_scope_lever"),
        _spec("risky_lever", ["fused_moe_kernel"], mean=0.2, tier="high_risk"),
        _spec("commit_lever", ["fused_moe_kernel"], mean=0.2, commit=True),
        _spec("h100_only_lever", ["fused_moe_kernel"], mean=0.2, hardware=["H100"]),
    ]


def _key(ranked):
    return [(c.spec.name, c.predicted_delta, c.rejected_reason, c.delta_source, c.demoted)
            for c in ranked]


def test_matches_a_direct_select_interventions_call():
    trace, lib, policy = _trace(), _library(), Policy(skip_high_risk=True)

    ours = catalog_predict_delta_ranking(trace, lib, policy, ctx=CTX)
    direct = select_interventions(trace, lib, policy, top_n=len(lib), ctx=CTX)

    assert _key(ours) == _key(direct)
    assert all(a.spec is b.spec for a, b in zip(ours, direct, strict=True))
    assert len(ours) == len(lib)
    # Sanity on the fixture: every branch of the sort key is actually exercised.
    assert [c.spec.name for c in ours][:4] == ["moe_lever", "whole_step_lever", "gemm_lever", "gemm_twin"]
    assert {c.rejected_reason.split(":")[0] for c in ours if c.rejected_reason} == {
        "not_applicable", "policy.skip_high_risk", "safety.requires_qualification_commit",
    }


def test_matches_the_loops_top_n_cut():
    trace, lib, policy = _trace(), _library(), Policy(skip_high_risk=True)

    ours = catalog_predict_delta_ranking(trace, lib, policy, ctx=CTX, top_n=5)

    assert _key(ours) == _key(select_interventions(trace, lib, policy, top_n=5, ctx=CTX))


def test_matches_on_the_real_catalog():
    trace, lib, policy = _trace(), load_library(workload="vllm-decode"), Policy(skip_high_risk=True)

    ours = catalog_predict_delta_ranking(trace, lib, policy, ctx=CTX)

    assert _key(ours) == _key(select_interventions(trace, lib, policy, top_n=len(lib), ctx=CTX))
    assert len(ours) == len(lib)


def test_accepts_a_one_shot_iterable():
    trace, lib, policy = _trace(), _library(), Policy()

    from_gen = catalog_predict_delta_ranking(trace, (s for s in lib), policy, ctx=CTX)

    assert _key(from_gen) == _key(catalog_predict_delta_ranking(trace, lib, policy, ctx=CTX))
    assert len(from_gen) == len(lib)


def test_does_not_mutate_its_inputs():
    trace, lib, policy = _trace(), _library(), Policy(skip_high_risk=True)
    trace_before = trace.model_dump()
    lib_before = [s.model_dump() for s in lib]
    ids_before = [id(s) for s in lib]
    policy_before = (policy.require_qualification_commit, policy.skip_high_risk, policy.use_history)

    catalog_predict_delta_ranking(trace, lib, policy, ctx=CTX)

    assert trace.model_dump() == trace_before
    assert [s.model_dump() for s in lib] == lib_before
    assert [id(s) for s in lib] == ids_before
    assert (policy.require_qualification_commit, policy.skip_high_risk, policy.use_history) == policy_before


def test_repeat_calls_are_identical():
    trace, lib, policy = _trace(), _library(), Policy(skip_high_risk=True)

    assert _key(catalog_predict_delta_ranking(trace, lib, policy, ctx=CTX)) == _key(
        catalog_predict_delta_ranking(trace, lib, policy, ctx=CTX)
    )


def test_never_reads_measured_outcomes_even_when_the_policy_would():
    """A baseline that read history would be scoring with outcomes. With
    ``use_history`` on and a record that would reorder the ranking, the
    baseline still ranks from priors alone."""
    trace, lib = _trace(), _library()
    policy = Policy(skip_high_risk=True, use_history=True)
    record = LeverRecord(
        intervention_name="moe_lever", gpu_sku=SKU, fingerprint="fp", runs=1, attempts=1,
        wins=0, losses=1, inconclusive=0, mean_delta=-0.5, best_delta=-0.5, worst_delta=-0.5,
        last_run_id="r1",
    )
    history = History(records={(record.intervention_name, SKU, "fp"): record}, runs_read=1)
    with_history = select_interventions(trace, lib, policy, top_n=len(lib), ctx=CTX,
                                        history=history, gpu_sku=SKU, fingerprint="fp")
    assert _key(with_history) != _key(select_interventions(trace, lib, policy, top_n=len(lib), ctx=CTX))

    ours = catalog_predict_delta_ranking(trace, lib, policy, ctx=CTX)

    assert all(c.delta_source == "prior" and not c.demoted for c in ours)
    assert _key(ours) == _key(select_interventions(trace, lib, Policy(skip_high_risk=True),
                                                   top_n=len(lib), ctx=CTX))


def _big_library() -> list[InterventionSpec]:
    """22 gate-passing candidates (22! orders, so two seeds colliding is not a
    real possibility) plus the three rejected ones from ``_library``."""
    extra = [_spec(f"lever_{i:02d}", ["gemm"], mean=0.01 * (i + 1)) for i in range(16)]
    return _library() + extra


def _names(ranked):
    return [c.spec.name for c in ranked]


def test_random_ranking_is_repeatable_for_a_fixed_seed():
    trace, lib, policy = _trace(), _big_library(), Policy(skip_high_risk=True)

    runs = [_key(random_ranking(trace, lib, policy, seed=7, ctx=CTX)) for _ in range(3)]

    assert runs[0] == runs[1] == runs[2]
    # Library order is not an input to the permutation either.
    assert _key(random_ranking(trace, list(reversed(lib)), policy, seed=7, ctx=CTX)) == runs[0]


def test_random_ranking_differs_across_seeds():
    trace, lib, policy = _trace(), _big_library(), Policy(skip_high_risk=True)

    a = _names(random_ranking(trace, lib, policy, seed=1, ctx=CTX))
    b = _names(random_ranking(trace, lib, policy, seed=2, ctx=CTX))

    assert a != b
    assert sorted(a) == sorted(b)
    # Nor is it the score order baseline 3 produces.
    assert a != _names(catalog_predict_delta_ranking(trace, lib, policy, ctx=CTX))


def test_random_ranking_applies_the_same_gate_and_puts_rejected_last():
    trace, lib, policy = _trace(), _big_library(), Policy(skip_high_risk=True)
    gate = catalog_predict_delta_ranking(trace, lib, policy, ctx=CTX)
    gate_reasons = {c.spec.name: c.rejected_reason for c in gate}
    gate_rejected = [c for c in gate if c.rejected_reason is not None]
    assert len(gate_rejected) == 3

    tails = []
    for seed in range(10):
        ranked = random_ranking(trace, lib, policy, seed=seed, ctx=CTX)
        assert {c.spec.name: c.rejected_reason for c in ranked} == gate_reasons
        flags = [c.rejected_reason is not None for c in ranked]
        assert flags == [False] * (len(lib) - 3) + [True] * 3
        tails.append(_key(ranked[-3:]))

    # The rejected tail is baseline 3's, identical for every seed — not shuffled.
    assert all(t == _key(gate_rejected) for t in tails)


def test_random_ranking_blanks_every_predicted_delta():
    trace, lib, policy = _trace(), _big_library(), Policy(skip_high_risk=True)
    assert any(c.predicted_delta > 0 for c in catalog_predict_delta_ranking(trace, lib, policy, ctx=CTX))

    ranked = random_ranking(trace, lib, policy, seed=5, ctx=CTX)

    assert [c.predicted_delta for c in ranked] == [0.0] * len(lib)


def test_random_ranking_top_n_cuts_after_shuffling():
    trace, lib, policy = _trace(), _big_library(), Policy(skip_high_risk=True)

    full = random_ranking(trace, lib, policy, seed=3, ctx=CTX)

    assert _key(random_ranking(trace, lib, policy, seed=3, ctx=CTX, top_n=5)) == _key(full[:5])


def test_random_ranking_does_not_mutate_its_inputs_or_the_global_rng():
    trace, lib, policy = _trace(), _big_library(), Policy(skip_high_risk=True)
    trace_before = trace.model_dump()
    lib_before = [s.model_dump() for s in lib]
    ids_before = [id(s) for s in lib]
    policy_before = (policy.require_qualification_commit, policy.skip_high_risk, policy.use_history)
    random.seed(12345)
    rng_before = random.getstate()

    random_ranking(trace, lib, policy, seed=11, ctx=CTX)

    assert random.getstate() == rng_before
    assert trace.model_dump() == trace_before
    assert [s.model_dump() for s in lib] == lib_before
    assert [id(s) for s in lib] == ids_before
    assert (policy.require_qualification_commit, policy.skip_high_risk, policy.use_history) == policy_before


# --- baseline 1: exact match -------------------------------------------------

from datetime import datetime, timezone  # noqa: E402

from gitm.playbook.schema import (  # noqa: E402
    EnvCapture,
    Evidence,
    Invalidation,
    MeasuredDelta,
    Playbook,
    PlaybookRow,
    Provenance,
    RowIdentity,
)
from gitm.research.distance_metric.baselines import (  # noqa: E402
    bottleneck_keyword_ranking,
    exact_match_ranking,
)
from gitm.tracer.schema import MemcpyEvent  # noqa: E402
from gitm.traffic.regime import Regime, SourceKind  # noqa: E402

_T0 = datetime(2026, 9, 2, tzinfo=timezone.utc)
_ENV = EnvCapture(engine="vllm", engine_version="0.11.0")


def _regime(input_p50=4096, concurrency=64) -> Regime:
    return Regime(source_kind=SourceKind.PRODUCTION, trace="t", requests=100, rate_rps=1.0,
                  io_ratio=8.0, input_p50=input_p50, input_p95=input_p50 * 2, output_p50=512,
                  output_p95=1024, burstiness=1.0, concurrency=concurrency)


def _query(regime=None) -> RowIdentity:
    return RowIdentity(model="m", model_revision="r", gpu_sku=SKU, env=_ENV,
                       regime=regime or _regime(), knobs={"placeholder": 1})


def _row(row_id, knob, value, tput, regime=None, **kw) -> PlaybookRow:
    regime = regime or _regime()
    return PlaybookRow(
        row_id=row_id,
        identity=RowIdentity(model="m", model_revision="r", gpu_sku=SKU, env=_ENV,
                             regime=regime, knobs={knob: value}),
        delta=MeasuredDelta(throughput_pct=tput, ttft_p99_ms=0.0, itl_p99_ms=0.0, repeats=5),
        provenance=Provenance(trace_source="t", trace_sha256="0" * 64,
                              regime_label=regime.label(), verified_at=_T0),
        evidence=Evidence.MEASURED, **kw)


def _lib_for_exact() -> list[InterventionSpec]:
    return [_spec("lever_a", ["gemm"]), _spec("lever_b", ["gemm"]), _spec("lever_c", ["gemm"]),
            _spec("risky", ["gemm"], tier="high_risk")]


def test_exact_match_ranks_by_the_rows_measured_delta_and_abstains_elsewhere():
    pb = Playbook(rows=[
        _row("a", "lever_a", 1, 4.0),
        _row("b", "lever_b", 1, 10.0),
        _row("near", "lever_c", 1, 50.0, regime=_regime(input_p50=4608)),  # not exact
    ])
    out = exact_match_ranking(pb, _query(), _trace(), _lib_for_exact(),
                              Policy(skip_high_risk=True), ctx=CTX)

    assert out.matched == ("lever_b", "lever_a")
    assert [c.predicted_delta for c in out.ranking[:2]] == [0.10, 0.04]
    assert out.abstained == {"lever_c": "uncalibrated"}
    names = [c.spec.name for c in out.ranking]
    assert names == ["lever_b", "lever_a", "lever_c", "risky"]
    assert out.ranking[2].rejected_reason == "no_exact_match:uncalibrated"
    assert out.ranking[3].rejected_reason == "policy.skip_high_risk"
    assert len(out.ranking) == 4


def test_exact_match_ignores_invalidated_rows_and_reports_no_match():
    pb = Playbook(rows=[_row("a", "lever_a", 1, 9.0,
                             invalidated=Invalidation(reason="bad", at=_T0))])
    out = exact_match_ranking(pb, _query(), _trace(), _lib_for_exact(),
                              Policy(skip_high_risk=True), ctx=CTX)

    assert out.matched == ()
    assert set(out.abstained) == {"lever_a", "lever_b", "lever_c"}
    assert set(out.abstained.values()) == {"no_match"}


def test_exact_match_excludes_rows_measured_at_the_target():
    pb = Playbook(rows=[_row("leak", "lever_a", 1, 9.0), _row("ok", "lever_b", 1, 3.0)])
    out = exact_match_ranking(pb, _query(), _trace(), _lib_for_exact(),
                              Policy(skip_high_risk=True), ctx=CTX,
                              exclude_row_ids=frozenset({"leak"}))

    assert out.matched == ("lever_b",)
    assert "lever_a" in out.abstained


def test_exact_match_abstains_on_knob_values_a_row_cannot_hold():
    spec = _spec("dict_lever", ["gemm"])
    spec = spec.model_copy(update={"value": {"level": 3}})
    out = exact_match_ranking(Playbook(rows=[]), _query(), _trace(), [spec],
                              Policy(), ctx=CTX)

    assert out.abstained == {"dict_lever": "knobs_not_representable_in_a_playbook_row"}
    assert out.matched == ()


def test_exact_match_does_not_mutate_inputs_and_is_repeatable():
    pb = Playbook(rows=[_row("a", "lever_a", 1, 4.0)])
    q, lib = _query(), _lib_for_exact()
    pb_before, q_before = pb.model_dump(), q.model_dump()
    first = exact_match_ranking(pb, q, _trace(), lib, Policy(), ctx=CTX)
    second = exact_match_ranking(pb, q, _trace(), lib, Policy(), ctx=CTX)

    assert pb.model_dump() == pb_before and q.model_dump() == q_before
    assert _key(first.ranking) == _key(second.ranking)


# --- baseline 2: bottleneck / keyword ---------------------------------------

def _memcpy_trace() -> Trace:
    kernels = [KernelEvent(name="gemm_kernel", start_ns=0, end_ns=100, stream_id=7,
                           device_id=0, correlation_id=1)]
    copies = [MemcpyEvent(start_ns=100, end_ns=900, stream_id=8, device_id=0, correlation_id=2,
                          bytes=1 << 20, src="host", dst="device")]
    return Trace(workload_id="vllm-decode", fingerprint="fp", run_id="r", device_count=1,
                 vendor="amd", captured_at_ns=0, duration_ns=900, events=kernels + copies)


def _knob_lib() -> list[InterventionSpec]:
    def s(name, knob):
        return _spec(name, ["gemm"]).model_copy(update={"knob": knob})
    return [s("z_graphs", "cuda_graph_mode"), s("a_cache", "kv_cache_dtype"),
            s("m_chunk", "enable_chunked_prefill"), s("b_cache2", "gpu_memory_utilization"),
            s("n_other", "tensor_parallel_size")]


def test_bottleneck_baseline_prefers_levers_matching_the_class_keywords():
    out = bottleneck_keyword_ranking(_memcpy_trace(), _knob_lib(), Policy(), ctx=CTX)

    assert out.bottleneck_class == "memory_bound"
    names = [c.spec.name for c in out.ranking]
    # kv_cache_dtype matches "cache" and "kv"; gpu_memory_utilization matches
    # "gpu_memory"; the rest match nothing and follow in name order.
    assert names[:2] == ["a_cache", "b_cache2"]
    assert names[2:] == ["m_chunk", "n_other", "z_graphs"]
    assert all(c.predicted_delta == 0.0 for c in out.ranking)


def test_bottleneck_baseline_changes_with_the_class_and_never_reads_scores():
    mem = bottleneck_keyword_ranking(_memcpy_trace(), _knob_lib(), Policy(), ctx=CTX)
    comp = bottleneck_keyword_ranking(_trace(), _knob_lib(), Policy(), ctx=CTX)

    assert comp.bottleneck_class == "idle_stall"  # one serialized stream
    assert [c.spec.name for c in comp.ranking][0] == "m_chunk"  # "prefill", "chunk"
    assert [c.spec.name for c in mem.ranking] != [c.spec.name for c in comp.ranking]
    # Same ranking whatever the catalog thinks each lever is worth.
    rich = [s.model_copy(update={"expected_delta_mean": 0.9}) for s in _knob_lib()]
    assert [c.spec.name for c in bottleneck_keyword_ranking(
        _memcpy_trace(), rich, Policy(), ctx=CTX).ranking] == [c.spec.name for c in mem.ranking]


def test_bottleneck_baseline_keeps_gate_rejections_last_and_is_complete():
    lib = _knob_lib() + [_spec("risky", ["gemm"], tier="high_risk")]
    out = bottleneck_keyword_ranking(_memcpy_trace(), lib, Policy(skip_high_risk=True), ctx=CTX)

    assert len(out.ranking) == len(lib)
    assert out.ranking[-1].spec.name == "risky"
    assert out.ranking[-1].rejected_reason == "policy.skip_high_risk"
