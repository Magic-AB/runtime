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
