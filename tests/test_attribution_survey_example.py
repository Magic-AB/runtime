"""Worked example for ``docs/attribution_methods_survey.md`` §10.

One identification and one silent abstention, both from the same CPU-only
fixture: ``mechanism_fixtures`` with a ``RegionSlowdown`` on attention at decode
steps 2 and 3 only, observed through the real ``monitor.residuals`` and
attributed by the real ``dr.attribute_dr``. It is scenario 3 of
``gitm/research/dr_probe.py`` (commit 28b7890), which prints the same numbers.

These tests pin dr.py's *current* behaviour, including what is wrong with it.
When dr.py learns to report why it abstains, or stops trusting a near-zero
standard error, they should fail and be rewritten against the new contract. The
status names that contract will use are pending (Andrew's v0); nothing here
depends on them.
"""

from __future__ import annotations

import dataclasses
from math import erfc, sqrt

import numpy as np

from gitm.optimizer.attribution import RankedHypotheses
from gitm.optimizer.dr import attribute_dr, doubly_robust_ate
from gitm.optimizer.invariants import INVARIANTS
from gitm.optimizer.mechanism_fixtures import RegionSlowdown, Scenario, generate, observe
from gitm.optimizer.monitor import Residuals
from gitm.planner.roofline import BatchConfig

ATTN = "attn_score_value"
BAND = next(i.band_width for i in INVARIANTS if i.id == "kernel_time")
POINT = BatchConfig(batch=8, kv_cache_len=2048)
N_STEPS = 8
#: ``Scenario.seed``; draws nothing at ``noise_cv=0.0``.
SEED = 0
INJECTED = RegionSlowdown(1.0, ops={ATTN}, steps={2, 3})


def _observe(*mechanisms):
    fx = generate(Scenario(points=[POINT], mechanisms=mechanisms, n_steps=N_STEPS,
                           noise_cv=0.0, seed=SEED))[0]
    return fx, observe(fx).residuals


def _rows(res: Residuals, op: str) -> list:
    return [k for k in res.per_kernel if k.op == op]


def _series(res: Residuals, op: str) -> np.ndarray:
    return np.array([k.r_kt for k in _rows(res, op)], dtype=float)


def _without(res: Residuals, op: str) -> Residuals:
    return Residuals(per_kernel=[k for k in res.per_kernel if k.op != op],
                     serialized_concurrency_fraction=res.serialized_concurrency_fraction)


def test_without_lm_head_the_null_effect_is_identified_but_its_p_values_are_noise():
    fx, injected = _observe(INJECTED)
    _, healthy = _observe()
    res = _without(injected, "lm_head")

    t = (np.abs(_series(res, ATTN)) > BAND).astype(float)
    assert (int(t.sum()), int((1 - t).sum())) == (64, 192)

    # Ground truth: the slowdown touches attention only, so every other op's
    # residuals are exactly the healthy run's and the true effect on each is 0.
    effects = sorted({k.op for k in res.per_kernel} - {ATTN})
    assert len(effects) == 4
    for e in effects:
        assert np.array_equal(_series(res, e), _series(healthy, e))

    ranked = attribute_dr(res, fx.graph)
    assert {(h.cause_op, h.effect_op) for h in ranked.hypotheses} == {(ATTN, e) for e in effects}
    assert all("n_treated=64)" in h.notes for h in ranked.hypotheses)

    pos = np.arange(len(t), dtype=float)
    by_effect = {h.effect_op: h for h in ranked.hypotheses}
    for e in effects:
        ate, se = doubly_robust_ate(_series(res, e), t, pos)
        # Identified: the point estimate is the true effect, to rounding.
        assert abs(ate) < 1e-12
        # Not inferred: the outcome is constant, so se is rounding error too. It
        # is not exactly 0, so attribute_dr's ``se not in (0.0, inf)`` guard
        # passes and p is one rounding error over another.
        assert 0.0 < se < 1e-12
        assert by_effect[e].p_value == erfc(abs(ate / se) / sqrt(2))
        assert by_effect[e].p_value < 1.0


def test_lm_head_truncation_abstains_silently_although_a_treatment_exists():
    fx, res = _observe(INJECTED)

    attn = _rows(res, ATTN)
    assert sum(abs(k.r_kt) > BAND for k in attn) == 64

    # lm_head runs once per step, so it is the shortest series and attribute_dr
    # truncates every op to its length.
    lengths = {op: len(_rows(res, op)) for op in {k.op for k in res.per_kernel}}
    assert lengths["lm_head"] == N_STEPS == min(lengths.values())
    # The rows that survive are attention's step 0, layers 0-7: none treated.
    survivors = attn[:N_STEPS]
    assert [k.layer for k in survivors] == list(range(N_STEPS))
    assert not any(abs(k.r_kt) > BAND for k in survivors)

    assert attribute_dr(res, fx.graph).hypotheses == []
    # The same rows without lm_head support four estimates, so the empty result
    # is a failure to estimate, not an absence of anything estimable.
    assert len(attribute_dr(_without(res, "lm_head"), fx.graph).hypotheses) == 4


def test_no_anomaly_no_controls_and_truncation_return_the_same_empty_result():
    fx_healthy, healthy = _observe()
    fx_all, all_steps = _observe(RegionSlowdown(0.5, ops={ATTN}))
    fx_trunc, truncated = _observe(INJECTED)

    assert not any(abs(v) > BAND for v in _series(healthy, ATTN))
    assert all(abs(v) > BAND for v in _series(all_steps, ATTN))

    results = [attribute_dr(healthy, fx_healthy.graph),
               attribute_dr(all_steps, fx_all.graph),
               attribute_dr(truncated, fx_trunc.graph)]
    assert results[0] == results[1] == results[2] == RankedHypotheses(hypotheses=[])
    # The return type has no field that could say which case it was.
    assert [f.name for f in dataclasses.fields(RankedHypotheses)] == ["hypotheses"]
