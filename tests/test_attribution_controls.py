"""Known-cause control harness: ``gitm/research/attribution_controls``.

CPU-only and seeded. The dr.py tests at the bottom pin dr.py's *current*
behaviour on the suite, including what is wrong with it. They document that
behaviour and are expected to change when dr.py is fixed.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import math
import subprocess
import sys
import typing
from pathlib import Path

import numpy as np
import pytest

from gitm.optimizer import mechanism_fixtures as mf
from gitm.research.attribution_controls import controls as ctl
from gitm.research.attribution_controls.adapters import DrEstimator, dr_estimator
from gitm.research.attribution_controls.controls import (
    AMBIGUOUS,
    KINDS,
    NULL,
    POSITIVE,
    Cause,
    ControlCase,
    EstimatorInput,
    Planted,
    Suite,
    Truth,
    default_suite,
)
from gitm.research.attribution_controls.doubles import (
    always_abstain,
    always_claim,
    fingerprint,
    oracle,
    random_claim,
)
from gitm.research.attribution_controls.gate import (
    INVALID,
    VALID,
    EstimatorBlocked,
    GuardedEstimator,
    NotValidated,
    ValidationResult,
    run_suite,
    validate,
)
from gitm.research.attribution_controls.judge import (
    CORRECT,
    MISSED,
    OUTCOMES,
    OVERCLAIMED_GROUP,
    VALID_ABSTENTION,
    WRONG_CONFIDENT,
    WRONG_HEDGED,
    JudgeConfig,
    covers,
    judge,
)
from gitm.research.attribution_controls.verdict import (
    Alternative,
    Claim,
    MalformedVerdict,
    Verdict,
    validate_verdict,
)
from gitm.research.attribution_controls.vocabulary import (
    ESTIMABLE,
    M1,
    M3K,
    M4,
    M5E,
    NOT_IDENTIFIABLE,
    STATUSES,
)

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "gitm/research/attribution_controls"
#: The threshold these tests run at. Not a recommendation: the harness has no default.
THRESHOLD = 0.95
CFG = JudgeConfig(confidence_threshold=THRESHOLD)
ATTN = "attn_score_value"


@pytest.fixture(scope="module")
def suite() -> Suite:
    return default_suite()


@pytest.fixture(scope="module")
def dr_validation(suite):
    return validate(dr_estimator, suite, CFG)


def _case(suite: Suite, name: str) -> ControlCase:
    return next(c for c in suite.cases if c.name == name)


def _abstain() -> Verdict:
    return Verdict(None, None, NOT_IDENTIFIABLE, ("not enough controls",), remedy="more steps")


def _claim(op=None, mechanism=None, confidence=1.0) -> Verdict:
    return Verdict(Claim.of(op, mechanism), confidence, ESTIMABLE, ("test",))


ATTN_GROUP = Truth((Planted(M1, (ATTN,), None, None, (("alpha", 1.0),)),),
                   frozenset({Cause(ATTN, M1), Cause(ATTN, M3K), Cause(ATTN, M4)}))
NOTHING = Truth((), frozenset())


# ── truth cannot reach the estimator ────────────────────────────────────────


def test_estimator_input_has_only_capture_fields():
    assert [f.name for f in dataclasses.fields(EstimatorInput)] == ["residuals", "graph", "trace"]
    hints = typing.get_type_hints(EstimatorInput)
    truth_types = {Truth, Cause, Planted, ControlCase, Suite, mf.Scenario, mf.Fixture}
    for name, hint in hints.items():
        args = set(typing.get_args(hint)) | {hint}
        assert not args & truth_types, name


_FORBIDDEN = (Truth, Cause, Planted, ControlCase, Suite, mf.Scenario, mf.Fixture,
              mf.RegionSlowdown, mf.AdditiveCost, mf.Serialization, mf.RegimeGate,
              mf.TracerOverhead, mf.Misroute, functools.partial)


def _reachable(root):
    seen, stack = set(), [root]
    while stack:
        o = stack.pop()
        if id(o) in seen or isinstance(o, str | bytes | int | float | bool | type(None) | type):
            continue
        seen.add(id(o))
        yield o
        if isinstance(o, dict):
            stack += [*o.keys(), *o.values()]
        elif isinstance(o, list | tuple | set | frozenset):
            stack += list(o)
        elif isinstance(o, np.ndarray):
            continue
        else:
            stack += list(getattr(o, "__dict__", {}).values())
            for cls in type(o).__mro__:
                for s in getattr(cls, "__slots__", ()):
                    if isinstance(s, str) and hasattr(o, s):
                        stack.append(getattr(o, s))


def test_no_truth_object_is_reachable_from_any_case_input(suite):
    for case in suite.cases:
        inp = case.build()
        assert type(inp) is EstimatorInput
        leaked = [type(o).__name__ for o in _reachable(inp) if isinstance(o, _FORBIDDEN)]
        assert not leaked, (case.name, leaked)


def test_nothing_can_be_attached_to_an_input(suite):
    inp = suite.cases[0].build()
    with pytest.raises(dataclasses.FrozenInstanceError):
        inp.truth = suite.cases[0].truth  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        object.__setattr__(inp, "truth", suite.cases[0].truth)


def test_run_suite_hands_the_estimator_one_input_and_nothing_else(suite):
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return _abstain()

    run_suite(spy, suite, CFG)
    assert len(calls) == len(suite.cases)
    for args, kwargs in calls:
        assert kwargs == {}
        assert len(args) == 1 and type(args[0]) is EstimatorInput


def _imported_modules(path: Path) -> set[str]:
    mods = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
            mods |= {f"{node.module}.{a.name}" for a in node.names}
    return mods


def test_only_the_adapter_imports_dr():
    # A source-level check: at runtime ``import gitm`` already loads dr.py
    # (gitm/__init__ -> gitm.api -> gitm.scheduler.loop -> gitm.optimizer.dr).
    dr_modules = {"gitm.optimizer.dr", "gitm.research.dr_probe"}
    for path in sorted(PKG.glob("*.py")):
        imports_dr = bool(_imported_modules(path) & dr_modules)
        assert imports_dr == (path.name == "adapters.py"), path.name
        if path.name != "adapters.py":
            names = {n.id for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Name)}
            names |= {n.attr for n in ast.walk(ast.parse(path.read_text()))
                      if isinstance(n, ast.Attribute)}
            assert not names & {"attribute_dr", "doubly_robust_ate", "dr"}, path.name


def test_status_names_live_in_one_file_and_absent_is_not_one():
    assert STATUSES == (ESTIMABLE, NOT_IDENTIFIABLE)
    assert not any("absent" in s.lower() for s in STATUSES)
    for path in sorted(PKG.glob("*.py")):
        has_placeholder = "PENDING-STATUS" in path.read_text()
        assert has_placeholder == (path.name == "vocabulary.py"), path.name


# ── the suite ───────────────────────────────────────────────────────────────


def test_suite_has_every_kind_with_the_required_variety(suite):
    assert {c.kind for c in suite.cases} == set(KINDS)
    names = {c.name for c in suite.cases}
    assert {"pos_attn_strong", "pos_attn_weak", "hostile_lm_head_truncation",
            "hostile_every_row_treated"} <= names
    assert len({c.seed for c in suite.of_kind(NULL)}) >= 3
    assert any(n.startswith("hostile_independent_ar1") for n in names)
    for c in suite.of_kind(NULL):
        assert c.truth.nothing_planted
    for c in suite.of_kind(POSITIVE) + suite.of_kind(AMBIGUOUS):
        assert not c.truth.nothing_planted


def test_truth_group_is_empty_exactly_when_nothing_is_planted():
    with pytest.raises(ValueError):
        Truth((), frozenset({Cause(ATTN, M1)}))
    with pytest.raises(ValueError):
        Truth((Planted(M1, (ATTN,), None, None),), frozenset())


def test_every_group_member_has_a_witness_with_the_identical_trace(suite):
    """The groups come from the fixtures, not from the doc: each member, planted
    instead, gives the same trace events as the case."""
    def events(trace):
        return [(e.name, e.start_ns, e.end_ns, e.stream_id) for e in trace.events]

    for case in suite.cases:
        if case.truth.nothing_planted or case.build().trace is None:
            continue
        assert {c for c, _ in case.witnesses} == case.truth.group, case.name
        base = events(case.build().trace)
        for cause, make in case.witnesses:
            assert events(make()) == base, (case.name, cause)


def test_ambiguous_cases_include_a_real_cost_vs_measurement_artifact(suite):
    case = _case(suite, "amb_cost_vs_tracer")
    assert case.truth.group == {Cause(None, M3K), Cause(None, M4), Cause(None, M5E)}
    assert case.truth.planted[0].mechanism == M5E


def test_series_generators_are_dr_probes():
    from gitm.research import dr_probe

    for seed in (0, 2, 10):
        a = ctl._ar1(np.random.default_rng(seed), 40)
        b = dr_probe._ar1(np.random.default_rng(seed), 40)
        assert np.array_equal(a, b)
    s = ctl.spike_series(45)
    assert ctl.series_residuals(s) == dr_probe._residuals(s)


def test_oracle_refuses_a_suite_with_one_input_and_two_truths(suite):
    c = _case(suite, "pos_attn_strong")
    twin = dataclasses.replace(c, name="twin", truth=NOTHING)
    with pytest.raises(ValueError, match="same input"):
        oracle(Suite((c, twin)))


# ── verdict check ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("v", [
    Verdict(None, None, "absent", ("x",)),
    Verdict(None, None, "estimable", ()),
    Verdict(Claim.of(ATTN), 1.5, ESTIMABLE),
    Verdict(Claim.of(ATTN), -0.1, ESTIMABLE),
    Verdict(Claim.of(ATTN), math.nan, ESTIMABLE),
    Verdict(Claim.of(ATTN), True, ESTIMABLE),
    Verdict(None, 0.5, ESTIMABLE),
    Verdict(Claim.of(ATTN), 0.5, NOT_IDENTIFIABLE, ("x",)),
    Verdict(None, None, NOT_IDENTIFIABLE, ()),
    Verdict(None, None, NOT_IDENTIFIABLE, ("  ",)),
    Verdict(None, None, NOT_IDENTIFIABLE, ["x"]),  # type: ignore[arg-type]
    Verdict(Claim.of(ATTN, "M9"), 0.5, ESTIMABLE),
    Verdict(Claim.of(None, None), 0.5, ESTIMABLE),
    Verdict(Claim(frozenset()), 0.5, ESTIMABLE),
    Verdict(ATTN, 0.5, ESTIMABLE),  # type: ignore[arg-type]
    Verdict(None, None, NOT_IDENTIFIABLE, ("x",), remedy=""),
    "not a verdict",
], ids=["absent-status", "unknown-status", "conf>1", "conf<0", "conf-nan", "conf-bool",
        "conf-without-claim", "abstain-with-claim", "abstain-no-reason", "blank-reason",
        "reasons-list", "unknown-mechanism", "empty-alternative", "empty-claim",
        "claim-not-Claim", "blank-remedy", "not-a-verdict"])
def test_malformed_verdicts_are_rejected(v):
    with pytest.raises(MalformedVerdict):
        validate_verdict(v)
    with pytest.raises(MalformedVerdict):
        judge(v, ATTN_GROUP, CFG)


@pytest.mark.parametrize("v", [
    _abstain(), _claim(ATTN), _claim(ATTN, M1, None), Verdict(None, None, None),
    Verdict(Claim.group([(ATTN, M1), (ATTN, M3K)]), 0.0, None, (), None),
])
def test_well_formed_verdicts_pass(v):
    assert validate_verdict(v) is v


# ── judge ───────────────────────────────────────────────────────────────────


def test_judge_config_has_no_default_threshold():
    with pytest.raises(TypeError):
        JudgeConfig()  # type: ignore[call-arg]
    for bad in (-0.01, 1.01, math.nan, True):
        with pytest.raises(ValueError):
            JudgeConfig(bad)


@pytest.mark.parametrize(("verdict", "truth", "outcome"), [
    (_claim(ATTN), ATTN_GROUP, CORRECT),
    (Verdict(Claim.group([(ATTN, M1), (ATTN, M3K), (ATTN, M4)]), 0.99, ESTIMABLE), ATTN_GROUP,
     CORRECT),
    (_abstain(), ATTN_GROUP, VALID_ABSTENTION),
    (_claim("mlp_down", confidence=0.99), ATTN_GROUP, WRONG_CONFIDENT),
    (_claim("mlp_down", confidence=0.5), ATTN_GROUP, WRONG_HEDGED),
    (Verdict(None, None, None), ATTN_GROUP, MISSED),
    (Verdict(None, None, ESTIMABLE, ("looked, found nothing",)), ATTN_GROUP, MISSED),
    (_claim(ATTN, M1, 0.99), ATTN_GROUP, OVERCLAIMED_GROUP),
], ids=["correct-op-only", "correct-group", "valid-abstention", "wrong-confident",
        "wrong-hedged", "missed-empty", "missed-no-abstention-reason", "overclaimed-group"])
def test_judge_each_outcome_by_hand(verdict, truth, outcome):
    assert judge(verdict, truth, CFG).outcome == outcome


def test_outcomes_are_the_six_the_spec_lists():
    assert OUTCOMES == (CORRECT, VALID_ABSTENTION, WRONG_CONFIDENT, WRONG_HEDGED, MISSED,
                        OVERCLAIMED_GROUP)


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_abstention_is_never_wrong_on_any_case(suite, threshold):
    cfg = JudgeConfig(threshold)
    for case in suite.cases:
        assert judge(_abstain(), case.truth, cfg).outcome == VALID_ABSTENTION, case.name
    for truth in (ATTN_GROUP, NOTHING):
        assert judge(_abstain(), truth, cfg).outcome == VALID_ABSTENTION
    assert {c.kind for c in suite.cases} == set(KINDS)


def test_threshold_boundary_and_unstated_confidence():
    assert judge(_claim("mlp_down", confidence=THRESHOLD), ATTN_GROUP, CFG).outcome == WRONG_CONFIDENT
    assert judge(_claim("mlp_down", confidence=0.9499), ATTN_GROUP, CFG).outcome == WRONG_HEDGED
    assert judge(_claim("mlp_down", confidence=None), ATTN_GROUP, CFG).outcome == WRONG_CONFIDENT
    assert judge(_claim(ATTN, M1, None), ATTN_GROUP, CFG).outcome == OVERCLAIMED_GROUP


def test_any_claim_on_nothing_planted_is_wrong():
    assert judge(_claim(ATTN), NOTHING, CFG).outcome == WRONG_CONFIDENT
    assert judge(_claim(ATTN, confidence=0.1), NOTHING, CFG).outcome == WRONG_HEDGED
    assert judge(Verdict(None, None, None), NOTHING, CFG).outcome == MISSED


def test_group_rules(suite):
    gate = _case(suite, "amb_gate_vs_onset").truth
    tracer = _case(suite, "amb_cost_vs_tracer").truth
    # Naming the planted member itself, confidently, still overclaims: the capture cannot tell it.
    assert gate.planted[0].mechanism == M4
    assert judge(_claim(ATTN, M4, 0.99), gate, CFG).outcome == OVERCLAIMED_GROUP
    assert judge(_claim(ATTN, M4, 0.5), gate, CFG).outcome == WRONG_HEDGED
    assert judge(_claim(ATTN), gate, CFG).outcome == CORRECT
    # The tracer case is on no op: an op claim is wrong; a class claim overclaims.
    assert judge(_claim(ATTN), tracer, CFG).outcome == WRONG_CONFIDENT
    assert judge(_claim(None, M5E), tracer, CFG).outcome == OVERCLAIMED_GROUP
    group = Verdict(Claim.group([(None, M3K), (None, M4), (None, M5E)]), 1.0, ESTIMABLE)
    assert judge(group, tracer, CFG).outcome == CORRECT
    # A group that also names something not planted is wrong, not correct.
    extra = Verdict(Claim.group([(ATTN, None), ("mlp_down", None)]), 1.0, ESTIMABLE)
    assert judge(extra, ATTN_GROUP, CFG).outcome == WRONG_CONFIDENT
    assert not covers(Alternative(ATTN, None), Cause(None, M5E))
    assert covers(Alternative(None, M5E), Cause(None, M5E))


# ── gate: doubles ───────────────────────────────────────────────────────────


def test_always_abstain_is_valid_and_useless_and_the_report_says_both(suite):
    v = validate(always_abstain(), suite, CFG)
    assert v.status == VALID and v.failures == ()
    assert v.report.identification_rate == 0.0
    assert v.report.positives_identified == 0 and v.report.positives_total == 5
    assert v.report.abstention_rate == 1.0
    line = v.summary()
    assert "VALID" in line and "positive identification 0/5 = 0%" in line
    assert "abstention 17/17 = 100%" in line


def test_oracle_is_valid_and_identifies_every_positive(suite):
    v = validate(oracle(suite), suite, CFG)
    assert v.status == VALID
    assert all(r.outcome == CORRECT for r in v.report.results if r.kind == POSITIVE)
    assert v.report.identification_rate == 1.0
    assert all(r.outcome in (CORRECT, VALID_ABSTENTION) for r in v.report.results)


def _ops_in(suite):
    return sorted({k.op for c in suite.cases for k in c.build().residuals.per_kernel})


def test_always_claim_is_invalid_for_every_op_and_each_outcome_is_explained(suite):
    """Valid-by-luck on none: the nulls make every always-claimer invalid, and
    every case's outcome follows from whether the op covers that case's group."""
    for op in [*_ops_in(suite), "not_an_op"]:
        v = validate(always_claim(op), suite, CFG)
        assert v.status == INVALID, op
        for case, r in zip(suite.cases, v.report.results, strict=True):
            expected = (CORRECT if case.truth.group
                        and all(covers(Alternative(op), m) for m in case.truth.group)
                        else WRONG_CONFIDENT)
            assert r.outcome == expected, (op, case.name)
        assert {c.name for c in suite.of_kind(NULL)} <= {f.case for f in v.failures}


def test_random_claim_is_valid_only_by_luck_at_seed_0(suite):
    # Seed 0: 17 uniform confidences, none >= 0.95 (chance 0.95**17 = 0.42), so
    # every wrong claim is hedged. The suite is too small to catch an estimator
    # that is confidently wrong only ~5% of the time.
    lucky = validate(random_claim(0), suite, CFG)
    assert lucky.status == VALID
    assert max(r.confidence for r in lucky.report.results) < THRESHOLD
    assert lucky.report.count(WRONG_HEDGED) == 16
    caught = validate(random_claim(1), suite, CFG)
    assert caught.status == INVALID
    assert {f.case for f in caught.failures} == {
        "pos_attn_strong", "hostile_independent_spikes_seed45", "hostile_independent_ar1_seed0"}


def test_validate_needs_a_positive_and_a_null(suite):
    with pytest.raises(ValueError):
        validate(always_abstain(), Suite(suite.of_kind(POSITIVE)), CFG)
    with pytest.raises(ValueError):
        validate(always_abstain(), Suite(suite.of_kind(NULL)), CFG)


def test_invalid_records_which_cases_and_why(suite):
    v = validate(always_claim(ATTN), suite, CFG)
    f = next(f for f in v.failures if f.case == "null_healthy_seed0")
    assert f.outcome == WRONG_CONFIDENT
    assert any("nothing was planted" in w for w in f.why)
    assert "null_healthy_seed0" in v.explain()


def test_malformed_verdict_names_the_case(suite):
    with pytest.raises(MalformedVerdict, match="pos_attn_strong"):
        run_suite(lambda inp: Verdict(None, 0.5, ESTIMABLE), suite, CFG)


# ── gate: guard ─────────────────────────────────────────────────────────────


def test_guard_refuses_an_unvalidated_estimator(suite):
    est = always_abstain()
    with pytest.raises(NotValidated):
        GuardedEstimator(est, None)  # type: ignore[arg-type]
    other = validate(always_abstain(), suite, CFG)
    with pytest.raises(NotValidated, match="not of this"):
        GuardedEstimator(est, other)
    with pytest.raises(NotValidated):
        ValidationResult(VALID, (), other.report, est)


def test_guard_blocks_an_invalid_estimator(suite):
    est = always_claim(ATTN)
    guarded = GuardedEstimator(est, validate(est, suite, CFG))
    with pytest.raises(EstimatorBlocked, match="null_healthy_seed0"):
        guarded(suite.cases[0].build())


def test_guard_passes_a_valid_estimator_through_unchanged(suite):
    fixed = _abstain()

    class Fixed:
        name = "fixed"

        def __call__(self, inp):
            return fixed

    est = Fixed()
    guarded = GuardedEstimator(est, validate(est, suite, CFG))
    assert guarded(suite.cases[0].build()) is fixed
    abst = always_abstain()
    g = GuardedEstimator(abst, validate(abst, suite, CFG))
    for case in suite.cases[:3]:
        assert g(case.build()) == abst(case.build())


def test_guard_blocks_dr(suite, dr_validation):
    guarded = GuardedEstimator(dr_estimator, dr_validation)
    with pytest.raises(EstimatorBlocked):
        guarded(suite.cases[0].build())
    with pytest.raises(NotValidated):
        GuardedEstimator(DrEstimator(), dr_validation)  # a different object


# ── determinism ─────────────────────────────────────────────────────────────


def test_two_runs_give_identical_reports(suite):
    for make in (lambda: oracle(suite), always_abstain, lambda: random_claim(1),
                 lambda: dr_estimator):
        assert run_suite(make(), suite, CFG) == run_suite(make(), default_suite(), CFG)


def test_case_inputs_are_deterministic():
    a, b = default_suite(), default_suite()
    for ca, cb in zip(a.cases, b.cases, strict=True):
        assert fingerprint(ca.build()) == fingerprint(cb.build()), ca.name


def test_run_local_is_deterministic_and_says_what_it_is():
    cmd = [sys.executable, "-m", "gitm.research.attribution_controls.run_local"]
    first = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    second = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert first == second
    assert first.startswith("CPU-ONLY SYNTHETIC SUITE")
    assert "nothing about real GPU traces" in first
    assert "case seeds: " in first and "random_claim seed 0" in first
    assert "dr.attribute_dr: INVALID" in first


# ── dr.py through the adapter ───────────────────────────────────────────────
# These pin dr.py's behaviour as of main at d3a1518, which includes #137. Before
# #137 the direction case was wrong-confident (E1) and three cases returned an
# empty result (E3); both changed. They are expected to change again when dr.py
# does. Seeds are pinned in controls.py.

DR_OUTCOMES = {
    # Correct op, but the confidence comes from p-values over a near-zero se (E6).
    "pos_attn_strong": CORRECT,
    "pos_attn_weak": CORRECT,
    "pos_mlp_down_steps0to3": CORRECT,
    "pos_qkv_additive": CORRECT,
    # After #137 the direction case names A, the true cause (it named B before).
    "pos_series_direction_seed3": CORRECT,
    # Still an empty result, with no reason, on every healthy run.
    "null_healthy_seed0": MISSED,
    "null_healthy_seed1": MISSED,
    "null_healthy_seed2": MISSED,
    # Now names attention, at confidence 0.9996 (empty before #137).
    "amb_gate_vs_onset": CORRECT,
    "amb_cost_vs_tracer": MISSED,
    # Now names attention, at 0.9231 (empty before #137).
    "hostile_lm_head_truncation": CORRECT,
    "hostile_every_row_treated": MISSED,
    # E4/E5: independent series. Wrong at every seed; confident where p is small.
    "hostile_independent_spikes_seed0": WRONG_HEDGED,
    "hostile_independent_spikes_seed45": WRONG_CONFIDENT,
    "hostile_independent_ar1_seed0": WRONG_HEDGED,
    "hostile_independent_ar1_seed2": WRONG_CONFIDENT,
    "hostile_independent_ar1_seed10": WRONG_CONFIDENT,
}


def test_dr_outcome_on_every_case(dr_validation):
    assert {r.case: r.outcome for r in dr_validation.report.results} == DR_OUTCOMES


def test_dr_truncation_is_now_found_but_with_a_near_zero_se(suite, dr_validation):
    r = next(r for r in dr_validation.report.results if r.case == "hostile_lm_head_truncation")
    assert r.outcome == CORRECT
    v = dr_estimator(_case(suite, "hostile_lm_head_truncation").build())
    assert v.claim == Claim.of(op=ATTN)
    # E6 is still there: the ATE is +0.000 with se=0.000, and the confidence
    # comes from a p-value over that.
    assert "se=0.000" in v.reasons[0]
    # The planted cause was there: 64 attention rows outside the band.
    inp = _case(suite, "hostile_lm_head_truncation").build()
    assert sum(abs(k.r_kt) > 0.4 for k in inp.residuals.per_kernel if k.op == ATTN) == 64


def test_dr_independent_series_are_confidently_wrong_where_p_is_small(dr_validation):
    by_case = {r.case: r for r in dr_validation.report.results}
    seed45 = by_case["hostile_independent_spikes_seed45"]
    assert seed45.claim == "A/*"
    assert seed45.confidence == 1.0 - 2.771156616538538e-05  # dr_probe scenario 5, seed 45
    assert by_case["hostile_independent_ar1_seed2"].confidence >= THRESHOLD
    assert by_case["hostile_independent_ar1_seed10"].claim == "A/*"  # p = 0.0
    assert by_case["hostile_independent_spikes_seed0"].confidence < THRESHOLD


def test_dr_is_invalid_and_the_report_shows_its_usefulness_too(dr_validation):
    assert dr_validation.status == INVALID
    assert {f.case for f in dr_validation.failures} == {
        "hostile_independent_spikes_seed45",
        "hostile_independent_ar1_seed2", "hostile_independent_ar1_seed10"}
    assert "positive identification 5/5 = 100%" in dr_validation.summary()


def test_dr_adapter_mapping(suite):
    v = dr_estimator(_case(suite, "pos_series_direction_seed3").build())
    assert v.claim == Claim.of(op="A") and v.status is None
    assert v.confidence == 1.0  # p underflows to 0.0
    assert v.reasons[0].startswith("top of 2 hypotheses: A -> B")
