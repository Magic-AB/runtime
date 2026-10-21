"""Distance metric v1 (docs/distance_metric_v1_design.md), on a CPU-only synthetic grid.

Nothing here measures a candidate. The grid's baseline captures are healthy
``mechanism_fixtures`` traces, so these tests check the metric's mechanics:
no outcome can reach it, it is a symmetric distance, each gate and component
does what §4 says, the config drives it, and pair selection runs blind.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
from scipy.stats import kendalltau

from gitm.optimizer import apply as apply_mod
from gitm.optimizer import history as history_mod
from gitm.optimizer.history import History, LeverRecord
from gitm.research.distance_metric import agreement as ag
from gitm.research.distance_metric.candidates import COMMON_SET_BELOW_MINIMUM, common_candidates
from gitm.research.distance_metric.config import DistanceConfig
from gitm.research.distance_metric.distance import COMPONENTS, GATES, distance, rel, tv
from gitm.research.distance_metric.features import (
    KnobRead,
    PointDescription,
    extract_features,
)
from gitm.research.distance_metric.pairs import (
    FAR,
    INCONCLUSIVE,
    INELIGIBLE,
    INTERMEDIATE,
    NEAR,
    select_pairs,
)
from gitm.research.distance_metric.synthetic import SyntheticEngine, synthetic_grid, synthetic_point

CFG = DistanceConfig()
SEED = 0
BASE = "c64_in4096_out512"
#: The doc's own "near-ish" pairs on O (§7.1).
DOC_NEAR = {("c32_in4096_out512", "c40_in4096_out512"), ("c40_in4096_out512", "c48_in4096_out512"),
            ("c48_in4096_out512", "c64_in4096_out512"), ("c128_in4096_out512", "c96_in4096_out512")}


@pytest.fixture(scope="module")
def grid():
    return {f.name: f for f in (extract_features(p, CFG) for p in synthetic_grid(CFG, seed=SEED))}


@pytest.fixture(scope="module")
def base(grid):
    return grid[BASE]


def _point(**kw) -> PointDescription:
    return synthetic_point(64, 4096, 512, seed=SEED, **kw)


def _same(r1, r2):
    assert r1.status == r2.status
    assert r1.distance == r2.distance
    assert r1.setting_component == r2.setting_component
    assert r1.components == r2.components
    assert r1.details == r2.details
    assert set(r1.reasons) == set(r2.reasons)


# ── leakage: no outcome can reach the metric ─────────────────────────────────


def test_entry_points_take_only_features_and_a_config():
    assert list(inspect.signature(extract_features).parameters) == ["point", "config"]
    assert list(inspect.signature(distance).parameters) == ["a", "b", "config"]
    assert list(inspect.signature(select_pairs).parameters) == ["points", "config"]


def test_point_description_has_exactly_the_pre_experiment_fields():
    """Pinned so that adding a slot (an outcome, a history, a speedup) fails
    here and has to be argued for against §1."""
    assert [f.name for f in dataclasses.fields(PointDescription)] == [
        "name", "model", "model_revision", "engine_name", "engine_version", "planner", "regime",
        "engine", "library", "graph_at", "planner_version", "baseline_trace", "scheduler_stats",
        "capture_source",
    ]


@pytest.mark.parametrize("field", ["measured_delta", "speedup", "ab_result", "history", "prior_runs"])
def test_an_outcome_field_cannot_be_added_to_a_point(field):
    point = _point()
    kwargs = {f.name: getattr(point, f.name) for f in dataclasses.fields(PointDescription)}
    with pytest.raises(TypeError):
        PointDescription(**kwargs, **{field: 0.1})
    with pytest.raises(TypeError):
        dataclasses.replace(point, **{field: 0.1})
    with pytest.raises(AttributeError):
        setattr(point, field, 0.1)
    with pytest.raises(AttributeError):
        object.__setattr__(point, field, 0.1)


def test_outcomes_hung_on_the_engine_change_nothing():
    """The engine is the one ``Any`` slot. Features read only named knobs off it,
    so measured deltas, a history and run records attached to it have no effect."""

    @dataclasses.dataclass(frozen=True)
    class EngineWithOutcomes(SyntheticEngine):
        measured_delta: float = -0.5
        speedup: float = 0.5
        ab_result: tuple = (("cuda_graphs_enable", -0.4),)
        prior_runs: tuple = ("run-1", "run-2")
        history: object = History(records={("cuda_graphs_enable", "A100-SXM4-80GB", "synthetic"): LeverRecord(
            intervention_name="cuda_graphs_enable", gpu_sku="A100-SXM4-80GB", fingerprint="synthetic",
            runs=1, attempts=1, wins=0, losses=1, inconclusive=0, mean_delta=-0.4, best_delta=-0.4,
            worst_delta=-0.4, last_run_id="run-1")}, runs_read=1)

    clean = extract_features(_point(), CFG)
    dirty = extract_features(_point(engine=EngineWithOutcomes()), CFG)
    assert dirty == clean
    other = extract_features(synthetic_point(96, 4096, 512, seed=SEED), CFG)
    _same(distance(dirty, other, CFG), distance(clean, other, CFG))


def test_overwriting_priors_with_measured_looking_deltas_does_not_move_d():
    """S reads coverage via ``predict_delta(..., delta_mean=1.0)``, so the
    library's effect estimates never enter d, even if someone writes a
    measured delta into them. (They do drive baseline 3's cap, which is the
    design: the cap is a predicted-quantity rule.)"""
    lib = _point().library
    rng = random.Random(1)
    poisoned = tuple(s.model_copy(update={"expected_delta_mean": rng.uniform(-0.5, 0.5)}) for s in lib)
    pairs = [(synthetic_point(64, 4096, 512, seed=SEED, library=L), synthetic_point(96, 4096, 512, seed=SEED, library=L))
             for L in (lib, poisoned)]
    (a0, b0), (a1, b1) = [(extract_features(a, CFG), extract_features(b, CFG)) for a, b in pairs]
    _same(distance(a0, b0, CFG), distance(a1, b1, CFG))


def test_pair_selection_runs_with_every_outcome_source_disabled(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("an outcome source was consulted")

    for name in ("load_history", "record_for", "runs_with_results"):
        monkeypatch.setattr(history_mod, name, boom)
    monkeypatch.setattr("gitm.agents.policy.record_for", boom)
    for cls in (apply_mod.DryRunApplicator, apply_mod.DictApplicator,
                apply_mod.ConfigFileApplicator, apply_mod.LiveEngineApplicator):
        monkeypatch.setattr(cls, "measure", boom)

    feats = [extract_features(p, CFG) for p in synthetic_grid(CFG, seed=SEED)]
    report = select_pairs(feats, CFG)

    assert len(report.pairs) == 21
    assert {(p.a, p.b) for p in report.by_status(NEAR)} == DOC_NEAR


# ── distance: identity, symmetry, gates ──────────────────────────────────────


def test_distance_to_itself_is_zero(grid):
    for f in grid.values():
        r = distance(f, f, CFG)
        assert r.status == "ok"
        assert r.distance == 0.0
        assert r.setting_component is None
        assert all(v == 0.0 for _, v in r.components)


def test_distance_is_symmetric(grid, base):
    feats = list(grid.values())
    perturbed = [
        dataclasses.replace(base, name="p1", knobs=_knobs(base, max_num_batched_tokens=KnobRead(False))),
        dataclasses.replace(base, name="p2", residual=dataclasses.replace(
            base.residual, serialized_concurrency_fraction=0.3)),
        dataclasses.replace(base, name="p3", candidates=_coverage(base, "kv_cache_dtype_fp8", 0.2)),
        dataclasses.replace(base, name="p4", model="other"),
    ]
    for a in feats + perturbed:
        for b in feats + perturbed:
            _same(distance(a, b, CFG), distance(b, a, CFG))


GATE_VALUES = {
    "model": "other-model", "model_revision": "r2", "sku": "AMD Instinct MI355X", "num_gpus": 8,
    "tensor_parallel_size": 8, "engine_name": "sglang", "engine_version": "0.9", "dtype": "fp8",
    "workload": "other-workload",
}


@pytest.mark.parametrize("reason,attr", GATES)
def test_each_gate_makes_the_pair_ineligible_with_its_own_reason(base, reason, attr):
    other = dataclasses.replace(base, name="other", **{attr: GATE_VALUES[attr]})
    r = distance(base, other, CFG)
    assert r.status == INELIGIBLE
    assert r.distance == math.inf
    assert r.reasons == (reason,)
    assert r.components == ()
    assert r.setting_component is None


def test_an_open_loop_point_with_nothing_measured_is_not_bridgeable(base):
    point = _point(name="open")
    open_loop = dataclasses.replace(point, regime=point.regime.model_copy(update={"concurrency": None}))
    f = extract_features(open_loop, CFG)
    assert not f.bridgeable and f.batch is None and f.op_shares is None and f.residual is None
    assert distance(base, f, CFG).reasons == ("not_bridgeable:open:open_loop_unmeasured",)


def test_a_point_with_no_baseline_capture_is_ineligible(base):
    f = extract_features(dataclasses.replace(_point(name="nocap"), baseline_trace=None), CFG)
    assert f.basis == "none" and f.ranked == () and f.residual is None
    assert distance(base, f, CFG).reasons == ("no_baseline_capture:nocap",)


def test_different_planners_are_ineligible(base):
    other = dataclasses.replace(base, name="other", planner_version="registry v2")
    assert distance(base, other, CFG).reasons == ("cross_planner_version",)


def test_every_failing_gate_is_reported_in_doc_order(base):
    other = dataclasses.replace(base, name="other", sku="H100", model="m2", workload="w")
    assert distance(base, other, CFG).reasons == ("cross_model", "cross_sku", "cross_workload")


# ── the bridge (§2) ───────────────────────────────────────────────────────────


def test_bridge_uses_measured_mean_running_when_there_is_one():
    stats = SimpleNamespace(n_samples=12, mean_running=40.6)
    f = extract_features(dataclasses.replace(_point(), scheduler_stats=stats), CFG)
    assert (f.batch, f.kv_cache_len, f.batch_source) == (41, 4352, "running")  # the loop helper's own label


def test_bridge_caps_concurrency_at_max_num_seqs():
    f = extract_features(synthetic_point(96, 4096, 512, seed=SEED, engine=SyntheticEngine(max_num_seqs=48)), CFG)
    assert (f.batch, f.batch_source) == (48, "min(concurrency, max_num_seqs)")


def test_bridge_kv_is_mid_decode():
    f = extract_features(synthetic_point(64, 8192, 513, seed=SEED), CFG)
    assert f.kv_cache_len == 8192 + 513 // 2


# ── the seven components (§4) ─────────────────────────────────────────────────


def _knobs(f, **override):
    m = f.knob_map()
    m.update(override)
    return tuple(sorted(m.items()))


def _coverage(f, name, value):
    return tuple(dataclasses.replace(c, coverage=value) if c.name == name else c for c in f.candidates)


def _shift(shares, src, dst, step):
    d = dict(shares)
    d[src] -= step
    d[dst] = d.get(dst, 0.0) + step
    return tuple(sorted(d.items()))


def _move_o(f, step):
    return dataclasses.replace(f, batch=round(f.batch * (1 + step)))


def _move_t(f, step):
    t = dict(f.traffic)
    t["input_p95"] *= 1 + step
    return dataclasses.replace(f, traffic=tuple(sorted(t.items())))


def _move_k(f, step):
    v = f.knob_map()["max_num_batched_tokens"].value
    return dataclasses.replace(f, knobs=_knobs(f, max_num_batched_tokens=KnobRead(True, round(v * (1 + step)))))


def _move_g(f, step):
    return dataclasses.replace(f, op_shares=_shift(f.op_shares, "attn_score_value", "lm_head", step / 4))


def _move_b(f, step):
    return dataclasses.replace(f, bound_shares=_shift(f.bound_shares, "memory_bound", "compute_bound", step / 4))


def _move_s(f, step):
    cov = f.candidate_map()["kv_cache_dtype_fp8"].coverage
    return dataclasses.replace(f, candidates=_coverage(f, "kv_cache_dtype_fp8", cov - step / 4))


def _move_r(f, step):
    return dataclasses.replace(f, residual=dataclasses.replace(
        f.residual, serialized_concurrency_fraction=f.residual.serialized_concurrency_fraction + step / 4))


MOVERS = {"O": _move_o, "T": _move_t, "K": _move_k, "G": _move_g, "B": _move_b, "S": _move_s, "R": _move_r}


@pytest.mark.parametrize("component", COMPONENTS)
def test_each_component_grows_with_its_own_input_and_sets_d(base, component):
    move = MOVERS[component]
    small = distance(base, dataclasses.replace(move(base, 0.2), name="small"), CFG)
    large = distance(base, dataclasses.replace(move(base, 0.6), name="large"), CFG)
    for r in (small, large):
        assert r.status == "ok"
        assert r.setting_component == component
        assert r.distance == r.component(component)
        assert all(v == 0.0 for c, v in r.components if c != component)
    assert 0.0 < small.distance < large.distance <= 1.0


def test_component_values_follow_the_doc_scalings(base):
    assert distance(base, _move_o(base, 0.5), CFG).component("O") == pytest.approx(rel(64, 96))
    # T: 1 - 2^(-|log2 ratio|), which is rel() for a plain ratio axis.
    assert distance(base, _move_t(base, 1.0), CFG).component("T") == pytest.approx(0.5)
    # S: the mean over the common set, not the max.
    common = common_candidates(base, base, CFG).common
    assert distance(base, _move_s(base, 0.4), CFG).component("S") == pytest.approx(0.1 / len(common))
    assert distance(base, _move_g(base, 0.4), CFG).component("G") == pytest.approx(0.1)


def test_k_scores_unreadable_and_categorical_knobs_as_the_doc_says(base):
    one_side = dataclasses.replace(base, name="x", knobs=_knobs(base, block_size=KnobRead(False)))
    r = distance(base, one_side, CFG)
    assert (r.component("K"), dict(r.details)["K"]) == (1.0, "block_size")

    both = dataclasses.replace(base, name="y", knobs=_knobs(base, block_size=KnobRead(False)))
    assert distance(one_side, both, CFG).component("K") == 0.0

    categorical = dataclasses.replace(base, name="z", knobs=_knobs(base, kv_cache_dtype=KnobRead(True, "fp8")))
    assert distance(base, categorical, CFG).component("K") == 1.0


def test_the_largest_component_sets_d_on_the_grid(grid):
    r = distance(grid["c40_in4096_out512"], grid["c64_in8192_out512"], CFG)
    assert r.component("O") == pytest.approx(1 - 4352 / 8448)
    assert r.component("T") == 0.5
    assert (r.setting_component, dict(r.details)["T"], r.distance) == ("T", "input_p50", 0.5)


def test_rel_and_tv_edges():
    assert rel(0, 0) == 0.0 and rel(0, 5) == 1.0 and rel(-1, -2) == 1.0 and rel(2, 4) == 0.5
    assert tv([("a", 1.0)], [("b", 1.0)]) == 1.0
    assert tv([("a", 0.5), ("b", 0.5)], [("a", 0.5), ("b", 0.5)]) == 0.0


# ── the config drives behaviour ───────────────────────────────────────────────


def test_moving_the_thresholds_moves_pairs_between_bins(grid):
    default = select_pairs(list(grid.values()), CFG)
    wider = select_pairs(list(grid.values()), dataclasses.replace(CFG, near_threshold=0.4))
    assert (default.counts[NEAR], default.counts[INTERMEDIATE]) == (4, 3)
    assert (wider.counts[NEAR], wider.counts[INTERMEDIATE]) == (7, 0)
    moved = {(p.a, p.b) for p in wider.by_status(NEAR)} - {(p.a, p.b) for p in default.by_status(NEAR)}
    assert moved == {(p.a, p.b) for p in default.by_status(INTERMEDIATE)}

    stricter = select_pairs(list(grid.values()), dataclasses.replace(CFG, far_threshold=0.6))
    # Only the four pairs at d >= 0.625 stay far; nine at 0.5 and one at 0.583 move.
    assert stricter.counts[FAR] == 4
    assert stricter.counts[INTERMEDIATE] == default.counts[INTERMEDIATE] + 10


def test_changing_the_cap_changes_the_candidate_count(grid):
    a, b = grid[BASE], grid["c96_in4096_out512"]
    for cap, expected_status in ((8, "ok"), (6, "ok"), (5, "inconclusive")):
        cfg = dataclasses.replace(CFG, candidate_cap=cap)
        sets = common_candidates(a, b, cfg)
        assert len(a.capped(cfg)) == len(b.capped(cfg)) == len(sets.capped) == cap
        assert sets.status == expected_status
        # The uncapped set, which S and Kset use, does not move with the cap,
        # and neither does d.
        assert len(sets.common) == len(common_candidates(a, b, CFG).common)
        assert distance(a, b, cfg).distance == distance(a, b, CFG).distance


def test_features_extracted_under_a_different_config_are_refused(grid, base):
    other_policy = dataclasses.replace(CFG, skip_high_risk=False)
    with pytest.raises(ValueError, match="re-extract"):
        distance(base, grid["c96_in4096_out512"], other_policy)
    with pytest.raises(ValueError, match="re-extract"):
        select_pairs(list(grid.values()), dataclasses.replace(CFG, kset_base_knobs=("block_size",)))


def test_min_common_and_min_per_bin_come_from_the_config(grid):
    feats = list(grid.values())
    assert select_pairs(feats, CFG).bin_checks[0].status == INCONCLUSIVE
    assert select_pairs(feats, dataclasses.replace(CFG, min_pairs_per_bin=4)).status == "ok"
    strict = select_pairs(feats, dataclasses.replace(CFG, min_common_candidates=9))
    assert strict.counts[INCONCLUSIVE] == 21
    assert all(p.reasons == (COMMON_SET_BELOW_MINIMUM,) for p in strict.pairs)


def test_config_rejects_inconsistent_values():
    with pytest.raises(ValueError):
        DistanceConfig(near_threshold=0.5, far_threshold=0.25)
    with pytest.raises(ValueError):
        DistanceConfig(n_grid_points=6)
    with pytest.raises(dataclasses.FrozenInstanceError):
        CFG.near_threshold = 0.3


# ── candidate matching (§3) ───────────────────────────────────────────────────


def test_names_at_one_point_only_are_outside_the_intersection(base):
    """A zero current value collapses a relative lever's grid to one name, so
    ``_x2`` and ``_x4`` exist at one point only (vllm_knobs.py:317)."""
    other = extract_features(_point(name="collapsed",
                                    engine=SyntheticEngine(max_num_batched_tokens=0, gpu_memory_utilization=0.8)), CFG)
    sets = common_candidates(base, other, CFG)
    assert sets.only_a == ("max_num_batched_tokens_dynamic_x2", "max_num_batched_tokens_dynamic_x4")
    assert sets.only_b == ()
    assert not set(sets.only_a) & set(sets.common)
    whole = common_candidates(base, base, CFG)
    # Two names are outside the intersection; the rest of the count follows the catalog.
    assert sets.sizes["common"] == whole.sizes["common"] - 2
    assert sets.sizes["capped"] == 6
    assert 0 < sets.sizes["same_values"] <= sets.sizes["common"]
    # Same name, different resolved value: in the common set, out of the sensitivity subset.
    # Which levers survive the cap follows the catalog's ranking, so the test pins the
    # property: only relative levers resolve to different values at the two points.
    differing = set(sets.capped) - set(sets.same_values)
    assert differing, "the collapsed point should change some resolved values"
    assert all("_dynamic_x" in n for n in differing), differing


def test_vetoed_candidates_leave_the_common_set_with_a_reason(base):
    @dataclasses.dataclass(frozen=True)
    class NoRestartHook(SyntheticEngine):
        gitm_restart_fn: object = None

    other = extract_features(_point(name="nohook", engine=NoRestartHook()), CFG)
    sets = common_candidates(base, other, CFG)
    assert sets.common == () and sets.capped == ()
    assert (sets.status, sets.reason) == (INCONCLUSIVE, COMMON_SET_BELOW_MINIMUM)
    assert ("cuda_graphs_enable", "nohook", "structural knob 'enforce_eager' and no restart hook") in sets.not_runnable
    r = distance(base, other, CFG)
    assert (r.status, r.distance, r.reasons, r.component("S")) == ("inconclusive", None, ("no_common_candidates",), None)


def test_a_small_common_set_is_inconclusive_not_dropped(grid):
    report = select_pairs(list(grid.values()), dataclasses.replace(CFG, candidate_cap=5))
    assert len(report.pairs) == 21
    assert report.counts[INCONCLUSIVE] == 21
    assert all(p.reasons == (COMMON_SET_BELOW_MINIMUM,) and p.distance.status == "ok" for p in report.pairs)


# ── rank agreement (§6) ───────────────────────────────────────────────────────


def _r(*scores):
    return [(f"c{i}", s) for i, s in enumerate(scores)]


def test_tau_perfect_agreement_and_reversal():
    assert ag.kendall_tau_b(_r(1, 2, 3, 4), _r(10, 20, 30, 40)).tau == 1.0
    assert ag.kendall_tau_b(_r(1, 2, 3, 4), _r(4, 3, 2, 1)).tau == -1.0
    # Order of the lists is irrelevant; only the scores are.
    assert ag.kendall_tau_b(list(reversed(_r(1, 2, 3))), _r(1, 2, 3)).tau == 1.0


def test_tau_with_ties_matches_the_hand_computed_value():
    # a: 1 1 2 3, b: 1 2 2 3 -> C=4, D=0, n0=6, n1=1, n2=1 -> 4/5.
    r = ag.kendall_tau_b(_r(1, 1, 2, 3), _r(1, 2, 2, 3))
    assert (r.concordant, r.discordant, r.tied_pairs_a, r.tied_pairs_b) == (4, 0, 1, 1)
    assert r.tau == pytest.approx(0.8)


@pytest.mark.parametrize("a,b,reason", [
    ([], [], ag.TOO_FEW_CANDIDATES),
    (_r(1.0), _r(2.0), ag.TOO_FEW_CANDIDATES),
    (_r(1, 1, 1), _r(2, 2, 2), ag.ALL_TIED),
    (_r(1, 1, 1), _r(1, 2, 3), ag.CONSTANT_RANKING),
    (_r(1, 2, 3), _r(5, 5, 5), ag.CONSTANT_RANKING),
])
def test_tau_is_inconclusive_rather_than_nan(a, b, reason):
    r = ag.kendall_tau_b(a, b)
    assert (r.status, r.tau, r.reason) == (ag.INCONCLUSIVE, None, reason)


def test_tau_matches_scipy_on_random_rankings_with_ties():
    rng = random.Random(20261012)
    checked = 0
    for _ in range(500):
        n = rng.randint(2, 12)
        a = [float(rng.randint(0, 4)) for _ in range(n)]
        b = [float(rng.randint(0, 4)) for _ in range(n)]
        ours = ag.kendall_tau_b(_r(*a), _r(*b))
        ref = kendalltau(a, b, variant="b").statistic
        if math.isnan(ref):
            assert ours.status == ag.INCONCLUSIVE and ours.reason in (ag.ALL_TIED, ag.CONSTANT_RANKING)
        else:
            assert ours.tau == pytest.approx(ref, abs=1e-12)
            checked += 1
    assert checked > 400


def test_tau_refuses_mismatched_or_malformed_rankings():
    with pytest.raises(ValueError, match="different candidates"):
        ag.kendall_tau_b(_r(1, 2), [("c0", 1), ("other", 2)])
    with pytest.raises(ValueError, match="twice"):
        ag.kendall_tau_b([("x", 1), ("x", 2)], [("x", 1), ("x", 2)])
    with pytest.raises(ValueError, match="non-finite"):
        ag.kendall_tau_b(_r(1, float("nan")), _r(1, 2))


def _iv(**pairs):
    return {tuple(k.split("_")): v for k, v in pairs.items()}


def test_resolved_concordance_counts_only_pairs_resolved_at_both_points():
    cfg = dataclasses.replace(CFG, min_resolved_pairs=2)
    names = [("x", 0.0), ("y", 0.0), ("z", 0.0)]
    a = _iv(x_y=(0.1, 0.3), x_z=(0.2, 0.4), y_z=(-0.1, 0.1))
    # y_z given in the other orientation: (z - y) in [0.05, 0.2] means y - z in [-0.2, -0.05].
    b = {("x", "y"): (0.1, 0.2), ("z", "x"): (0.1, 0.3), ("z", "y"): (0.05, 0.2)}
    r = ag.resolved_concordance(names, names, a, b, cfg)
    # x-y: + and + (concordant); x-z: + and - (discordant); y-z: tied at a.
    assert (r.status, r.n_pairs, r.n_resolved, r.n_concordant, r.concordance) == (ag.OK, 3, 2, 1, 0.5)
    assert (r.tied_fraction_a, r.tied_fraction_b) == (pytest.approx(1 / 3), 0.0)


def test_resolved_concordance_flags_few_resolved_pairs_and_needs_a_threshold():
    names = [("x", 0.0), ("y", 0.0)]
    iv = {("x", "y"): (-0.1, 0.1)}
    with pytest.raises(ValueError, match="min_resolved_pairs"):
        ag.resolved_concordance(names, names, iv, iv, CFG)
    r = ag.resolved_concordance(names, names, iv, iv, dataclasses.replace(CFG, min_resolved_pairs=1))
    assert (r.status, r.reason, r.concordance, r.n_resolved) == (ag.INCONCLUSIVE, ag.FEW_RESOLVED_PAIRS, None, 0)
    with pytest.raises(ValueError, match="no difference interval"):
        ag.resolved_concordance(names, names, {}, iv, dataclasses.replace(CFG, min_resolved_pairs=1))


# ── pair selection (§7.1-§7.2) ────────────────────────────────────────────────


def test_pairs_are_binned_by_the_thresholds_inclusive_at_both_ends(grid):
    report = select_pairs(list(grid.values()), CFG)
    assert report.counts == {NEAR: 4, INTERMEDIATE: 3, FAR: 14, INCONCLUSIVE: 0, INELIGIBLE: 0}
    assert {(p.a, p.b) for p in report.by_status(NEAR)} == DOC_NEAR
    for p in report.pairs:
        d = p.distance.distance
        assert p.status == (NEAR if d <= 0.25 else FAR if d >= 0.5 else INTERMEDIATE)
    # d == 0.25 exactly is near and d == 0.5 exactly is far (§7.2's <= and >=).
    assert any(p.distance.distance == 0.25 and p.status == NEAR for p in report.pairs)
    assert any(p.distance.distance == 0.5 and p.status == FAR for p in report.pairs)
    assert all(p.reasons == ("middle_band_unused",) for p in report.by_status(INTERMEDIATE))


def test_under_filled_bins_are_reported_inconclusive(grid):
    report = select_pairs(list(grid.values()), CFG)
    near, far = report.bin_checks
    assert (near.bin, near.n, near.required, near.status, near.reason) == (NEAR, 4, 6, INCONCLUSIVE, "near_bin_underfilled")
    assert (far.bin, far.n, far.status) == (FAR, 14, "ok")
    assert report.status == INCONCLUSIVE


def test_ineligible_pairs_are_kept_in_the_report_with_their_reason(grid, base):
    alien = dataclasses.replace(base, name="zz_h100", sku="H100")
    report = select_pairs([*grid.values(), alien], CFG)
    assert len(report.pairs) == 28
    bad = report.by_status(INELIGIBLE)
    assert len(bad) == 7 and all(p.reasons == ("cross_sku",) for p in bad)
    assert {p.b for p in bad} == {"zz_h100"}
    assert set(bad) <= set(report.excluded)


def test_duplicate_point_names_are_refused(base):
    with pytest.raises(ValueError, match="unique"):
        select_pairs([base, base], CFG)


def test_features_are_hashable_and_deterministic(grid):
    again = {f.name: f for f in (extract_features(p, CFG) for p in synthetic_grid(CFG, seed=SEED))}
    assert again == grid
    assert len({hash(f) for f in grid.values()}) == len(grid)


# ── the local runner ──────────────────────────────────────────────────────────


def _run_local(*args: str) -> str:
    out = subprocess.run([sys.executable, "-m", "gitm.research.distance_metric.run_local", *args],
                         capture_output=True, text=True, check=True)
    return out.stdout


def test_run_local_is_deterministic_and_says_it_is_not_a_validation():
    first, second = _run_local(), _run_local()
    assert first == second
    assert first.count("NOT A VALIDATION RESULT") == 2
    assert "seed: 0" in first
    assert "near: 4 / 6 required -> inconclusive (near_bin_underfilled)" in first
    assert "seed: 7" in _run_local("--seed", "7")
