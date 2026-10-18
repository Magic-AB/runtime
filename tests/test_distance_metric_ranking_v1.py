"""Ranking v1 on synthetic points only. The deltas below are invented to make
the transfer visible; nothing here is a measurement."""

from __future__ import annotations

import dataclasses

import pytest

from gitm.agents.policy import Policy
from gitm.research.distance_metric.baselines import catalog_predict_delta_ranking
from gitm.research.distance_metric.config import DistanceConfig
from gitm.research.distance_metric.features import extract_features
from gitm.research.distance_metric.ranking_v1 import (
    FALLBACK, TRANSFERRED, SourceMeasurement, rank_v1,
)
from gitm.research.distance_metric.synthetic import synthetic_point

CFG = DistanceConfig()
SEED = 0
POLICY = Policy(skip_high_risk=True)


def _pt(c, i=4096, o=512, **kw):
    return synthetic_point(c, i, o, seed=SEED, **kw)


@pytest.fixture(scope="module")
def world():
    target = _pt(40)
    lib = list(target.library)
    feats = lambda p: extract_features(p, CFG)  # noqa: E731
    base = catalog_predict_delta_ranking(target.baseline_trace, lib, POLICY, ctx=target.planner.gate)
    live = [c.spec.name for c in base if c.rejected_reason is None]
    assert len(live) >= 4
    return target, lib, feats, live


def _call(world, sources, max_d=0.25, **kw):
    target, lib, feats, _ = world
    return rank_v1(feats(target), sources, target.baseline_trace, lib, POLICY, CFG,
                   max_d=max_d, ctx=target.planner.gate, gpu_sku=target.planner.sku,
                   fingerprint=None, **kw)


def test_a_near_source_reorders_the_ranking_by_its_measured_deltas(world):
    target, lib, feats, live = world
    last = live[-1]
    src = SourceMeasurement(feats(_pt(32)), {last: 0.40, live[0]: -0.20})

    out = _call(world, [src])

    assert out.mode == TRANSFERRED and out.source == "c32_in4096_out512"
    assert out.distance == pytest.approx(0.2) and out.setting_component == "O"
    assert out.transferred == tuple(sorted([last, live[0]]))
    names = [c.spec.name for c in out.ranking]
    assert names[0] == last  # +40% measured beats every prior
    assert names.index(live[0]) > names.index(last)
    assert len(names) == len(lib)


def test_far_sources_fall_back_to_the_catalog_ranking_unchanged(world):
    target, lib, feats, live = world
    src = SourceMeasurement(feats(_pt(128)), {live[-1]: 0.9})  # d = 0.6875 from c40

    out = _call(world, [src])

    base = catalog_predict_delta_ranking(target.baseline_trace, lib, POLICY, ctx=target.planner.gate)
    assert out.mode == FALLBACK and out.source is None and out.transferred == ()
    assert [(c.spec.name, c.predicted_delta) for c in out.ranking] == [
        (c.spec.name, c.predicted_delta) for c in base]
    assert "no source within max_d=0.25" in out.reason and "c128_in4096_out512" in out.reason


def test_the_nearest_eligible_source_is_used(world):
    target, lib, feats, live = world
    near = SourceMeasurement(feats(_pt(48)), {live[-1]: 0.3})   # d = 1/6
    farther = SourceMeasurement(feats(_pt(32)), {live[-2]: 0.3})  # d = 0.2

    out = _call(world, [farther, near])

    assert out.source == "c48_in4096_out512"
    assert [c.name for c in out.considered] == ["c32_in4096_out512", "c48_in4096_out512"]


def test_an_ineligible_source_is_never_used_and_is_reported(world):
    target, lib, feats, live = world
    bad = dataclasses.replace(feats(_pt(48)), name="other_model", model="m2")

    out = _call(world, [SourceMeasurement(bad, {live[-1]: 0.5})], max_d=1.0)

    assert out.mode == FALLBACK
    assert out.considered[0].status == "ineligible" and out.considered[0].reasons == ("cross_model",)


def test_the_target_cannot_be_its_own_source(world):
    target, lib, feats, live = world
    with pytest.raises(ValueError, match="is the target"):
        _call(world, [SourceMeasurement(feats(target), {live[0]: 0.1})])


def test_max_d_is_required_and_bounded(world):
    target, lib, feats, live = world
    with pytest.raises(TypeError):
        rank_v1(feats(target), [], target.baseline_trace, lib, POLICY, CFG)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        _call(world, [], max_d=1.5)


def test_no_sources_is_a_stated_fallback(world):
    out = _call(world, [])
    assert out.mode == FALLBACK and out.reason == "no sources given"


def test_source_deltas_for_unknown_candidates_are_ignored(world):
    target, lib, feats, live = world
    out = _call(world, [SourceMeasurement(feats(_pt(32)), {"not_in_library": 0.9, live[1]: 0.1})])
    assert out.transferred == (live[1],)


def test_inputs_are_not_mutated_and_repeat_calls_agree(world):
    target, lib, feats, live = world
    deltas = {live[-1]: 0.4}
    src = SourceMeasurement(feats(_pt(32)), deltas)
    first = _call(world, [src])
    second = _call(world, [src])
    assert deltas == {live[-1]: 0.4}
    assert [(c.spec.name, c.predicted_delta) for c in first.ranking] == [
        (c.spec.name, c.predicted_delta) for c in second.ranking]
