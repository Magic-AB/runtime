"""The ranked list the fast loop consumes (gitm.agents.ranked_list, ENG-97).

Levers come from the real library; the class and restart-cost fields come from a small
yaml written here, because ``library.yaml`` has neither until ENG-79 lands. Synthetic
rankings only: nothing here reads live telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gitm.agents.policy import RankedCandidate
from gitm.agents.ranked_list import RESTART_SECONDS, rank_for_loop, read_lever_meta
from gitm.kernels.library import load_library

LIBRARY_YAML = Path(__file__).resolve().parents[1] / "gitm/kernels/library.yaml"


@pytest.fixture(scope="module")
def specs():
    return {s.name: s for s in load_library()}


def _meta_file(tmp_path, rows):
    p = tmp_path / "meta.yaml"
    p.write_text(yaml.safe_dump({"interventions": rows}))
    return p


def _rc(specs, name, delta, **kw):
    return RankedCandidate(spec=specs[name], predicted_delta=delta, **kw)


@pytest.fixture
def three(specs):
    names = list(specs)[:3]
    return names, [_rc(specs, n, d) for n, d in zip(names, (0.10, 0.08, 0.05), strict=True)]


def _meta(tmp_path, names, classes=(1, 1, 1), costs=("none", "restart", "rebuild")):
    rows = [{"name": n, "class": c, "restart_cost": r, "expected_delta_lo": 0.01,
             "expected_delta_hi": 0.2, "expected_delta_mean": 0.1}
            for n, c, r in zip(names, classes, costs, strict=True)]
    return read_lever_meta(_meta_file(tmp_path, rows))


def test_the_real_library_has_no_class_or_restart_cost_yet():
    meta = read_lever_meta(LIBRARY_YAML)
    assert meta, "library.yaml read nothing"
    assert all("class_missing" in m.problems and "restart_cost_missing" in m.problems
               for m in meta.values()), "ENG-79 landed: update this test and drop strict=False uses"


def test_strict_leaves_out_levers_with_no_class_and_says_so(specs):
    meta = read_lever_meta(LIBRARY_YAML)
    ranked = [_rc(specs, n, 0.05) for n in list(specs)[:4]]
    out = rank_for_loop(ranked, meta)
    assert out.entries == []
    assert {r for _, r in out.excluded} == {"class_missing,restart_cost_missing"}


def test_not_strict_keeps_them_marked_unknown(specs):
    meta = read_lever_meta(LIBRARY_YAML)
    out = rank_for_loop([_rc(specs, n, 0.05) for n in list(specs)[:3]], meta, strict=False)
    assert len(out.entries) == 3
    assert all(e.lever_class is None and e.restart_cost is None for e in out.entries)


def test_order_is_the_rankings_unless_cost_weighted(three, tmp_path):
    names, ranked = three
    out = rank_for_loop(ranked, _meta(tmp_path, names))
    assert out.top == names
    assert [e.rank for e in out.entries] == [1, 2, 3]
    assert [e.try_cost_s for e in out.entries] == [1.0, 60.0, 600.0]


def test_cost_weighted_prefers_cheap_tries_and_demoted_goes_last(specs, tmp_path):
    a, b, c = list(specs)[:3]
    ranked = [_rc(specs, a, 0.10), _rc(specs, b, 0.08), _rc(specs, c, 0.20, demoted=True)]
    meta = _meta(tmp_path, [a, b, c], costs=("rebuild", "none", "none"))
    out = rank_for_loop(ranked, meta, cost_weighted=True)
    assert out.top == [b, a, c]


def test_class_two_is_out_of_the_mvp(three, tmp_path):
    names, ranked = three
    out = rank_for_loop(ranked, _meta(tmp_path, names, classes=(1, 2, 3)))
    assert out.top == names[:1]
    assert out.excluded == [(names[1], "class_2_outside_mvp"), (names[2], "class_3_outside_mvp")]
    assert rank_for_loop(ranked, _meta(tmp_path, names, classes=(1, 2, 3)),
                         mvp_classes=(1, 2)).top == names[:2]


def test_rejected_and_noop_levers_are_left_out_with_their_reason(specs, tmp_path):
    a, b, c = list(specs)[:3]
    ranked = [_rc(specs, a, 0.1), _rc(specs, b, 0.1, rejected_reason="oom_history"),
              _rc(specs, c, 0.1, baseline_noop="already set")]
    out = rank_for_loop(ranked, _meta(tmp_path, [a, b, c]))
    assert out.top == [a]
    assert dict(out.excluded) == {b: "rejected:oom_history", c: "baseline_noop:already set"}


def test_a_lever_the_yaml_does_not_name_is_left_out(three, tmp_path):
    names, ranked = three
    meta = _meta(tmp_path, names[:2], costs=("none", "none"), classes=(1, 1))
    assert ("not_in_meta" in dict(rank_for_loop(ranked, meta).excluded).values())


def test_each_entry_says_what_put_it_there(specs, tmp_path):
    a = list(specs)[0]
    cand = _rc(specs, a, 0.12, delta_source="measured", motivated_by="kv_cache_preemption",
               targets=(("attn_score_value", 0.42),))
    out = rank_for_loop([cand], _meta(tmp_path, [a], classes=(1,), costs=("none",)))
    why = " | ".join(out.entries[0].why)
    assert "measured" in why and "kv_cache_preemption" in why and "attn_score_value" in why


def test_top_n_truncates_the_list(three, tmp_path):
    names, ranked = three
    assert rank_for_loop(ranked, _meta(tmp_path, names), top_n=2).top == names[:2]


@pytest.mark.parametrize("field, bad, reason", [
    ("class", 4, "class_invalid:4"), ("class", True, "class_invalid:True"),
    ("restart_cost", "slow", "restart_cost_invalid:'slow'"),
])
def test_invalid_fields_are_recorded_not_raised(tmp_path, field, bad, reason):
    row = {"name": "x", "class": 1, "restart_cost": "none", field: bad}
    meta = read_lever_meta(_meta_file(tmp_path, [row]))
    assert meta["x"].problems == (reason,)


def test_a_file_that_is_not_a_library_is_refused(tmp_path):
    with pytest.raises(ValueError, match="interventions"):
        read_lever_meta(_meta_file(tmp_path, None))
    with pytest.raises(ValueError, match="twice"):
        read_lever_meta(_meta_file(tmp_path, [{"name": "a"}, {"name": "a"}]))


def test_the_output_is_json_and_versioned(three, tmp_path):
    names, ranked = three
    d = json.loads(rank_for_loop(ranked, _meta(tmp_path, names)).to_json())
    assert d["schema_version"] == 1
    assert [e["name"] for e in d["entries"]] == names
    assert set(RESTART_SECONDS) == {"none", "restart", "rebuild"}
