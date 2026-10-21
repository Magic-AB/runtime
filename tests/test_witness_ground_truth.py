"""Witness against the time its own fixture injected (synthetic; exercises arithmetic only).

A refutation-style check: the generator knows how much sync, memcpy and collective time
it put in. A confidently wrong witness would say a cause is absent when it is there, or
report seconds far from what was injected. Passing here says the arithmetic holds on
non-overlapping synthetic records. It says nothing about real captures.
"""

from __future__ import annotations

import pytest

from gitm.optimizer import witness
from gitm.optimizer import witness_fixtures as wf

Inj = wf.Injection
_ROWS = {"10": "gap time under a synchronization record",
         "11": "gap time under a memcpy record",
         "22": "exposed collective communication"}

CASES = {
    "sync_only": dict(sync=Inj(0.5, 200_000)),
    "memcpy_only": dict(memcpy=Inj(0.5, 200_000)),
    "sync_and_memcpy": dict(sync=Inj(0.5, 200_000), memcpy=Inj(0.5, 200_000)),
    "collective_only": dict(all_reduce=Inj(0.5, 300_000)),
    "none": dict(),
}


def _rows(tmp_path, seed, kw):
    fx = wf.generate(seed, window_s=10.0, **kw)
    report = witness.run(wf.write_run_dir(tmp_path, fx)).to_dict()
    return fx, {r["name"]: r for r in report["results"] if r["scope"] in ("partition", "device")}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("case", CASES)
def test_injected_causes_are_never_absent_and_seconds_match(tmp_path, case, seed):
    fx, rows = _rows(tmp_path, seed, CASES[case])
    for row_id, name in _ROWS.items():
        injected = fx.realised_ns[row_id] / 1e9
        r = rows[name]
        if injected > 0:
            assert r["state"] == "fired", (case, name, r["state"])
            assert r["status"] != "absent"
            assert r["seconds"] == pytest.approx(injected, rel=1e-3), (case, name)
        else:
            assert r["state"] != "fired", (case, name, "fired with nothing injected")
            assert not r["seconds"], (case, name, r["seconds"])


def test_time_goes_on_the_row_that_was_injected(tmp_path):
    fx, rows = _rows(tmp_path, 1, CASES["memcpy_only"])
    assert rows[_ROWS["10"]]["state"] != "fired"
    assert rows[_ROWS["11"]]["state"] == "fired"
