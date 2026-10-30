"""Smooth/bursty pairs (gitm.traffic.arrival_pair, ENG-153): same requests, different timing."""

from __future__ import annotations

import pytest

from gitm.traffic.arrival_pair import arrival_pair, main
from gitm.traffic.regime import index_of_dispersion


def _disp(tr):
    return index_of_dispersion(tr.arrivals, bin_s=1.0, span_s=tr.meta.span_s)


@pytest.mark.parametrize("rate", [2.0, 4.0, 8.0])
def test_the_pair_has_the_same_requests_and_only_the_timing_differs(rate):
    smooth, bursty = arrival_pair(rate_rps=rate, duration_s=300, seed=1)
    assert len(smooth) == len(bursty) == round(rate * 300)
    assert sorted(smooth.input_tokens) == sorted(bursty.input_tokens)
    assert sorted(smooth.output_tokens) == sorted(bursty.output_tokens)
    assert smooth.arrivals != bursty.arrivals
    assert smooth.meta.span_s == pytest.approx(300.0) == pytest.approx(bursty.meta.span_s)
    assert smooth.rate_rps() == pytest.approx(bursty.rate_rps())


@pytest.mark.parametrize("seed", range(5))
def test_bursty_is_much_clumpier_and_smooth_is_near_poisson(seed):
    smooth, bursty = arrival_pair(rate_rps=4.0, duration_s=300, seed=seed)
    assert 0.6 < _disp(smooth) < 1.6
    assert _disp(bursty) > 5 * _disp(smooth)


def test_same_seed_same_pair_and_different_patterns_have_different_hashes():
    a, b = arrival_pair(rate_rps=4, duration_s=120, seed=2), arrival_pair(
        rate_rps=4, duration_s=120, seed=2)
    assert a[0].requests == b[0].requests and a[1].requests == b[1].requests
    assert a[0].meta.sha256 != a[1].meta.sha256


@pytest.mark.parametrize("kw", [
    dict(rate_rps=0, duration_s=100), dict(rate_rps=4, duration_s=0),
    dict(rate_rps=4, duration_s=100, burst_size=1),
    dict(rate_rps=4, duration_s=100, burst_spread_s=100),
    dict(rate_rps=0.1, duration_s=100),  # too few requests for bursts of 24
])
def test_bad_arguments_are_refused(kw):
    with pytest.raises(ValueError):
        arrival_pair(**kw)


def test_the_command_writes_both_files_and_prints_both_bench_commands(tmp_path, capsys):
    prefix = tmp_path / "p"
    assert main(["--rate", "4", "--duration", "120", "--out-prefix", str(prefix),
                 "--model", "some/model"]) == 0
    text = capsys.readouterr().out
    assert (tmp_path / "p_smooth.jsonl").exists() and (tmp_path / "p_bursty.jsonl").exists()
    assert text.count("vllm bench serve") == 2 and "not run" in text


@pytest.mark.parametrize("seed", range(10))
def test_both_traces_survive_the_timed_trace_round_trip(tmp_path, seed):
    from gitm.traffic.replay import read_timed_trace, write_timed_trace
    from gitm.traffic.validate import REPLAY_THRESHOLDS, compare

    for tr in arrival_pair(rate_rps=4, duration_s=300, seed=seed):
        out = tmp_path / f"{tr.meta.source}.jsonl"
        write_timed_trace(tr, out)
        report = compare(tr, read_timed_trace(out), thresholds=REPLAY_THRESHOLDS)
        assert report.passed, report.render()
