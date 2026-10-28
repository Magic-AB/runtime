"""Synthetic varying traffic (gitm.traffic.varying, ENG-178), scored by ENG-98's metric.

Everything here is synthetic. Nothing was replayed on an engine; the tests check that the
trace is well formed and that the traffic-distance metric calls the changing stretches
different and the matching ones the same.
"""

from __future__ import annotations

import pytest

from gitm.traffic import write_timed_trace
from gitm.traffic.shift import TrafficProfile, detect_shift
from gitm.traffic.varying import DEFAULT_PHASES, Phase, slice_trace, varying_trace

WINDOW_S = 100.0
SEEDS = range(20)
PHASE_S = 120


def _window(trace, phase_index):
    start = phase_index * PHASE_S
    return TrafficProfile.from_trace(slice_trace(trace, start, start + WINDOW_S))


def _verdicts(trace):
    """Each phase's window against the first phase's window (the baseline)."""
    base = _window(trace, 0)
    return {i: detect_shift(base, _window(trace, i)) for i in range(1, 4)}


def test_same_seed_same_trace_and_a_different_seed_differs():
    a, b, c = varying_trace(seed=3), varying_trace(seed=3), varying_trace(seed=4)
    assert a.requests == b.requests and a.meta.sha256 == b.meta.sha256
    assert a.requests != c.requests and a.meta.sha256 != c.meta.sha256


def test_trace_is_well_formed_and_reconciles():
    t = varying_trace(seed=0)
    arr = t.arrivals
    assert arr[0] == 0.0 and arr == sorted(arr)
    assert len(t) == t.meta.rows_emitted == t.meta.rows_read
    assert all(r.input_tokens >= 1 and r.output_tokens >= 1 for r in t.requests)
    assert t.meta.source == "synthetic-varying"
    assert any("synthetic" in n for n in t.meta.notes)


def test_phases_have_the_rates_and_lengths_they_declare():
    t = varying_trace(seed=0)
    quiet = slice_trace(t, 0, 120)
    burst = slice_trace(t, 120, 240)
    long_ = slice_trace(t, 240, 360)
    assert 3.0 < quiet.rate_rps() < 5.0
    assert 9.0 < burst.rate_rps() < 15.0
    med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    assert med(long_.input_tokens) > 3 * med(quiet.input_tokens)
    assert med(long_.output_tokens) > 2 * med(quiet.output_tokens)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_metric_calls_the_burst_and_long_prompt_stretches_different(seed):
    v = _verdicts(varying_trace(seed=seed))
    for phase in (1, 2):  # the burst, then the long mixed prompts
        assert v[phase].shifted, (seed, phase, v[phase])


def test_the_metric_calls_matching_stretches_the_same_across_seeds():
    """The return to the baseline's own phase, and an independent quiet stretch."""
    fired = total = 0
    for seed in SEEDS:
        a, other = varying_trace(seed=seed), varying_trace(seed=seed + 100)
        base = _window(a, 0)
        for window in (_window(a, 3), _window(other, 0)):
            total += 1
            fired += detect_shift(base, window).shifted
    assert fired <= 0.05 * total, f"{fired}/{total} matching windows were called shifted"


def test_a_clumpy_window_at_an_unchanged_rate_is_not_called_a_rate_shift():
    """Same rate and lengths, but the window's arrivals come in clumps.

    The window's request count is far noisier than Poisson, so a count that happens to land
    high must not read as a rate shift. (shift.py does not compare burstiness itself; the
    default burst phase is caught by its 3x rate.)
    """
    phases = (
        Phase("smooth", 120, rate_rps=6.0, input_median=600),
        Phase("clumped", 120, rate_rps=6.0, input_median=600, burst_size=30, burst_spread_s=1.0),
    )
    fired, n = 0, 100
    for seed in range(n):
        t = varying_trace(seed=seed, phases=phases)
        base, win = _window(t, 0), _window(t, 1)
        assert win.dispersion > 3 * base.dispersion  # it really is clumpier
        fired += detect_shift(base, win).shifted
    # Measured: 16/100 when the noise used the baseline's dispersion for both sides, 4/100 now.
    assert fired <= 8, f"{fired}/{n} clumpy windows were called shifted"


def test_slice_trace_rebases_and_uses_the_nominal_window_as_its_span():
    t = varying_trace(seed=0)
    s = slice_trace(t, 100, 160)
    assert s.meta.span_s == 60.0
    assert all(0.0 <= r.arrival_s < 60.0 for r in s.requests)
    assert len(s) == sum(1 for r in t.requests if 100 <= r.arrival_s < 160)
    with pytest.raises(ValueError):
        slice_trace(t, 10, 10)


@pytest.mark.parametrize("bad", [
    Phase("x", 0, rate_rps=1, input_median=10),
    Phase("x", 10, rate_rps=0, input_median=10),
    Phase("x", 10, rate_rps=1, input_median=0),
    Phase("x", 10, rate_rps=1, input_median=10, burst_size=0),
])
def test_bad_phases_are_refused(bad):
    with pytest.raises(ValueError):
        varying_trace(phases=(bad,))
    with pytest.raises(ValueError):
        varying_trace(phases=())


def test_the_trace_writes_as_a_replayable_timed_trace(tmp_path):
    t = varying_trace(seed=0, phases=DEFAULT_PHASES[:2])
    out = tmp_path / "varying.jsonl"
    plan = write_timed_trace(t, out)
    assert out.exists() and out.stat().st_size > 0
    assert plan is not None
