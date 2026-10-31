"""The request-shape ring (gitm.traffic.ring, ENG-189)."""

from __future__ import annotations

import threading

import pytest

from gitm.traffic.ring import BYTES_PER_SLOT, RequestRing, main, prefix_hash
from gitm.traffic.shift import TrafficProfile


def test_it_keeps_the_most_recent_capacity_requests_oldest_first():
    ring = RequestRing(4)
    for i in range(10):
        ring.record(float(i), 100 + i, 10 + i, i)
    assert len(ring) == 4
    assert [r[0] for r in ring.snapshot()] == [6.0, 7.0, 8.0, 9.0]
    assert ring.snapshot()[0] == (6.0, 106, 16, 6)


def test_memory_is_fixed_at_construction():
    assert RequestRing(1000).nbytes == 1000 * BYTES_PER_SLOT == 24_000


def test_a_slice_is_a_canonical_trace_rebased_to_the_window():
    ring = RequestRing(100)
    for i in range(20):
        ring.record(10.0 + i, 200, 50, 7)
    t = ring.slice(15.0, 25.0)
    assert len(t) == 10 and t.requests[0].arrival_s == 0.0
    assert t.meta.span_s == 10.0 and t.meta.source == "traffic-monitor"
    assert t.requests[0].prefix_blocks == (7,) and t.meta.has_prefix_identity
    with pytest.raises(ValueError):
        ring.slice(5.0, 5.0)


def test_a_request_with_no_recorded_output_has_no_output_length():
    ring = RequestRing(10)
    ring.record(0.0, 300)
    assert ring.slice(0.0, 1.0).requests[0].output_tokens is None


def test_a_slice_feeds_the_traffic_shift_profile():
    ring = RequestRing(5000)
    for i in range(1000):
        ring.record(i * 0.1, 500 + i % 50, 100 + i % 20, 0)
    p = TrafficProfile.from_trace(ring.slice(0.0, 100.0))
    assert p.requests == 1000 and p.span_s == 100.0


def test_concurrent_recording_loses_nothing():
    ring = RequestRing(100_000)

    def work():
        for j in range(5000):
            ring.record(float(j), 10, 10, 1)

    ts = [threading.Thread(target=work) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(ring) == 40_000


def test_prefix_hash_depends_only_on_the_leading_block():
    a = list(range(1000, 1600))
    assert prefix_hash(a) == prefix_hash(a[:512] + [5, 6, 7])  # same first 512, different tail
    assert prefix_hash(a) != prefix_hash([1] + a[1:])  # differs inside the first block


def test_bad_capacity_is_refused_and_the_benchmark_runs(capsys):
    with pytest.raises(ValueError):
        RequestRing(0)
    assert main(["--n", "2000"]) == 0
    out = capsys.readouterr().out
    assert "record():" in out and "prefix_hash():" in out and "ring memory" in out
