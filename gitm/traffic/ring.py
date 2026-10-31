"""A fixed-size ring of request shapes, and what it costs to record one (ENG-189).

The traffic monitor keeps a rolling buffer of request *shapes* (prompt and output lengths, arrival
time, a hash of the prompt's leading block), never prompts or traces, and saves the slice around
an event so offline can replay it. This is the buffer, and a benchmark of what recording one
request costs. It does not hook a serving engine: where the hook sits is still open (ENG-189), and
vLLM's ``/metrics`` has no per-request data, so the numbers below are the cost of the *data path*
on the request thread, not of the hook itself.

Memory is fixed at construction: 24 bytes per slot (an 8-byte arrival time, two 4-byte lengths, an
8-byte prefix hash), preallocated, so recording never allocates and the ring never grows.
"""

from __future__ import annotations

import argparse
import hashlib
import threading
import time
from array import array

from gitm.traffic.schema import CanonicalRequest, Trace, TraceMeta

__all__ = ["RequestRing", "prefix_hash", "BYTES_PER_SLOT", "main"]

BYTES_PER_SLOT = 8 + 4 + 4 + 8


def prefix_hash(token_ids: list[int], block_tokens: int = 512) -> int:
    """A 64-bit hash of the first ``block_tokens`` tokens: requests that share a leading block
    share the hash. Only the first block is read, so the cost does not grow with prompt length."""
    head = array("I", token_ids[:block_tokens])
    return int.from_bytes(hashlib.blake2b(head.tobytes(), digest_size=8).digest(), "little")


class RequestRing:
    """Preallocated ring of the most recent ``capacity`` requests, safe to record from many threads."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._arrival = array("d", bytes(8 * capacity))
        self._inp = array("I", bytes(4 * capacity))
        self._out = array("I", bytes(4 * capacity))
        self._pfx = array("Q", bytes(8 * capacity))
        self._next = 0  # total ever recorded; slot is _next % capacity
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return min(self._next, self.capacity)

    @property
    def nbytes(self) -> int:
        return self.capacity * BYTES_PER_SLOT

    def record(self, arrival_s: float, input_tokens: int, output_tokens: int = 0,
               prefix: int = 0) -> None:
        with self._lock:
            i = self._next % self.capacity
            self._arrival[i] = arrival_s
            self._inp[i] = input_tokens
            self._out[i] = output_tokens
            self._pfx[i] = prefix
            self._next += 1

    def snapshot(self) -> list[tuple[float, int, int, int]]:
        """The retained requests, oldest first, as ``(arrival_s, input, output, prefix_hash)``."""
        with self._lock:
            n, cap, nxt = len(self), self.capacity, self._next
            idx = [(nxt - n + k) % cap for k in range(n)]
            rows = [(self._arrival[i], self._inp[i], self._out[i], self._pfx[i]) for i in idx]
        rows.sort(key=lambda r: r[0])
        return rows

    def slice(self, start_s: float, end_s: float) -> Trace:
        """The requests that arrived in ``[start_s, end_s)`` as a canonical trace, re-based to 0.

        A request recorded with no output length (0) has ``output_tokens=None``. The prefix hash
        becomes a one-block prefix chain, so requests that shared it still share it on replay.
        """
        if end_s <= start_s:
            raise ValueError("end_s must be after start_s")
        rows = [r for r in self.snapshot() if start_s <= r[0] < end_s]
        reqs = [CanonicalRequest(
            arrival_s=round(a - start_s, 6), input_tokens=max(i, 1),
            output_tokens=o or None, prefix_blocks=(p,) if p else ()) for a, i, o, p in rows]
        meta = TraceMeta(
            source="traffic-monitor", path=f"ring:{start_s:.0f}-{end_s:.0f}s",
            sha256=hashlib.sha256(repr(rows).encode()).hexdigest(),
            rows_read=len(reqs), rows_emitted=len(reqs), span_s=end_s - start_s,
            has_prefix_identity=any(r.prefix_blocks for r in reqs),
            notes=["recorded request shapes only; prompts were never stored"])
        return Trace(meta=meta, requests=reqs)


def _per_call_ns(fn, n: int) -> float:
    fn()  # warm
    t0 = time.perf_counter_ns()
    for _ in range(n):
        fn()
    return (time.perf_counter_ns() - t0) / n


def _contended_ns(ring: RequestRing, threads: int, n: int) -> float:
    """Mean ns per record when ``threads`` threads record at once."""
    barrier = threading.Barrier(threads)
    totals = [0.0] * threads

    def work(k: int) -> None:
        barrier.wait()
        t0 = time.perf_counter_ns()
        for j in range(n):
            ring.record(j * 1e-3, 512, 128, 7)
        totals[k] = (time.perf_counter_ns() - t0) / n

    ts = [threading.Thread(target=work, args=(k,)) for k in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return sum(totals) / threads


def main(argv: list[str] | None = None) -> int:
    """Print what recording one request costs, and what a ring of a given size occupies."""
    ap = argparse.ArgumentParser(prog="python -m gitm.traffic.ring")
    ap.add_argument("--n", type=int, default=200_000, help="calls per measurement")
    a = ap.parse_args(argv)
    ring = RequestRing(1 << 20)
    rec = _per_call_ns(lambda: ring.record(1.0, 512, 128, 7), a.n)
    print(f"record(): {rec:,.0f} ns per request, single thread (lock taken, {BYTES_PER_SLOT} B/slot)")
    for threads in (4, 16):
        print(f"record(): {_contended_ns(ring, threads, a.n // threads):,.0f} ns per request "
              f"with {threads} threads recording at once")
    for toks in (512, 4096, 32768):
        ids = list(range(1000, 1000 + toks))
        print(f"prefix_hash(): {_per_call_ns(lambda ids=ids: prefix_hash(ids), a.n // 10):,.0f} ns "
              f"for a {toks}-token prompt (first 512 tokens hashed)")
    print("\nring memory (24 B per request):")
    print(f"{'rate':>8} {'window':>8} {'requests':>10} {'memory':>10}")
    for rps in (10, 60, 500):
        for label, secs in (("10 min", 600), ("1 hour", 3600)):
            n = rps * secs
            print(f"{rps:>6}/s {label:>8} {n:>10,} {n * BYTES_PER_SLOT / 1e6:>8.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
