"""Test estimators that exercise the judge and the gate without dr.py.

:func:`oracle` reads the truth, which no real estimator may do. It recognises
a case by a fingerprint of its input, so it can only answer for the suite it
was built from, and it refuses a suite in which two cases with different truths
have the same input (such a suite would be ambiguous by construction).
"""

from __future__ import annotations

import hashlib

import numpy as np

from gitm.research.attribution_controls.controls import EstimatorInput, Suite, Truth
from gitm.research.attribution_controls.verdict import Alternative, Claim, Verdict
from gitm.research.attribution_controls.vocabulary import ESTIMABLE, NOT_IDENTIFIABLE


def fingerprint(inp: EstimatorInput) -> str:
    h = hashlib.sha256()
    for k in inp.residuals.per_kernel:
        h.update(repr((k.op, k.layer, k.r_kt, k.r_mt)).encode())
    if inp.trace is not None:
        for e in inp.trace.events:
            h.update(repr((e.name, e.start_ns, e.end_ns, e.stream_id)).encode())
    return h.hexdigest()


def _ops(inp: EstimatorInput) -> list[str]:
    return sorted({k.op for k in inp.residuals.per_kernel})


class Oracle:
    name = "oracle"

    def __init__(self, suite: Suite) -> None:
        self._truth: dict[str, Truth] = {}
        for case in suite.cases:
            fp = fingerprint(case.build())
            if fp in self._truth and self._truth[fp] != case.truth:
                raise ValueError(f"{case.name}: same input as another case with a different truth")
            self._truth[fp] = case.truth

    def __call__(self, inp: EstimatorInput, /) -> Verdict:
        truth = self._truth.get(fingerprint(inp))
        if truth is None:
            raise KeyError("oracle: this input is not from the suite it was built for")
        if truth.nothing_planted:
            return Verdict(None, None, NOT_IDENTIFIABLE,
                           ("oracle: nothing planted, and a verdict cannot report absence",))
        claim = Claim.group(Alternative(c.op, c.mechanism) for c in truth.group)
        return Verdict(claim, 1.0, ESTIMABLE, ("oracle: read the planted truth",))


def oracle(suite: Suite) -> Oracle:
    return Oracle(suite)


class AlwaysAbstain:
    name = "always_abstain"

    def __call__(self, inp: EstimatorInput, /) -> Verdict:
        return Verdict(None, None, NOT_IDENTIFIABLE, ("always_abstain: never estimates",),
                       remedy="use an estimator that estimates")


def always_abstain() -> AlwaysAbstain:
    return AlwaysAbstain()


class AlwaysClaim:
    def __init__(self, op: str) -> None:
        self.op = op
        self.name = f"always_claim({op})"

    def __call__(self, inp: EstimatorInput, /) -> Verdict:
        return Verdict(Claim.of(op=self.op), 1.0, ESTIMABLE, (f"always names {self.op}",))


def always_claim(op: str) -> AlwaysClaim:
    return AlwaysClaim(op)


class RandomClaim:
    """Names one of the input's ops at random, with a uniform random confidence.

    The draw is seeded by ``seed`` and the input's fingerprint, so it does not
    depend on the order the cases are run in.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.name = f"random_claim(seed={seed})"

    def __call__(self, inp: EstimatorInput, /) -> Verdict:
        ops = _ops(inp)
        if not ops:
            return Verdict(None, None, None)
        rng = np.random.default_rng([self.seed, int(fingerprint(inp)[:12], 16)])
        op = ops[int(rng.integers(len(ops)))]
        return Verdict(Claim.of(op=op), float(rng.random()), ESTIMABLE, ("random pick",))


def random_claim(seed: int) -> RandomClaim:
    return RandomClaim(seed)
