"""The verdict every estimator under test returns, and the check that rejects a malformed one.

Modelled on ``separation.Decision``: a status, the reasons for it, and what
would change it. A verdict either claims a cause (with a confidence), abstains
(status ``NOT_IDENTIFIABLE`` with at least one reason), or says nothing. It has
no way to say "no mechanism is present": that is a fact about a control's truth
(see :mod:`.vocabulary`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from gitm.research.attribution_controls.vocabulary import MECHANISMS, NOT_IDENTIFIABLE, STATUSES


class MalformedVerdict(ValueError):
    """The verdict breaks the contract below; it is not judged."""


@dataclass(frozen=True, slots=True)
class Alternative:
    """One possible cause: an op, a mechanism class, or both. ``None`` leaves that part open."""

    op: str | None = None
    mechanism: str | None = None

    def describe(self) -> str:
        return f"{self.op or '*'}/{self.mechanism or '*'}"


@dataclass(frozen=True, slots=True)
class Claim:
    """The claimed cause. One alternative names a single cause; several name a group."""

    alternatives: frozenset[Alternative]

    @classmethod
    def of(cls, op: str | None = None, mechanism: str | None = None) -> Claim:
        return cls(frozenset({Alternative(op, mechanism)}))

    @classmethod
    def group(cls, alternatives: Iterable[Alternative | tuple[str | None, str | None]]) -> Claim:
        return cls(frozenset(a if isinstance(a, Alternative) else Alternative(*a)
                             for a in alternatives))

    def describe(self) -> str:
        alts = sorted(a.describe() for a in self.alternatives)
        return alts[0] if len(alts) == 1 else "{" + ", ".join(alts) + "}"


@dataclass(frozen=True, slots=True)
class Verdict:
    claim: Claim | None
    #: In [0, 1]. A claim with ``None`` is treated by the judge as confident.
    confidence: float | None
    #: One of :data:`STATUSES`, or ``None`` for an estimator with no status vocabulary.
    status: str | None
    reasons: tuple[str, ...] = ()
    remedy: str | None = None

    @property
    def abstains(self) -> bool:
        return self.status == NOT_IDENTIFIABLE


def validate_verdict(v: object) -> Verdict:
    """Return ``v`` if it is well formed, else raise :class:`MalformedVerdict`."""
    if not isinstance(v, Verdict):
        raise MalformedVerdict(f"expected a Verdict, got {type(v).__name__}")
    if v.status is not None and v.status not in STATUSES:
        raise MalformedVerdict(f"status {v.status!r} is not one of {STATUSES} or None")
    if not isinstance(v.reasons, tuple) or not all(isinstance(r, str) and r.strip()
                                                    for r in v.reasons):
        raise MalformedVerdict("reasons must be a tuple of non-empty strings")
    if v.remedy is not None and not (isinstance(v.remedy, str) and v.remedy.strip()):
        raise MalformedVerdict("remedy must be None or a non-empty string")
    c = v.confidence
    if c is not None:
        if isinstance(c, bool) or not isinstance(c, int | float) or math.isnan(c) or not 0 <= c <= 1:
            raise MalformedVerdict(f"confidence {c!r} is not a number in [0, 1]")
        if v.claim is None:
            raise MalformedVerdict("a confidence without a claim")
    if v.claim is not None:
        if not isinstance(v.claim, Claim) or not isinstance(v.claim.alternatives, frozenset):
            raise MalformedVerdict("claim must be a Claim over a frozenset of Alternatives")
        if not v.claim.alternatives:
            raise MalformedVerdict("a claim with no alternatives")
        for a in v.claim.alternatives:
            if not isinstance(a, Alternative):
                raise MalformedVerdict(f"claim alternative {a!r} is not an Alternative")
            if a.op is None and a.mechanism is None:
                raise MalformedVerdict("an alternative that names neither an op nor a mechanism")
            if a.op is not None and not (isinstance(a.op, str) and a.op):
                raise MalformedVerdict(f"op {a.op!r} is not a non-empty string")
            if a.mechanism is not None and a.mechanism not in MECHANISMS:
                raise MalformedVerdict(f"mechanism {a.mechanism!r} is not one of {MECHANISMS}")
    if v.abstains:
        if v.claim is not None:
            raise MalformedVerdict("an abstention that also claims a cause")
        if not v.reasons:
            raise MalformedVerdict("an abstention must give at least one reason")
    return v
