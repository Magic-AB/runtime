"""Score one verdict against one control's truth.

The rules, in order (the first that applies decides):

1. ``NOT_IDENTIFIABLE`` (which the verdict check requires to carry a reason)
   is ``VALID_ABSTENTION`` on every case, whatever the truth.
2. No claim is ``MISSED``: the estimator said nothing and gave no abstention
   reason. An empty result is not an abstention.
3. A claim is compared, alternative by alternative, with the truth group. An
   alternative *covers* a member when every part it names matches (``op=None``
   or ``mechanism=None`` leaves that part open; a member on no particular op is
   covered only by an alternative that names no op).
   * Nothing planted, or an alternative that covers no member: a wrong claim.
   * Every member covered: ``CORRECT``. Naming only the op, when the members
     differ only in mechanism class, covers them all.
   * Some members covered, none stray: it picks out part of a group the
     capture cannot split. ``OVERCLAIMED_GROUP`` if confident.
   A wrong or part-of-group claim is confident when its confidence is at or
   above the threshold, or is ``None`` (an unqualified claim is a confident one);
   below the threshold it is ``WRONG_HEDGED``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gitm.research.attribution_controls.controls import Cause, Truth
from gitm.research.attribution_controls.verdict import Alternative, Verdict, validate_verdict

CORRECT = "correct"
VALID_ABSTENTION = "valid_abstention"
WRONG_CONFIDENT = "wrong_confident"
WRONG_HEDGED = "wrong_hedged"
MISSED = "missed"
OVERCLAIMED_GROUP = "overclaimed_group"
OUTCOMES = (CORRECT, VALID_ABSTENTION, WRONG_CONFIDENT, WRONG_HEDGED, MISSED, OVERCLAIMED_GROUP)


@dataclass(frozen=True)
class JudgeConfig:
    #: Claims at or above this confidence count as confident. No default: the
    #: value is the team's decision, so every caller must state it.
    confidence_threshold: float

    def __post_init__(self) -> None:
        t = self.confidence_threshold
        if isinstance(t, bool) or not isinstance(t, int | float) or math.isnan(t) or not 0 <= t <= 1:
            raise ValueError(f"confidence_threshold {t!r} is not a number in [0, 1]")


@dataclass(frozen=True)
class Judgement:
    outcome: str
    reasons: tuple[str, ...]


def covers(a: Alternative, m: Cause) -> bool:
    op_ok = a.op is None or (m.op is not None and a.op == m.op)
    mech_ok = a.mechanism is None or (m.mechanism is not None and a.mechanism == m.mechanism)
    return op_ok and mech_ok


def judge(verdict: Verdict, truth: Truth, config: JudgeConfig) -> Judgement:
    v = validate_verdict(verdict)
    if v.abstains:
        return Judgement(VALID_ABSTENTION, ("not identifiable: " + "; ".join(v.reasons),))
    if v.claim is None:
        return Judgement(MISSED, ("no claim and no abstention reason",))

    claim = v.claim.describe()
    confident = v.confidence is None or v.confidence >= config.confidence_threshold
    conf = ("confidence not given, taken as confident" if v.confidence is None
            else f"confidence {v.confidence:.4g} {'>=' if confident else '<'} "
                 f"threshold {config.confidence_threshold:g}")
    wrong = WRONG_CONFIDENT if confident else WRONG_HEDGED

    if truth.nothing_planted:
        return Judgement(wrong, (f"claimed {claim}, but nothing was planted", conf))
    stray = sorted(a.describe() for a in v.claim.alternatives
                   if not any(covers(a, m) for m in truth.group))
    if stray:
        return Judgement(wrong, (f"claimed {', '.join(stray)}, which matches nothing planted "
                                 f"(truth: {truth.describe()})", conf))
    covered = {m for m in truth.group if any(covers(a, m) for a in v.claim.alternatives)}
    if covered == truth.group:
        return Judgement(CORRECT, (f"claimed {claim}; truth {truth.describe()}",))
    missing = ", ".join(sorted(m.describe() for m in truth.group - covered))
    why = (f"claimed {claim}, part of a group a single capture cannot split "
           f"(truth: {truth.describe()}; not covered: {missing})")
    return Judgement(OVERCLAIMED_GROUP if confident else WRONG_HEDGED, (why, conf))
