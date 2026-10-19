"""Rankings the distance metric must beat to be worth using.

Each baseline predicts the target point's candidate ranking without measuring
anything at the target, so it can be scored against the target's measured
ranking exactly like a transferred one.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, replace

from pydantic import ValidationError

from gitm.agents.autoresearch import _CLASS_KEYWORDS, classify_bottleneck
from gitm.agents.policy import Policy, RankedCandidate, select_interventions
from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.monitor import Residuals
from gitm.optimizer.preconditions import GateContext
from gitm.playbook.match import UNCALIBRATED_POLICY, MatchPolicy, MatchStatus, lookup
from gitm.playbook.schema import Playbook, RowIdentity
from gitm.tracer.schema import Trace


def catalog_predict_delta_ranking(
    trace: Trace,
    library: Iterable[InterventionSpec],
    policy: Policy,
    *,
    ctx: GateContext | None = None,
    top_n: int | None = None,
) -> list[RankedCandidate]:
    """Baseline 3: the loop's own ``predict_delta`` ranking of the catalog on
    the target's trace, with no history.

    History is never passed: it carries measured deltas, and a baseline that
    reads outcomes is not a pre-experiment prediction. ``top_n`` defaults to the
    whole library because rank agreement needs a complete ranking; pass the
    loop's ``top_n_interventions`` to reproduce its cut.
    """
    specs = list(library)
    return select_interventions(
        trace, specs, policy, top_n=len(specs) if top_n is None else top_n, ctx=ctx,
    )


def random_ranking(
    trace: Trace,
    library: Iterable[InterventionSpec],
    policy: Policy,
    *,
    seed: int,
    ctx: GateContext | None = None,
    top_n: int | None = None,
) -> list[RankedCandidate]:
    """Baseline 4: the candidates that pass the loop's gate in a uniformly
    random order, rejected candidates last.

    The gate is baseline 3's own call, so a candidate rejected there is rejected
    here with the same reason; only the order of the survivors is random.
    Survivors are put in name order before shuffling so the permutation for a
    seed depends on which candidates passed, not on their predicted scores or
    the library's order. Rejected candidates keep baseline 3's order.
    ``predicted_delta`` is 0.0 on every returned candidate so no score can be
    read into a random order.

    ``seed`` is required so a run's random baseline can be regenerated exactly.
    A private ``random.Random`` is used; the global RNG is never touched.
    """
    gated = catalog_predict_delta_ranking(trace, library, policy, ctx=ctx)
    passed = sorted((c for c in gated if c.rejected_reason is None), key=lambda c: c.spec.name)
    rejected = [c for c in gated if c.rejected_reason is not None]
    random.Random(seed).shuffle(passed)
    ranked = [replace(c, predicted_delta=0.0) for c in passed + rejected]
    return ranked if top_n is None else ranked[:top_n]


@dataclass(frozen=True)
class ExactMatchRanking:
    """Baseline 1's answer: a ranking plus which candidates it abstained on.

    ``ranking`` is complete (every library candidate appears) so it can be
    scored like any other, but only ``matched`` candidates carry a prediction.
    The rest are unranked, not predicted to be bad, and sit after the matched
    ones in name order with ``predicted_delta`` 0.0. A scorer should treat them
    as abstentions; ``abstained`` maps each to the reason.
    """

    ranking: list[RankedCandidate]
    matched: tuple[str, ...]
    abstained: dict[str, str]


def exact_match_ranking(
    playbook: Playbook,
    query: RowIdentity,
    trace: Trace,
    library: Iterable[InterventionSpec],
    policy: Policy,
    *,
    ctx: GateContext | None = None,
    match_policy: MatchPolicy = UNCALIBRATED_POLICY,
    exclude_row_ids: frozenset[str] = frozenset(),
) -> ExactMatchRanking:
    """Baseline 1: the playbook's measured result, only where its regime is
    exactly the target's.

    For each candidate that passes the loop's gate, ask :func:`lookup` for a row
    with the same model, GPU, environment, concurrency, regime and knob set as
    the target. A candidate with an ``EXACT_REGIME`` row is ranked by that row's
    measured throughput change; every other candidate is an abstention, whatever
    the nearest row was. That is deliberate: it is what the shipped policy does,
    since no axis is calibrated, and it is the "just reuse the old number"
    comparator the distance metric has to beat.

    ``query`` supplies everything but the knobs; each candidate's own knob set
    replaces ``query.knobs``. The playbook's rows come from other runs, so this
    reads measured outcomes by design, which is why ``exclude_row_ids`` exists:
    pass the ids of any row measured at the target point itself, or the baseline
    would be scoring with the answer. A knob value a row cannot hold (not a
    bool, int, float or str) is an abstention, not an error.
    """
    gated = catalog_predict_delta_ranking(trace, library, policy, ctx=ctx)
    usable = Playbook(
        rows=[r for r in playbook.rows if r.row_id not in exclude_row_ids],
    ) if exclude_row_ids else playbook

    matched: list[tuple[float, RankedCandidate]] = []
    abstained: dict[str, str] = {}
    rejected: list[RankedCandidate] = []
    for cand in gated:
        if cand.rejected_reason is not None:
            rejected.append(cand)
            continue
        try:
            q = query.model_copy(update={"knobs": dict(cand.spec.knob_values)})
            RowIdentity.model_validate(q.model_dump())
        except ValidationError:
            abstained[cand.spec.name] = "knobs_not_representable_in_a_playbook_row"
            continue
        result = lookup(usable, q, match_policy)
        if result.status is MatchStatus.EXACT_REGIME and result.row is not None:
            delta = result.row.delta.throughput_pct / 100.0
            matched.append((delta, replace(
                cand, predicted_delta=delta, delta_source="measured", demoted=False)))
        else:
            abstained[cand.spec.name] = result.status.value

    matched.sort(key=lambda t: (-t[0], t[1].spec.name))
    unranked = [
        replace(c, predicted_delta=0.0, delta_source="prior",
                rejected_reason=f"no_exact_match:{abstained[c.spec.name]}")
        for c in sorted((c for c in gated
                         if c.rejected_reason is None and c.spec.name in abstained),
                        key=lambda c: c.spec.name)
    ]
    ranking = [c for _, c in matched] + unranked + rejected
    return ExactMatchRanking(
        ranking=ranking,
        matched=tuple(c.spec.name for _, c in matched),
        abstained=abstained,
    )


@dataclass(frozen=True)
class BottleneckRanking:
    """Baseline 2's answer: the class the trace was put in and the ranking."""

    bottleneck_class: str
    keywords: tuple[str, ...]
    ranking: list[RankedCandidate]


def bottleneck_keyword_ranking(
    trace: Trace,
    library: Iterable[InterventionSpec],
    policy: Policy,
    *,
    residuals: Residuals | None = None,
    ctx: GateContext | None = None,
    top_n: int | None = None,
) -> BottleneckRanking:
    """Baseline 2: classify the bottleneck, then prefer levers whose knob names
    match that class's keywords.

    This is the autoresearch proposer's own rule (``classify_bottleneck`` and its
    keyword map) applied to the library instead of a generated surface. A
    candidate scores one point per keyword found in any of its knob names; higher
    scores rank first, and ties break by name, so no predicted delta enters the
    order. Candidates the loop's gate rejects stay last with its reasons. The
    static ``_RULES`` table is not used: it holds one entry in total, so it could
    not rank a library.

    The keyword map is a heuristic by its own docstring, so this baseline is
    expected to be weak. It stays in the comparison because it is what the
    repository does today when it has no history to go on.
    """
    cls = classify_bottleneck(trace, residuals)
    keywords = _CLASS_KEYWORDS.get(cls, ())
    gated = catalog_predict_delta_ranking(trace, library, policy, ctx=ctx)

    def score(c: RankedCandidate) -> int:
        names = [k.lower() for k in c.spec.knob_values]
        return sum(1 for kw in keywords for n in names if kw in n)

    passed = sorted((c for c in gated if c.rejected_reason is None),
                    key=lambda c: (-score(c), c.spec.name))
    rejected = [c for c in gated if c.rejected_reason is not None]
    ranked = [replace(c, predicted_delta=0.0) for c in passed + rejected]
    return BottleneckRanking(cls, keywords, ranked if top_n is None else ranked[:top_n])
