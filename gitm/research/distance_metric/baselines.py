"""Rankings the distance metric must beat to be worth using.

Each baseline predicts the target point's candidate ranking without measuring
anything at the target, so it can be scored against the target's measured
ranking exactly like a transferred one.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import replace

from gitm.agents.policy import Policy, RankedCandidate, select_interventions
from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.preconditions import GateContext
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
