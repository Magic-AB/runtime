"""Candidate ranking v1: transfer a nearby point's measured result, or fall back.

Given a target operating point and source points whose candidates were measured,
pick the nearest eligible source by the distance metric. If it is within
``max_d``, feed its measured deltas to the loop's own
:func:`~gitm.agents.policy.select_interventions` as history, so the target's
ranking goes through the existing InterventionSpec path and gate. If no source
is close enough, return the catalog's ``predict_delta`` ranking unchanged and
say why.

What this is not: a validated method. It has only been exercised on synthetic
points with invented deltas, and ``max_d`` has no default because the right
value is for the team to choose after the pre-registered validation (design doc
§7). Nothing here reads the target's own measurements; a source whose name
matches the target is refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from gitm.agents.policy import Policy, RankedCandidate, select_interventions
from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.history import History, LeverRecord
from gitm.optimizer.preconditions import GateContext
from gitm.research.distance_metric.baselines import catalog_predict_delta_ranking
from gitm.research.distance_metric.config import DistanceConfig
from gitm.research.distance_metric.distance import distance
from gitm.research.distance_metric.features import PointFeatures
from gitm.tracer.schema import Trace

TRANSFERRED = "transferred"
FALLBACK = "fallback_catalog"


@dataclass(frozen=True)
class SourceMeasurement:
    """A source point and what its candidates measured there
    (candidate name -> ``speedup - 1``)."""

    features: PointFeatures
    measured_delta: Mapping[str, float]


@dataclass(frozen=True)
class SourceConsidered:
    name: str
    status: str
    distance: float | None
    setting_component: str | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RankingV1:
    mode: str  # TRANSFERRED or FALLBACK
    ranking: list[RankedCandidate]
    source: str | None
    distance: float | None
    setting_component: str | None
    transferred: tuple[str, ...]
    considered: tuple[SourceConsidered, ...]
    reason: str | None = None


def _history(measured: Mapping[str, float], *, sku: str | None, fingerprint: str | None) -> History:
    records = {}
    for name, delta in measured.items():
        records[(name, sku, fingerprint)] = LeverRecord(
            intervention_name=name, gpu_sku=sku, fingerprint=fingerprint, runs=1, attempts=1,
            wins=int(delta > 0), losses=int(delta < 0), inconclusive=int(delta == 0),
            mean_delta=delta, best_delta=delta, worst_delta=delta, last_run_id="transferred",
        )
    return History(records=records, runs_read=1)


def rank_v1(
    target: PointFeatures,
    sources: Sequence[SourceMeasurement],
    trace: Trace,
    library: Sequence[InterventionSpec],
    policy: Policy,
    config: DistanceConfig,
    *,
    max_d: float,
    ctx: GateContext | None = None,
    gpu_sku: str | None = None,
    fingerprint: str | None = None,
) -> RankingV1:
    """Rank ``library`` at ``target``. ``max_d`` is required: no default is chosen."""
    if not 0.0 <= max_d <= 1.0:
        raise ValueError(f"max_d must be in [0, 1], got {max_d}")
    considered: list[SourceConsidered] = []
    usable: list[tuple[float, str, SourceMeasurement, str | None]] = []
    for src in sources:
        if src.features.name == target.name:
            raise ValueError(
                f"source {src.features.name!r} is the target: a point's own measurements "
                "cannot be used to rank it")
        res = distance(target, src.features, config)
        considered.append(SourceConsidered(
            src.features.name, res.status, res.distance, res.setting_component, res.reasons))
        if res.distance is not None and res.distance != float("inf") and res.distance <= max_d:
            usable.append((res.distance, src.features.name, src, res.setting_component))

    baseline = catalog_predict_delta_ranking(trace, library, policy, ctx=ctx)
    if not usable:
        why = ("no sources given" if not sources else
               f"no source within max_d={max_d}: " + "; ".join(
                   f"{c.name} {c.status}" + (f" d={c.distance:.3f}" if c.distance not in (None, float('inf')) else "")
                   + (f" ({', '.join(c.reasons)})" if c.reasons else "")
                   for c in considered))
        return RankingV1(FALLBACK, baseline, None, None, None, (), tuple(considered), why)

    usable.sort(key=lambda t: (t[0], t[1]))
    d, _, src, component = usable[0]
    names = {s.name for s in library}
    measured = {n: v for n, v in src.measured_delta.items() if n in names}
    ranked = select_interventions(
        trace, list(library), replace(policy, use_history=True), top_n=len(library), ctx=ctx,
        history=_history(measured, sku=gpu_sku, fingerprint=fingerprint),
        gpu_sku=gpu_sku, fingerprint=fingerprint)
    return RankingV1(TRANSFERRED, ranked, src.features.name, d, component,
                     tuple(sorted(measured)), tuple(considered))
