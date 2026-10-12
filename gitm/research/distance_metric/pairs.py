"""Pair selection and binning, before any outcome exists (design doc §7.1-§7.2).

:func:`select_pairs` takes point features and a config, nothing else. It
enumerates every unordered pair and gives each exactly one status:

* ``ineligible``: a gate failed, a point is not bridgeable or has no baseline
  capture, or the planners differ;
* ``inconclusive``: the capped common candidate set is below the minimum, or
  there is no common candidate at all;
* ``intermediate``: eligible, with d strictly between the thresholds; reported,
  but not used in the near/far test;
* ``near`` (d <= near threshold) or ``far`` (d >= far threshold): chosen.

Each bin must hold at least ``min_pairs_per_bin`` chosen pairs. An under-filled
bin is reported as inconclusive with a named reason; the doc's remedy is to
adjust the grid and re-run this check, which needs no outcome.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from gitm.research.distance_metric.candidates import CandidateSets, common_candidates
from gitm.research.distance_metric.config import DistanceConfig
from gitm.research.distance_metric.distance import DistanceResult, distance
from gitm.research.distance_metric.features import PointFeatures

NEAR = "near"
FAR = "far"
INTERMEDIATE = "intermediate"
INELIGIBLE = "ineligible"
INCONCLUSIVE = "inconclusive"
OK = "ok"


@dataclass(frozen=True)
class PairRecord:
    a: str
    b: str
    status: str
    distance: DistanceResult
    candidates: CandidateSets | None
    reasons: tuple[str, ...] = ()

    @property
    def chosen(self) -> bool:
        return self.status in (NEAR, FAR)


@dataclass(frozen=True)
class BinCheck:
    bin: str
    n: int
    required: int
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class PairSelectionReport:
    config: DistanceConfig
    points: tuple[str, ...]
    pairs: tuple[PairRecord, ...]
    bin_checks: tuple[BinCheck, ...]

    @property
    def status(self) -> str:
        return OK if all(c.status == OK for c in self.bin_checks) else INCONCLUSIVE

    def by_status(self, status: str) -> tuple[PairRecord, ...]:
        return tuple(p for p in self.pairs if p.status == status)

    @property
    def chosen(self) -> tuple[PairRecord, ...]:
        return tuple(p for p in self.pairs if p.chosen)

    @property
    def excluded(self) -> tuple[PairRecord, ...]:
        return tuple(p for p in self.pairs if not p.chosen)

    @property
    def counts(self) -> dict[str, int]:
        return {s: len(self.by_status(s)) for s in (NEAR, INTERMEDIATE, FAR, INCONCLUSIVE, INELIGIBLE)}


def bin_for(d: float, config: DistanceConfig) -> str:
    if d <= config.near_threshold:
        return NEAR
    if d >= config.far_threshold:
        return FAR
    return INTERMEDIATE


def _record(a: PointFeatures, b: PointFeatures, config: DistanceConfig) -> PairRecord:
    dist = distance(a, b, config)
    if dist.status == INELIGIBLE:
        return PairRecord(a.name, b.name, INELIGIBLE, dist, None, dist.reasons)
    cands = common_candidates(a, b, config)
    reasons = tuple(r for r in (*dist.reasons, cands.reason) if r)
    if reasons:
        return PairRecord(a.name, b.name, INCONCLUSIVE, dist, cands, reasons)
    status = bin_for(dist.distance, config)
    return PairRecord(a.name, b.name, status, dist, cands,
                      ("middle_band_unused",) if status == INTERMEDIATE else ())


def select_pairs(points: Sequence[PointFeatures], config: DistanceConfig) -> PairSelectionReport:
    names = [p.name for p in points]
    if len(set(names)) != len(names):
        raise ValueError(f"point names must be unique: {names}")
    ordered = sorted(points, key=lambda p: p.name)
    pairs = tuple(_record(a, b, config) for a, b in combinations(ordered, 2))
    checks = []
    for bin_name in (NEAR, FAR):
        n = sum(p.status == bin_name for p in pairs)
        short = n < config.min_pairs_per_bin
        checks.append(BinCheck(bin_name, n, config.min_pairs_per_bin,
                               INCONCLUSIVE if short else OK,
                               f"{bin_name}_bin_underfilled" if short else None))
    return PairSelectionReport(config, tuple(sorted(names)), pairs, tuple(checks))
