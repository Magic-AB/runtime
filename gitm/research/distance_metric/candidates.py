"""The common candidate set of a pair (design doc §3, §7.3).

A candidate is its expanded spec name, resolved against each point's own
engine, so ``max_num_seqs_dynamic_x2`` is one candidate even when the two points
resolve it to different values. Three sets are reported for every pair:

* ``common``: names present in both expansions and runnable at both (gate,
  policy and both vetoes pass). This is the uncapped C_pre. It feeds the scope
  component S and the knob set Kset.
* ``capped``: the intersection of the two points' own capped lists. Each point
  keeps the first ``candidate_cap`` runnable candidates of its own baseline-3
  ranking, so every point has one fixed set to measure. Eligibility and the
  sensitivity subset use this set.
* ``same_values``: the capped names whose resolved ``knob_values`` are equal at
  both points (§3's sensitivity restriction).

A pair whose capped set is below ``min_common_candidates`` is inconclusive with
a named reason. It is reported, never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

from gitm.research.distance_metric.config import DistanceConfig
from gitm.research.distance_metric.features import PointFeatures, check_extracted_with

OK = "ok"
INCONCLUSIVE = "inconclusive"
COMMON_SET_BELOW_MINIMUM = "common_set_below_minimum"


@dataclass(frozen=True)
class CandidateSets:
    a: str
    b: str
    common: tuple[str, ...]
    only_a: tuple[str, ...]
    only_b: tuple[str, ...]
    #: (name, point, reason) for names expanded at both points but not runnable
    #: at the named point.
    not_runnable: tuple[tuple[str, str, str], ...]
    capped: tuple[str, ...]
    same_values: tuple[str, ...]
    status: str
    reason: str | None = None

    @property
    def sizes(self) -> dict[str, int]:
        return {"common": len(self.common), "capped": len(self.capped),
                "same_values": len(self.same_values)}


def common_candidates(a: PointFeatures, b: PointFeatures, config: DistanceConfig) -> CandidateSets:
    ca, cb = a.candidate_map(), b.candidate_map()
    both = sorted(ca.keys() & cb.keys())
    common = tuple(n for n in both if ca[n].runnable and cb[n].runnable)
    not_runnable = tuple(
        (n, p.name, c[n].reason or "")
        for n in both
        for p, c in ((a, ca), (b, cb))
        if not c[n].runnable
    )
    check_extracted_with(config, a, b)
    capped = tuple(sorted(set(a.capped(config)) & set(b.capped(config))))
    same_values = tuple(n for n in capped if ca[n].knob_values == cb[n].knob_values)
    small = len(capped) < config.min_common_candidates
    return CandidateSets(
        a=a.name,
        b=b.name,
        common=common,
        only_a=tuple(sorted(ca.keys() - cb.keys())),
        only_b=tuple(sorted(cb.keys() - ca.keys())),
        not_runnable=not_runnable,
        capped=capped,
        same_values=same_values,
        status=INCONCLUSIVE if small else OK,
        reason=COMMON_SET_BELOW_MINIMUM if small else None,
    )
