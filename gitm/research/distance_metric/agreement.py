"""Rank agreement between two rankings of one candidate set (design doc §6).

Rankings are plain ``[(name, score), ...]`` lists, so the same code scores a
transferred measured ranking, baseline 3, or the random baseline. Higher score
means better; only the order matters.

Neither statistic ever returns NaN. Where it is undefined (fewer than two
candidates, or a ranking with no order in it) the result says so with an
``inconclusive`` status and a named reason.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from gitm.research.distance_metric.config import DistanceConfig

OK = "ok"
INCONCLUSIVE = "inconclusive"
TOO_FEW_CANDIDATES = "too_few_candidates"
ALL_TIED = "all_tied"
CONSTANT_RANKING = "constant_ranking"
FEW_RESOLVED_PAIRS = "few_resolved_pairs"

Ranking = Sequence[tuple[str, float]]
#: (lo, hi) interval for ``delta_x - delta_y``, keyed by the pair ``(x, y)``.
DiffIntervals = Mapping[tuple[str, str], tuple[float, float]]


@dataclass(frozen=True)
class TauResult:
    status: str
    tau: float | None
    reason: str | None
    n: int
    concordant: int = 0
    discordant: int = 0
    #: Candidate pairs tied in ``a`` (resp. ``b``), whatever the other ranking says.
    tied_pairs_a: int = 0
    tied_pairs_b: int = 0


@dataclass(frozen=True)
class ResolvedConcordance:
    status: str
    #: Fraction of resolved candidate pairs ordered the same way at both points.
    concordance: float | None
    reason: str | None
    n: int
    n_pairs: int
    n_resolved: int
    n_concordant: int
    #: Fraction of candidate pairs whose interval contains 0, at each point.
    tied_fraction_a: float | None
    tied_fraction_b: float | None


def _scores(ranking: Ranking, label: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, score in ranking:
        if name in out:
            raise ValueError(f"ranking {label} lists {name!r} twice")
        if not math.isfinite(score):
            raise ValueError(f"ranking {label} has a non-finite score for {name!r}: {score}")
        out[name] = float(score)
    return out


def _paired(a: Ranking, b: Ranking) -> tuple[list[str], dict[str, float], dict[str, float]]:
    sa, sb = _scores(a, "a"), _scores(b, "b")
    if sa.keys() != sb.keys():
        raise ValueError(
            "rankings cover different candidates; restrict both to the common set first: "
            f"only in a {sorted(sa.keys() - sb.keys())}, only in b {sorted(sb.keys() - sa.keys())}"
        )
    return sorted(sa), sa, sb


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def kendall_tau_b(a: Ranking, b: Ranking) -> TauResult:
    """Kendall's tau_b, tie-corrected in both rankings:

        tau_b = (C - D) / sqrt((n0 - n1) (n0 - n2))

    with n0 = n(n-1)/2 and n1, n2 the pairs tied in ``a`` and in ``b``.
    """
    names, sa, sb = _paired(a, b)
    n = len(names)
    if n < 2:
        return TauResult(INCONCLUSIVE, None, TOO_FEW_CANDIDATES, n)
    c = d = n1 = n2 = 0
    for x, y in combinations(names, 2):
        da, db = _sign(sa[x] - sa[y]), _sign(sb[x] - sb[y])
        n1 += da == 0
        n2 += db == 0
        if da and db:
            c += da == db
            d += da != db
    n0 = n * (n - 1) // 2
    if n1 == n0 and n2 == n0:
        return TauResult(INCONCLUSIVE, None, ALL_TIED, n, c, d, n1, n2)
    if n1 == n0 or n2 == n0:
        return TauResult(INCONCLUSIVE, None, CONSTANT_RANKING, n, c, d, n1, n2)
    tau = (c - d) / math.sqrt((n0 - n1) * (n0 - n2))
    return TauResult(OK, tau, None, n, c, d, n1, n2)


def _interval(intervals: DiffIntervals, x: str, y: str, label: str) -> tuple[float, float]:
    if (x, y) in intervals:
        lo, hi = intervals[(x, y)]
    elif (y, x) in intervals:
        lo, hi = (-intervals[(y, x)][1], -intervals[(y, x)][0])
    else:
        raise ValueError(f"no difference interval for ({x!r}, {y!r}) at point {label}")
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        raise ValueError(f"bad interval {(lo, hi)} for ({x!r}, {y!r}) at point {label}")
    return lo, hi


def _direction(lo: float, hi: float) -> int:
    """+1 / -1 when the interval excludes 0, else 0 (a noise-level tie)."""
    return 1 if lo > 0 else -1 if hi < 0 else 0


def resolved_concordance(
    a: Ranking,
    b: Ranking,
    intervals_a: DiffIntervals,
    intervals_b: DiffIntervals,
    config: DistanceConfig,
) -> ResolvedConcordance:
    """§6's secondary statistic: concordance among *resolved* candidate pairs.

    A candidate pair is resolved when the interval of the difference of the two
    candidates' deltas excludes 0 at both points. Each ``intervals_*`` maps
    ``(x, y)`` to the interval for ``delta_x - delta_y`` at that point; either
    orientation of a pair may be given. Fewer than ``config.min_resolved_pairs``
    resolved pairs flags the result instead of reporting a weak concordance.
    """
    if config.min_resolved_pairs is None:
        raise ValueError(
            "config.min_resolved_pairs is unset: the design doc only says 'few', so "
            "the caller must choose a number before this statistic can be reported"
        )
    names, _, _ = _paired(a, b)
    n = len(names)
    pairs = list(combinations(names, 2))
    if n < 2:
        return ResolvedConcordance(INCONCLUSIVE, None, TOO_FEW_CANDIDATES, n, 0, 0, 0, None, None)
    resolved = concordant = tied_a = tied_b = 0
    for x, y in pairs:
        da = _direction(*_interval(intervals_a, x, y, "a"))
        db = _direction(*_interval(intervals_b, x, y, "b"))
        tied_a += da == 0
        tied_b += db == 0
        if da and db:
            resolved += 1
            concordant += da == db
    fa, fb = tied_a / len(pairs), tied_b / len(pairs)
    if resolved < config.min_resolved_pairs:
        return ResolvedConcordance(INCONCLUSIVE, None, FEW_RESOLVED_PAIRS, n, len(pairs),
                                   resolved, concordant, fa, fb)
    return ResolvedConcordance(OK, concordant / resolved, None, n, len(pairs), resolved,
                               concordant, fa, fb)
