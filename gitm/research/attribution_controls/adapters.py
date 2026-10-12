"""dr.py as an estimator under test. The only module in this package that imports it.

``attribute_dr`` returns ranked (cause op -> effect op) hypotheses with
p-values. It has no mechanisms, no status and no reasons, so the mapping to a
:class:`Verdict` is partly arbitrary. Each choice:

* **Claimed cause** = the cause op of the top hypothesis (dr.py ranks by
  ``|z|``, so this is also the smallest p). Lossy: the effect op, the sign
  ("+ slower" / "- faster") and every other hypothesis are dropped, including a
  different cause op ranked second.
* **No mechanism class.** dr.py has none, so the claim names only the op. On a
  case whose group differs only in class, an op-only claim covers the group.
* **Confidence** = ``1 - p`` of that hypothesis. Arbitrary: p is a two-sided
  normal approximation to a test of "no effect of the cause's anomaly on the
  effect's residual", not the probability that the cause is right, and it is
  not calibrated (survey §1, E4-E6).
* **Status** = ``None``. dr.py has no status vocabulary, and saying
  "estimable" on its behalf would assert a check it does not make.
* **An empty result** = no claim, no reasons: the judge scores it ``MISSED``,
  not an abstention, because dr.py does not say why it is empty. The healthy
  run, no control rows and the lm_head truncation all return the same empty
  value (survey §1, E3).
* **Graph**: the series cases have none. ``attribute_dr`` never reads its graph
  argument (survey §1, E7), so it is passed ``predict_graph()``, as the probe does.
"""

from __future__ import annotations

from gitm.optimizer.dr import attribute_dr
from gitm.planner.graph import predict_graph
from gitm.research.attribution_controls.controls import EstimatorInput
from gitm.research.attribution_controls.verdict import Claim, Verdict


class DrEstimator:
    name = "dr.attribute_dr"

    def __call__(self, inp: EstimatorInput, /) -> Verdict:
        graph = inp.graph if inp.graph is not None else predict_graph()
        hyps = attribute_dr(inp.residuals, graph).hypotheses
        if not hyps:
            return Verdict(claim=None, confidence=None, status=None)
        top = hyps[0]
        return Verdict(
            claim=Claim.of(op=top.cause_op),
            confidence=1.0 - top.p_value,
            status=None,
            reasons=(f"top of {len(hyps)} hypotheses: {top.cause_op} -> {top.effect_op}, "
                     f"p={top.p_value:.4g}, {top.direction}, {top.notes}",),
        )


dr_estimator = DrEstimator()
