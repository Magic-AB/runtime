"""Every externally owned name the harness uses, in one place.

Status names are the placeholders of ``docs/attribution_methods_survey.md``
§0.1. Andrew's estimator and its status vocabulary are not in the repo, so
nothing here is his. When his names land, replace the two strings below and
nothing else changes. There is deliberately no "absent" status: whether a
mechanism was present is a property of a control's truth, never a verdict.

Mechanism labels are ``docs/mechanism_model.md`` §3's IDs, keyed by the
fixture class in ``gitm/optimizer/mechanism_fixtures.py`` that injects each one.
The survey (§0.2) notes that list is being expanded, so the labels may change.
"""

from __future__ import annotations

ESTIMABLE = "[PENDING-STATUS: estimable]"
NOT_IDENTIFIABLE = "[PENDING-STATUS: not-identifiable]"
STATUSES = (ESTIMABLE, NOT_IDENTIFIABLE)

M1 = "M1"    # RegionSlowdown
M2 = "M2"    # Serialization
M3K = "M3k"  # AdditiveCost(where="kernel")
M3G = "M3g"  # AdditiveCost(where="gap")
M4 = "M4"    # RegimeGate
M5E = "M5e"  # TracerOverhead (an observation distortion, not an execution mechanism)
M5M = "M5m"  # Misroute (an observation distortion, not an execution mechanism)
MECHANISMS = (M1, M2, M3K, M3G, M4, M5E, M5M)
