"""Every number distance metric v1 uses, in one frozen record.

``docs/distance_metric_v1_design.md`` is a working draft. Its thresholds, grid,
cap and cut-offs are the previous author's defaults, not preregistered values,
so each default below says so. Nothing downstream hard-codes any of them: every
function that needs one takes a :class:`DistanceConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

from gitm.agents.policy import Policy
from gitm.playbook.match import DEFAULT_AXES

#: (concurrency, input_p50, output_p50) for the design doc's 7-point grid (§7.1).
DOC_GRID: tuple[tuple[int, int, int], ...] = (
    (32, 4096, 512),
    (40, 4096, 512),
    (48, 4096, 512),
    (64, 4096, 512),
    (96, 4096, 512),
    (128, 4096, 512),
    (64, 8192, 512),
)

#: The fixed knobs §4 adds to Kset on top of the candidates' own knobs.
DOC_KSET_BASE: tuple[str, ...] = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "enable_chunked_prefill",
    "enforce_eager",
    "kv_cache_dtype",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "block_size",
)


@dataclass(frozen=True)
class DistanceConfig:
    # --- bins (§7.2) ---------------------------------------------------------
    near_threshold: float = 0.25  # default from design doc, not preregistered
    far_threshold: float = 0.5  # default from design doc, not preregistered
    min_pairs_per_bin: int = 6  # default from design doc, not preregistered

    # --- grid (§7.1) ---------------------------------------------------------
    n_grid_points: int = 7  # default from design doc, not preregistered
    grid: tuple[tuple[int, int, int], ...] = DOC_GRID  # default from design doc, not preregistered

    # --- candidates (§3, §7.1, §7.3) -----------------------------------------
    candidate_cap: int = 8  # default from design doc, not preregistered
    min_common_candidates: int = 6  # default from design doc, not preregistered
    skip_high_risk: bool = True  # default from design doc, not preregistered
    require_qualification_commit: bool = False  # default from design doc, not preregistered

    # --- distance (§4) -------------------------------------------------------
    traffic_axes: tuple[str, ...] = DEFAULT_AXES  # default from design doc, not preregistered
    kset_base_knobs: tuple[str, ...] = DOC_KSET_BASE  # default from design doc, not preregistered

    # --- measurement (§7.3, §7.4) --------------------------------------------
    reps_per_arm: int = 3  # default from design doc, not preregistered
    bootstrap_draws: int = 4000  # default from design doc, not preregistered
    interval_level: float = 0.95  # default from design doc, not preregistered

    # --- inconclusive cut-offs (§7.5, §7.6) ----------------------------------
    max_failed_fraction: float = 0.25  # default from design doc, not preregistered
    max_tied_fraction: float = 0.5  # default from design doc, not preregistered
    max_rep_cv: float = 0.05  # default from design doc, not preregistered
    max_point_drift: float = 0.02  # default from design doc, not preregistered
    min_self_tau: float = 0.6  # default from design doc, not preregistered
    min_conclusive_fraction: float = 0.5  # default from design doc, not preregistered
    reject_delta_tau_upper: float = 0.2  # default from design doc, not preregistered
    #: How many resolved candidate pairs the secondary statistic needs before it
    #: is reported rather than flagged. The design doc says only "few", so there
    #: is no default: :func:`~gitm.research.distance_metric.agreement.resolved_concordance`
    #: refuses to run until a caller sets it.
    min_resolved_pairs: int | None = None

    # --- top-k secondaries (§6) ----------------------------------------------
    top1_within: int = 3  # default from design doc, not preregistered
    top_overlap_k: int = 5  # default from design doc, not preregistered

    def __post_init__(self) -> None:
        if not 0.0 <= self.near_threshold < self.far_threshold <= 1.0:
            raise ValueError(
                f"need 0 <= near_threshold < far_threshold <= 1, got "
                f"{self.near_threshold} / {self.far_threshold}"
            )
        if len(self.grid) != self.n_grid_points:
            raise ValueError(f"grid has {len(self.grid)} points but n_grid_points={self.n_grid_points}")
        for name in ("candidate_cap", "min_common_candidates", "min_pairs_per_bin", "reps_per_arm"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.min_resolved_pairs is not None and self.min_resolved_pairs < 1:
            raise ValueError("min_resolved_pairs must be >= 1 when set")

    def policy(self) -> Policy:
        """The experiment's fixed policy. History is always off: it carries
        measured outcomes, which no input to the metric may (§1)."""
        return Policy(
            require_qualification_commit=self.require_qualification_commit,
            skip_high_risk=self.skip_high_risk,
            use_history=False,
        )
