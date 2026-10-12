"""Run an estimator over a suite, decide whether it is valid, and block it if not.

Validity and usefulness are reported side by side and never combined. An
estimator that always abstains is VALID with a positive-identification rate of
0; "valid" means only that it never confidently named a wrong cause (or one
member of a group the capture cannot split) on any control.

:func:`validate` is the only way to obtain a :class:`ValidationResult`, and a
:class:`GuardedEstimator` cannot be built without one for the same estimator
object. A guarded estimator whose validation is INVALID raises
:class:`EstimatorBlocked` on every call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gitm.research.attribution_controls.controls import (
    NULL,
    POSITIVE,
    Estimator,
    EstimatorInput,
    Suite,
)
from gitm.research.attribution_controls.judge import (
    CORRECT,
    MISSED,
    OUTCOMES,
    OVERCLAIMED_GROUP,
    VALID_ABSTENTION,
    WRONG_CONFIDENT,
    WRONG_HEDGED,
    JudgeConfig,
    judge,
)
from gitm.research.attribution_controls.verdict import MalformedVerdict, Verdict

VALID, INVALID = "VALID", "INVALID"
#: The outcomes that make an estimator INVALID.
BLOCKING = (WRONG_CONFIDENT, OVERCLAIMED_GROUP)


class EstimatorBlocked(RuntimeError):
    """The estimator failed validation; its verdicts must not be used."""


class NotValidated(RuntimeError):
    """A guard was requested without a validation of this estimator."""


def estimator_name(est: object) -> str:
    return str(getattr(est, "name", None) or getattr(est, "__name__", None) or type(est).__name__)


@dataclass(frozen=True)
class CaseResult:
    case: str
    kind: str
    truth: str
    claim: str | None
    confidence: float | None
    status: str | None
    outcome: str
    why: tuple[str, ...]
    verdict_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SuiteReport:
    estimator: str
    confidence_threshold: float
    results: tuple[CaseResult, ...]
    counts: tuple[tuple[str, int], ...]
    abstentions: int
    positives_identified: int
    positives_total: int

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def abstention_rate(self) -> float:
        return self.abstentions / self.n if self.n else 0.0

    @property
    def identification_rate(self) -> float:
        return self.positives_identified / self.positives_total if self.positives_total else 0.0

    def count(self, outcome: str) -> int:
        return dict(self.counts)[outcome]

    def usefulness(self) -> str:
        return (f"positive identification {self.positives_identified}/{self.positives_total} "
                f"= {self.identification_rate:.0%}; abstention {self.abstentions}/{self.n} "
                f"= {self.abstention_rate:.0%}")

    def table(self) -> str:
        rows = [("case", "kind", "truth", "claim", "conf", "outcome")]
        for r in self.results:
            conf = "-" if r.confidence is None else f"{r.confidence:.4f}"
            rows.append((r.case, r.kind, r.truth, r.claim or "-", conf, r.outcome))
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        return "\n".join("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
                         for row in rows)


def run_suite(estimator: Estimator, suite: Suite, config: JudgeConfig) -> SuiteReport:
    """Call the estimator once per case on that case's input only, and judge each verdict."""
    name = estimator_name(estimator)
    results = []
    for case in suite.cases:
        verdict: Verdict = estimator(case.build())
        try:
            j = judge(verdict, case.truth, config)
        except MalformedVerdict as e:
            raise MalformedVerdict(f"{name} on {case.name}: {e}") from e
        results.append(CaseResult(
            case=case.name, kind=case.kind, truth=case.truth.describe(),
            claim=None if verdict.claim is None else verdict.claim.describe(),
            confidence=verdict.confidence, status=verdict.status, outcome=j.outcome,
            why=j.reasons, verdict_reasons=verdict.reasons,
        ))
    positives = [r for r in results if r.kind == POSITIVE]
    return SuiteReport(
        estimator=name, confidence_threshold=config.confidence_threshold,
        results=tuple(results),
        counts=tuple((o, sum(r.outcome == o for r in results)) for o in OUTCOMES),
        abstentions=sum(r.outcome == VALID_ABSTENTION for r in results),
        positives_identified=sum(r.outcome == CORRECT for r in positives),
        positives_total=len(positives),
    )


_SEAL = object()


@dataclass(frozen=True)
class ValidationResult:
    status: str
    #: Every case with a blocking outcome, and the judge's reasons.
    failures: tuple[CaseResult, ...]
    report: SuiteReport
    estimator: object = field(repr=False, compare=False)
    _seal: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _SEAL:
            raise NotValidated("a ValidationResult can only come from validate()")

    @property
    def valid(self) -> bool:
        return self.status == VALID

    def summary(self) -> str:
        """Validity and usefulness on one line, so neither is read without the other."""
        r = self.report
        blocking = ", ".join(f"{o} {r.count(o)}" for o in BLOCKING)
        other = ", ".join(f"{o} {r.count(o)}" for o in (WRONG_HEDGED, MISSED))
        return f"{r.estimator}: {self.status} ({blocking}) | {r.usefulness()} | {other}"

    def explain(self) -> str:
        lines = [self.summary()]
        for f in self.failures:
            lines.append(f"  {f.case} [{f.kind}] -> {f.outcome}: " + "; ".join(f.why))
        return "\n".join(lines)


def validate(estimator: Estimator, suite: Suite, config: JudgeConfig) -> ValidationResult:
    """INVALID if any case is ``WRONG_CONFIDENT`` or ``OVERCLAIMED_GROUP``.

    The suite must contain a positive and a null control, so that neither
    always-claiming nor never-claiming can pass vacuously.
    """
    if not suite.of_kind(POSITIVE) or not suite.of_kind(NULL):
        raise ValueError("validation needs at least one positive and one null control")
    report = run_suite(estimator, suite, config)
    failures = tuple(r for r in report.results if r.outcome in BLOCKING)
    return ValidationResult(INVALID if failures else VALID, failures, report, estimator, _SEAL)


class GuardedEstimator:
    """Passes calls through to a VALID estimator unchanged; raises for an INVALID one."""

    def __init__(self, estimator: Estimator, validation: ValidationResult) -> None:
        if not isinstance(validation, ValidationResult):
            raise NotValidated(f"{estimator_name(estimator)} has no validation; run validate() first")
        if validation.estimator is not estimator:
            raise NotValidated(f"the validation is of {validation.report.estimator}, "
                               f"not of this {estimator_name(estimator)} object")
        self._estimator = estimator
        self.validation = validation
        self.name = f"guarded({estimator_name(estimator)})"

    def __call__(self, inp: EstimatorInput, /) -> Verdict:
        if not self.validation.valid:
            raise EstimatorBlocked(self.validation.explain())
        return self._estimator(inp)
