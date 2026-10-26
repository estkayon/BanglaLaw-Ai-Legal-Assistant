from dataclasses import dataclass

from loguru import logger


@dataclass
class EvalMetrics:
    """Evaluation metrics for a single QA pair."""

    qa_id: str
    question: str
    expected_act: str
    expected_section: str
    predicted_answer: str
    predicted_citations: list[dict]
    confidence: float
    is_confident: bool

    @property
    def citation_hit(self) -> bool:
        """Check if any predicted citation matches the expected act."""
        for cite in self.predicted_citations:
            if self.expected_act and self.expected_act.lower() in cite.get("act_title", "").lower():
                return True
        return False

    @property
    def section_hit(self) -> bool:
        """Check if any predicted citation matches the expected section number."""
        for cite in self.predicted_citations:
            if self.expected_section and self.expected_section == cite.get("section_number", ""):
                return True
        return False

    @property
    def has_answer(self) -> bool:
        """Check if the system produced a confident answer (not a fallback)."""
        return self.is_confident and len(self.predicted_answer) > 20

    def to_dict(self) -> dict:
        return {
            "qa_id": self.qa_id,
            "question": self.question,
            "expected_act": self.expected_act,
            "expected_section": self.expected_section,
            "citation_hit": self.citation_hit,
            "section_hit": self.section_hit,
            "has_answer": self.has_answer,
            "confidence": self.confidence,
        }


@dataclass
class EvalSummary:
    """Aggregated evaluation results across all QA pairs."""

    total: int
    answered: int
    citation_hits: int
    section_hits: int
    avg_confidence: float

    @property
    def answer_rate(self) -> float:
        return self.answered / self.total if self.total else 0.0

    @property
    def citation_accuracy(self) -> float:
        return self.citation_hits / self.total if self.total else 0.0

    @property
    def section_accuracy(self) -> float:
        return self.section_hits / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "answered": self.answered,
            "answer_rate": round(self.answer_rate, 3),
            "citation_hits": self.citation_hits,
            "citation_accuracy": round(self.citation_accuracy, 3),
            "section_hits": self.section_hits,
            "section_accuracy": round(self.section_accuracy, 3),
            "avg_confidence": round(self.avg_confidence, 3),
        }

    def log_summary(self) -> None:
        logger.info(
            f"Evaluation: {self.total} pairs | "
            f"Answer rate: {self.answer_rate:.1%} | "
            f"Citation accuracy: {self.citation_accuracy:.1%} | "
            f"Section accuracy: {self.section_accuracy:.1%} | "
            f"Avg confidence: {self.avg_confidence:.3f}"
        )


def compute_summary(results: list[EvalMetrics]) -> EvalSummary:
    """Aggregate individual eval results into a summary."""
    if not results:
        return EvalSummary(0, 0, 0, 0, 0.0)

    return EvalSummary(
        total=len(results),
        answered=sum(1 for r in results if r.has_answer),
        citation_hits=sum(1 for r in results if r.citation_hit),
        section_hits=sum(1 for r in results if r.section_hit),
        avg_confidence=sum(r.confidence for r in results) / len(results),
    )
