import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


class ResponseQualityMonitor:
    """Tracks response quality metrics over time from user queries and feedback."""

    METRICS_FILE = "quality_metrics.jsonl"

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / self.METRICS_FILE

    def log_query(
        self,
        query: str,
        language: str,
        confidence: float,
        is_confident: bool,
        num_citations: int,
    ) -> None:
        """Log a single query's quality metrics."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": "query",
            "query_length": len(query),
            "language": language,
            "confidence": round(confidence, 3),
            "is_confident": is_confident,
            "num_citations": num_citations,
        }
        self._append(entry)

    def log_feedback(self, query: str, helpful: bool) -> None:
        """Log user feedback."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": "feedback",
            "query_length": len(query),
            "helpful": helpful,
        }
        self._append(entry)

    def _append(self, entry: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_metrics(self) -> list[dict]:
        """Load all logged metrics."""
        if not self.metrics_path.exists():
            return []
        entries = []
        for line in self.metrics_path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                entries.append(json.loads(line))
        return entries

    def compute_summary(self) -> dict:
        """Compute aggregate quality stats from logged metrics."""
        metrics = self.load_metrics()
        if not metrics:
            return {"total_queries": 0, "total_feedback": 0}

        queries = [m for m in metrics if m["type"] == "query"]
        feedback = [m for m in metrics if m["type"] == "feedback"]

        total_queries = len(queries)
        confident_count = sum(1 for q in queries if q["is_confident"])
        avg_confidence = (
            sum(q["confidence"] for q in queries) / total_queries if total_queries else 0
        )
        avg_citations = (
            sum(q["num_citations"] for q in queries) / total_queries if total_queries else 0
        )

        # Language breakdown
        bn_count = sum(1 for q in queries if q["language"] == "bn")
        en_count = total_queries - bn_count

        # Feedback stats
        total_feedback = len(feedback)
        helpful_count = sum(1 for f in feedback if f["helpful"])
        helpful_rate = helpful_count / total_feedback if total_feedback else 0

        summary = {
            "total_queries": total_queries,
            "confident_rate": round(confident_count / total_queries, 3) if total_queries else 0,
            "avg_confidence": round(avg_confidence, 3),
            "avg_citations": round(avg_citations, 1),
            "bangla_queries": bn_count,
            "english_queries": en_count,
            "total_feedback": total_feedback,
            "helpful_rate": round(helpful_rate, 3),
        }

        logger.info(
            f"Quality: {total_queries} queries | "
            f"Confident: {summary['confident_rate']:.1%} | "
            f"Helpful: {summary['helpful_rate']:.1%} | "
            f"BN/EN: {bn_count}/{en_count}"
        )
        return summary
