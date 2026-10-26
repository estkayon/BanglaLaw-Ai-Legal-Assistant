import json
from pathlib import Path

from loguru import logger

from src.evaluation.metrics import EvalMetrics, EvalSummary, compute_summary
from src.rag.chain import LegalRAGChain


def load_qa_pairs(filepath: Path | None = None) -> list[dict]:
    """Load QA pairs from JSON file."""
    path = filepath or Path("src/evaluation/qa_pairs.json")
    if not path.exists():
        raise FileNotFoundError(f"QA pairs file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(data)} QA pairs from {path}")
    return data


def evaluate_single(qa_pair: dict, chain: LegalRAGChain) -> EvalMetrics:
    """Evaluate a single QA pair against the RAG pipeline."""
    question = qa_pair["question"]
    result = chain.query(question)

    return EvalMetrics(
        qa_id=qa_pair["id"],
        question=question,
        expected_act=qa_pair.get("expected_act", ""),
        expected_section=qa_pair.get("expected_section", ""),
        predicted_answer=result.answer,
        predicted_citations=result.citations,
        confidence=result.confidence,
        is_confident=result.is_confident,
    )


def run_evaluation(
    chain: LegalRAGChain,
    qa_path: Path | None = None,
    category: str | None = None,
) -> tuple[list[EvalMetrics], EvalSummary]:
    """Run evaluation on all QA pairs and return results + summary."""
    qa_pairs = load_qa_pairs(qa_path)

    if category:
        qa_pairs = [q for q in qa_pairs if q.get("category") == category]
        logger.info(f"Filtered to {len(qa_pairs)} pairs for category: {category}")

    results: list[EvalMetrics] = []
    for i, qa in enumerate(qa_pairs, 1):
        logger.info(f"Evaluating [{i}/{len(qa_pairs)}]: {qa['question'][:50]}...")
        try:
            metrics = evaluate_single(qa, chain)
            results.append(metrics)
        except Exception as e:
            logger.error(f"Failed on {qa['id']}: {e}")
            continue

    summary = compute_summary(results)
    summary.log_summary()
    return results, summary


def save_results(
    results: list[EvalMetrics],
    summary: EvalSummary,
    output_dir: Path | None = None,
) -> Path:
    """Save evaluation results to JSON."""
    out_dir = output_dir or Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "eval_results.json"

    data = {
        "summary": summary.to_dict(),
        "results": [r.to_dict() for r in results],
    }

    output_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Evaluation results saved to {output_file}")
    return output_file


def log_to_mlflow(summary: EvalSummary, results: list[EvalMetrics]) -> None:
    """Log evaluation metrics to MLflow. Silently skips if MLflow is not configured."""
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow not installed. Skipping metric logging.")
        return

    from src.utils.config import settings

    if not settings.mlflow_tracking_uri:
        logger.warning("MLFLOW_TRACKING_URI not set. Skipping metric logging.")
        return

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run():
        mlflow.log_metrics(
            {
                "total_pairs": summary.total,
                "answer_rate": summary.answer_rate,
                "citation_accuracy": summary.citation_accuracy,
                "section_accuracy": summary.section_accuracy,
                "avg_confidence": summary.avg_confidence,
            }
        )

        # Log per-category breakdown
        categories = {r.to_dict().get("category", "unknown") for r in results}
        for cat in categories:
            cat_results = [r for r in results if r.to_dict().get("category") == cat]
            cat_summary = compute_summary(cat_results)
            mlflow.log_metrics(
                {
                    f"{cat}_citation_accuracy": cat_summary.citation_accuracy,
                    f"{cat}_answer_rate": cat_summary.answer_rate,
                }
            )

        logger.info("Metrics logged to MLflow")
