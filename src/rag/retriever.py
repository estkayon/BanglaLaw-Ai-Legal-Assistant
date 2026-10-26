from dataclasses import dataclass

from loguru import logger

from src.ingestion.vectorstore import LegalVectorStore


@dataclass
class RetrievalResult:
    """A single retrieved document with its relevance score."""

    chunk_id: str
    text: str
    metadata: dict
    distance: float

    @property
    def confidence(self) -> float:
        """Convert cosine distance to a 0-1 confidence score."""
        return max(0.0, 1.0 - self.distance)

    @property
    def source_citation(self) -> str:
        """Human-readable citation for this chunk."""
        act_title = self.metadata.get("act_title", "Unknown Act")
        section_number = self.metadata.get("section_number", "?")
        return f"{act_title}, ধারা {section_number}"

    @property
    def source_url(self) -> str:
        return self.metadata.get("source_url", "")


class LegalRetriever:
    """Retrieves relevant legal text chunks from the vector store."""

    DEFAULT_TOP_K = 5
    CONFIDENCE_THRESHOLD = 0.25

    def __init__(
        self,
        vector_store: LegalVectorStore,
        top_k: int | None = None,
        confidence_threshold: float | None = None,
    ):
        self.vector_store = vector_store
        self.top_k = top_k or self.DEFAULT_TOP_K
        self.confidence_threshold = confidence_threshold or self.CONFIDENCE_THRESHOLD

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve relevant legal chunks for a query."""
        k = top_k or self.top_k
        raw_results = self.vector_store.query(query_text=query, n_results=k)

        results = []
        for i in range(len(raw_results["ids"])):
            result = RetrievalResult(
                chunk_id=raw_results["ids"][i],
                text=raw_results["documents"][i],
                metadata=raw_results["metadatas"][i],
                distance=raw_results["distances"][i],
            )
            results.append(result)

        logger.debug(
            f"Query: '{query[:50]}...' -> {len(results)} results, "
            f"top confidence: {results[0].confidence:.2f}"
            if results
            else "no results"
        )
        return results

    def retrieve_with_threshold(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Retrieve only results above the confidence threshold."""
        results = self.retrieve(query, top_k)
        filtered = [r for r in results if r.confidence >= self.confidence_threshold]

        if len(filtered) < len(results):
            logger.debug(
                f"Filtered {len(results) - len(filtered)} low-confidence results "
                f"(threshold={self.confidence_threshold})"
            )

        return filtered

    def is_confident(self, results: list[RetrievalResult]) -> bool:
        """Check if retrieval results are confident enough to generate an answer."""
        if not results:
            return False
        return results[0].confidence >= self.confidence_threshold

    def format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieved results into a context string for the LLM prompt."""
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            citation = result.source_citation
            context_parts.append(f"[Source {i}: {citation}]\n{result.text}\n")

        return "\n---\n".join(context_parts)

    def format_citations(self, results: list[RetrievalResult]) -> list[dict]:
        """Extract citation info from results for the API response."""
        return [
            {
                "source": result.source_citation,
                "section_number": result.metadata.get("section_number", ""),
                "act_title": result.metadata.get("act_title", ""),
                "url": result.source_url,
                "confidence": round(result.confidence, 3),
            }
            for result in results
        ]
