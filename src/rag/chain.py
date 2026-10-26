import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from loguru import logger

from src.rag.prompt_templates import (
    get_low_confidence_response,
    get_qa_prompt,
    get_system_prompt,
)
from src.rag.retriever import LegalRetriever, RetrievalResult
from src.utils.config import settings


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""

    answer: str
    language: str
    citations: list[dict]
    confidence: float
    is_confident: bool
    query: str


class LegalRAGChain:
    """RAG chain that retrieves legal context and generates cited answers."""

    BANGLA_PATTERN = re.compile(r"[\u0980-\u09FF]")

    def __init__(
        self,
        retriever: LegalRetriever,
        model_name: str = "llama-3.1-70b-versatile",
        temperature: float = 0.1,
    ):
        self.retriever = retriever

        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

        self.llm = ChatGroq(
            model=model_name,
            api_key=settings.groq_api_key,
            temperature=temperature,
            max_tokens=2048,
        )
        logger.info(f"RAG chain initialized with model: {model_name}")

    def detect_language(self, text: str) -> str:
        """Detect if text is Bangla or English based on character presence."""
        bangla_chars = len(self.BANGLA_PATTERN.findall(text))
        total_alpha = sum(1 for c in text if c.isalpha())

        if total_alpha == 0:
            return "en"

        bangla_ratio = bangla_chars / total_alpha
        return "bn" if bangla_ratio > 0.3 else "en"

    def _compute_confidence(self, results: list[RetrievalResult]) -> float:
        """Compute overall confidence from retrieval results."""
        if not results:
            return 0.0
        # Weighted average: top results matter more
        weights = [1.0 / (i + 1) for i in range(len(results))]
        total_weight = sum(weights)
        weighted_conf = sum(r.confidence * w for r, w in zip(results, weights, strict=False))
        return round(weighted_conf / total_weight, 3)

    def _generate_answer(self, query: str, context: str, language: str) -> str:
        """Call the LLM to generate an answer from retrieved context."""
        system_prompt = get_system_prompt(language)
        qa_prompt = get_qa_prompt(language).format(context=context, question=query)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=qa_prompt),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def query(self, question: str) -> RAGResponse:
        """Run the full RAG pipeline: detect language -> retrieve -> generate."""
        language = self.detect_language(question)
        logger.info(f"Query [{language}]: {question[:80]}")

        # Retrieve relevant legal chunks
        results = self.retriever.retrieve(question)
        confidence = self._compute_confidence(results)
        is_confident = self.retriever.is_confident(results)

        # If not confident, return fallback
        if not is_confident:
            logger.warning(f"Low confidence ({confidence}) for query: {question[:50]}")
            return RAGResponse(
                answer=get_low_confidence_response(language),
                language=language,
                citations=[],
                confidence=confidence,
                is_confident=False,
                query=question,
            )

        # Build context and generate answer
        context = self.retriever.format_context(results)
        answer = self._generate_answer(question, context, language)
        citations = self.retriever.format_citations(results)

        logger.info(f"Generated answer with {len(citations)} citations (confidence={confidence})")

        return RAGResponse(
            answer=answer,
            language=language,
            citations=citations,
            confidence=confidence,
            is_confident=True,
            query=question,
        )
