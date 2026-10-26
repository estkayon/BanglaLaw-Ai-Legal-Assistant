from unittest.mock import MagicMock, patch

import pytest

from src.rag.chain import LegalRAGChain
from src.rag.prompt_templates import (
    get_low_confidence_response,
    get_qa_prompt,
    get_system_prompt,
)
from src.rag.retriever import LegalRetriever, RetrievalResult

# --- RetrievalResult tests ---


class TestRetrievalResult:
    def test_confidence_from_distance(self):
        result = RetrievalResult(
            chunk_id="1_s100",
            text="test",
            metadata={"act_title": "Test Act", "section_number": "৫"},
            distance=0.3,
        )
        assert result.confidence == pytest.approx(0.7, abs=0.01)

    def test_confidence_clamps_to_zero(self):
        result = RetrievalResult(
            chunk_id="1_s100",
            text="test",
            metadata={},
            distance=1.5,
        )
        assert result.confidence == 0.0

    def test_source_citation(self):
        result = RetrievalResult(
            chunk_id="755_s31189",
            text="test",
            metadata={"act_title": "মূল্য সংযোজন কর আইন", "section_number": "২"},
            distance=0.2,
        )
        assert "মূল্য সংযোজন কর আইন" in result.source_citation
        assert "ধারা ২" in result.source_citation

    def test_source_url(self):
        url = "https://bdlaws.minlaw.gov.bd/act-755/section-31189.html"
        result = RetrievalResult(
            chunk_id="1",
            text="test",
            metadata={"source_url": url},
            distance=0.1,
        )
        assert result.source_url == url

    def test_missing_metadata_defaults(self):
        result = RetrievalResult(
            chunk_id="1",
            text="test",
            metadata={},
            distance=0.5,
        )
        assert "Unknown Act" in result.source_citation
        assert result.source_url == ""


# --- LegalRetriever tests ---


class TestLegalRetriever:
    def _make_retriever(self, distances: list[float]) -> LegalRetriever:
        mock_store = MagicMock()
        mock_store.query.return_value = {
            "ids": [f"id_{i}" for i in range(len(distances))],
            "documents": [f"doc_{i}" for i in range(len(distances))],
            "metadatas": [
                {"act_title": "Test Act", "section_number": str(i), "source_url": ""}
                for i in range(len(distances))
            ],
            "distances": distances,
        }
        return LegalRetriever(vector_store=mock_store, confidence_threshold=0.5)

    def test_retrieve_returns_results(self):
        retriever = self._make_retriever([0.1, 0.3, 0.6])
        results = retriever.retrieve("test query")
        assert len(results) == 3
        assert results[0].distance == 0.1

    def test_retrieve_with_threshold_filters(self):
        retriever = self._make_retriever([0.1, 0.3, 0.8])
        results = retriever.retrieve_with_threshold("test query")
        # distance 0.8 -> confidence 0.2, below 0.5 threshold
        assert len(results) == 2

    def test_is_confident_true(self):
        retriever = self._make_retriever([0.2])
        results = retriever.retrieve("test")
        assert retriever.is_confident(results) is True

    def test_is_confident_false_empty(self):
        mock_store = MagicMock()
        mock_store.query.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }
        retriever = LegalRetriever(vector_store=mock_store)
        assert retriever.is_confident([]) is False

    def test_format_context(self):
        retriever = self._make_retriever([0.1])
        results = retriever.retrieve("test")
        context = retriever.format_context(results)
        assert "[Source 1:" in context
        assert "doc_0" in context

    def test_format_context_empty(self):
        mock_store = MagicMock()
        retriever = LegalRetriever(vector_store=mock_store)
        assert retriever.format_context([]) == ""

    def test_format_citations(self):
        retriever = self._make_retriever([0.2])
        results = retriever.retrieve("test")
        citations = retriever.format_citations(results)
        assert len(citations) == 1
        assert "source" in citations[0]
        assert "confidence" in citations[0]


# --- Prompt templates tests ---


class TestPromptTemplates:
    def test_system_prompt_bangla(self):
        prompt = get_system_prompt("bn")
        assert "BanglaLaw" in prompt
        assert "আইনি" in prompt

    def test_system_prompt_english(self):
        prompt = get_system_prompt("en")
        assert "BanglaLaw" in prompt
        assert "legal" in prompt.lower()

    def test_qa_prompt_has_placeholders(self):
        for lang in ("bn", "en"):
            prompt = get_qa_prompt(lang)
            assert "{context}" in prompt
            assert "{question}" in prompt

    def test_low_confidence_bangla(self):
        msg = get_low_confidence_response("bn")
        assert "পর্যাপ্ত তথ্য" in msg
        assert "⚠️" in msg

    def test_low_confidence_english(self):
        msg = get_low_confidence_response("en")
        assert "sufficient information" in msg
        assert "⚠️" in msg


# --- LegalRAGChain tests ---


class TestLegalRAGChain:
    def test_detect_language_bangla(self):
        # mock_retriever = MagicMock()
        with patch.object(LegalRAGChain, "__init__", lambda self, *a, **kw: None):
            chain = LegalRAGChain.__new__(LegalRAGChain)
            chain.BANGLA_PATTERN = LegalRAGChain.BANGLA_PATTERN
        assert chain.detect_language("ভাড়াটিয়া উচ্ছেদ আইন কি?") == "bn"

    def test_detect_language_english(self):
        with patch.object(LegalRAGChain, "__init__", lambda self, *a, **kw: None):
            chain = LegalRAGChain.__new__(LegalRAGChain)
            chain.BANGLA_PATTERN = LegalRAGChain.BANGLA_PATTERN
        assert chain.detect_language("What is the tenant eviction law?") == "en"

    def test_detect_language_mixed_defaults_bangla(self):
        with patch.object(LegalRAGChain, "__init__", lambda self, *a, **kw: None):
            chain = LegalRAGChain.__new__(LegalRAGChain)
            chain.BANGLA_PATTERN = LegalRAGChain.BANGLA_PATTERN
        assert chain.detect_language("আমার property র আইন কি?") == "bn"

    def test_detect_language_empty(self):
        with patch.object(LegalRAGChain, "__init__", lambda self, *a, **kw: None):
            chain = LegalRAGChain.__new__(LegalRAGChain)
            chain.BANGLA_PATTERN = LegalRAGChain.BANGLA_PATTERN
        assert chain.detect_language("123 !!!") == "en"

    def test_compute_confidence_weighted(self):
        with patch.object(LegalRAGChain, "__init__", lambda self, *a, **kw: None):
            chain = LegalRAGChain.__new__(LegalRAGChain)

        results = [
            RetrievalResult("1", "a", {}, 0.1),  # confidence 0.9
            RetrievalResult("2", "b", {}, 0.5),  # confidence 0.5
        ]
        conf = chain._compute_confidence(results)
        # Top result weighted more, so confidence > simple average of 0.7
        assert conf > 0.7

    def test_compute_confidence_empty(self):
        with patch.object(LegalRAGChain, "__init__", lambda self, *a, **kw: None):
            chain = LegalRAGChain.__new__(LegalRAGChain)
        assert chain._compute_confidence([]) == 0.0
