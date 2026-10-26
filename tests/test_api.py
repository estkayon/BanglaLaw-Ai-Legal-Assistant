from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.schemas import QueryRequest


@pytest.fixture
def client():
    """Test client with RAG pipeline mocked out."""
    app = create_app()

    # Override lifespan initialization
    with (
        patch("src.api.routes._rag_chain") as mock_chain,
        patch("src.api.routes._vector_store") as mock_store,
    ):
        mock_store.count = 100

        mock_result = MagicMock()
        mock_result.answer = "Test answer about law."
        mock_result.language = "en"
        mock_result.citations = [
            {
                "source": "Test Act, ধারা ১",
                "section_number": "১",
                "act_title": "Test Act",
                "url": "https://bdlaws.minlaw.gov.bd/act-1/section-1.html",
                "confidence": 0.85,
            }
        ]
        mock_result.confidence = 0.85
        mock_result.is_confident = True
        mock_result.query = "What is the law?"
        mock_chain.query.return_value = mock_result

        yield TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "vector_store_count" in data


class TestQueryEndpoint:
    def test_query_success(self, client):
        response = client.post(
            "/api/v1/query",
            json={"question": "What is the tenant eviction law?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "confidence" in data
        assert "language" in data

    def test_query_with_language(self, client):
        response = client.post(
            "/api/v1/query",
            json={"question": "ভাড়াটিয়া আইন কি?", "language": "bn"},
        )
        assert response.status_code == 200

    def test_query_too_short(self, client):
        response = client.post(
            "/api/v1/query",
            json={"question": "ab"},
        )
        assert response.status_code == 422

    def test_query_invalid_language(self, client):
        response = client.post(
            "/api/v1/query",
            json={"question": "test question", "language": "fr"},
        )
        assert response.status_code == 422


class TestFeedbackEndpoint:
    def test_feedback_success(self, client, tmp_path):
        with patch("src.api.routes.FEEDBACK_FILE", tmp_path / "feedback.jsonl"):
            response = client.post(
                "/api/v1/feedback",
                json={"query": "test question", "helpful": True, "comment": "Great!"},
            )
        assert response.status_code == 200
        assert response.json()["status"] == "recorded"

    def test_feedback_without_comment(self, client, tmp_path):
        with patch("src.api.routes.FEEDBACK_FILE", tmp_path / "feedback.jsonl"):
            response = client.post(
                "/api/v1/feedback",
                json={"query": "test", "helpful": False},
            )
        assert response.status_code == 200


class TestQueryRequestValidation:
    def test_valid_request(self):
        req = QueryRequest(question="What is property law?")
        assert req.top_k == 5
        assert req.language is None

    def test_custom_top_k(self):
        req = QueryRequest(question="test question", top_k=10)
        assert req.top_k == 10

    def test_top_k_bounds(self):
        with pytest.raises(ValueError):
            QueryRequest(question="test", top_k=0)
        with pytest.raises(ValueError):
            QueryRequest(question="test", top_k=25)
