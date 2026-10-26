import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas import (
    Citation,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from src.ingestion.vectorstore import LegalVectorStore
from src.rag.chain import LegalRAGChain
from src.rag.retriever import LegalRetriever

router = APIRouter()

# Module-level state — initialized by lifespan in main.py
_rag_chain: LegalRAGChain | None = None
_vector_store: LegalVectorStore | None = None

FEEDBACK_FILE = Path("logs/feedback.jsonl")


def init_rag_pipeline() -> None:
    """Initialize the RAG pipeline components. Called once at startup."""
    global _rag_chain, _vector_store

    _vector_store = LegalVectorStore()
    retriever = LegalRetriever(vector_store=_vector_store)
    _rag_chain = LegalRAGChain(retriever=retriever)
    logger.info("RAG pipeline initialized")


def get_rag_chain() -> LegalRAGChain:
    if _rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return _rag_chain


def get_vector_store() -> LegalVectorStore:
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return _vector_store


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check with vector store status."""
    store = get_vector_store()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        vector_store_count=store.count,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def query_law(request: QueryRequest):
    """Ask a legal question and get a cited answer."""
    chain = get_rag_chain()

    try:
        result = chain.query(request.question)
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to process your question. Please try again."
        ) from None

    return QueryResponse(
        answer=result.answer,
        language=result.language,
        citations=[Citation(**c) for c in result.citations],
        confidence=result.confidence,
        is_confident=result.is_confident,
        query=result.query,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Record user feedback on a response."""
    feedback_entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "query": request.query,
        "helpful": request.helpful,
        "comment": request.comment,
    }

    try:
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")
        logger.info(f"Feedback recorded: helpful={request.helpful}")
    except OSError as e:
        logger.error(f"Failed to write feedback: {e}")

    return FeedbackResponse()
