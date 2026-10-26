from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """User's legal question."""

    question: str = Field(..., min_length=3, max_length=2000, examples=["ভাড়াটিয়া উচ্ছেদের আইন কি?"])
    language: str | None = Field(
        None, pattern="^(bn|en)$", description="Force response language. Auto-detected if omitted."
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of sources to retrieve.")


class Citation(BaseModel):
    """A single source citation."""

    source: str
    section_number: str
    act_title: str
    url: str
    confidence: float


class QueryResponse(BaseModel):
    """Response to a legal question."""

    answer: str
    language: str
    citations: list[Citation]
    confidence: float
    is_confident: bool
    query: str
    disclaimer: str = "⚠️ This is not legal advice. Consult a lawyer for your specific situation."
    share_text: str = ""


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    vector_store_count: int = 0


class FeedbackRequest(BaseModel):
    """User feedback on a response."""

    query: str
    helpful: bool
    comment: str = Field("", max_length=1000)


class FeedbackResponse(BaseModel):
    """Confirmation of feedback submission."""

    status: str = "recorded"
    message: str = "Thank you for your feedback."


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str = ""
