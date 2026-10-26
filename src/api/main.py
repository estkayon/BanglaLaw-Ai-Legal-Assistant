from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from src.api.routes import init_rag_pipeline, router
from src.utils.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("Starting BanglaLaw API...")
    try:
        init_rag_pipeline()
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        logger.warning("API running without RAG — only /health will work")
    yield
    logger.info("Shutting down BanglaLaw API")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="BanglaLaw API",
        description="AI-powered legal information assistant for Bangladesh",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware (order matters: last added runs first)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # Routes
    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_debug,
    )
