import time
from collections import defaultdict

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter per client IP."""

    def __init__(self, app, max_requests: int | None = None, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests or settings.api_rate_limit
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_rate_limited(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self.window

        # Remove expired entries
        self.requests[client_ip] = [t for t in self.requests[client_ip] if t > window_start]

        if len(self.requests[client_ip]) >= self.max_requests:
            return True

        self.requests[client_ip].append(now)
        return False

    async def dispatch(self, request: Request, call_next) -> Response:
        client_ip = self._get_client_ip(request)

        if self._is_rate_limited(client_ip):
            logger.warning(f"Rate limited: {client_ip}")
            return Response(
                content='{"error": "Rate limit exceeded. Please try again later."}',
                status_code=429,
                media_type="application/json",
            )

        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with timing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.time()
        method = request.method
        path = request.url.path

        response = await call_next(request)

        duration_ms = round((time.time() - start) * 1000, 1)
        logger.info(f"{method} {path} -> {response.status_code} ({duration_ms}ms)")

        return response
