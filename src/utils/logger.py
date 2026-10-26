import sys

from loguru import logger

from src.utils.config import settings


def setup_logger() -> None:
    """Configure loguru with structured output for console and file."""
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Console output
    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.app_log_level,
        colorize=True,
    )

    # File output (production)
    if settings.is_production:
        logger.add(
            "logs/banglalaw_{time:YYYY-MM-DD}.log",
            format=log_format,
            level="INFO",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
        )


setup_logger()
