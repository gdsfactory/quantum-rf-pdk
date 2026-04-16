"""Logging configuration for QPDK."""

import sys

from loguru import logger

# Fancy formatting for the logger
# You can customize this further if needed
FANCY_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


def configure_logger(level: str = "INFO", log_format: str = FANCY_FORMAT):
    """Configures the logger with a fancy format.

    Args:
        level: The logging level to use.
        log_format: The format to use.
    """
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
    )


# Initialize with default level
configure_logger()

__all__ = ["configure_logger", "logger"]
