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


def configure_logger(
    level: str = "INFO",
    log_format: str = FANCY_FORMAT,
    *,
    colorize: bool = True,
):
    """Configures the logger with a fancy format.

    Args:
        level: The logging level to use.
        log_format: The format to use.
        colorize: Whether to emit ANSI colour escapes. Turn this off when
            stderr is a log file rather than a terminal, so the escape codes do
            not end up baked into the log.
    """
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=colorize,
    )


# Compact alternative to the fancy format, for when the log is a file that
# somebody will read while a long job is still running.
PLAIN_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"


# Initialize with default level
configure_logger()

__all__ = ["FANCY_FORMAT", "PLAIN_FORMAT", "configure_logger", "logger"]
