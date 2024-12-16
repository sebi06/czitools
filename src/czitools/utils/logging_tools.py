from loguru import logger as loguru_logger
import sys


def set_logging():
    """
    Configures the loguru logger to output logs to the standard output (stdout) with colorized formatting.
    The log format includes:
    - Time in green
    - Log level
    - Log message
    Returns:
        loguru.Logger: Configured loguru logger instance.
    """

    loguru_logger.remove()
    loguru_logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time}s</green> - <level>{level}</level> - <level>{message}</level>",
    )

    return loguru_logger
