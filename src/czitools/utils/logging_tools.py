import sys
import logging

try:
    import colorlog

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


def set_logging(level=logging.INFO, format_string=None, colorize=True):
    """
    Configures a standard Python logger with customizable level and format.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG, logging.WARNING)
        format_string: Custom format string for log messages. If None, uses default format.
        colorize: Whether to colorize the output (requires colorlog package)

    Returns:
        logging.Logger: Configured standard Python logger instance.
    """

    # Create a logger
    logger = logging.getLogger("czitools")

    # Clear any existing handlers
    logger.handlers.clear()

    # Set the logging level
    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create formatter
    if colorize and HAS_COLORLOG:
        # Use colorlog for colored output
        if format_string is None:
            format_string = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        formatter = colorlog.ColoredFormatter(
            format_string,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    else:
        # Use standard formatter
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        formatter = logging.Formatter(format_string)

        # If colorlog is not available but colorize was requested, warn the user
        if colorize and not HAS_COLORLOG:
            print("Warning: colorlog package not found. Install with 'pip install colorlog' for colored output.")

    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger
