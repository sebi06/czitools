# -*- coding: utf-8 -*-
"""Logging helper and backend enum for the OME-Zarr export tools.

Vendored (with light edits) from ``czi_omezarr_utils.logging_utils`` in the
``omezarr_playground`` repository as part of czitools Stage 5.
"""

import logging
from enum import Enum, unique
from pathlib import Path


@unique
class omezarr_package(Enum):
    """Selectable OME-Zarr write backend."""

    OME_ZARR = 1
    NGFF_ZARR = 2


@unique
class compression_type(Enum):
    """Selectable compression type for OME-Zarr write backend."""

    BLOSC = 1
    ZSTD = 2
    NONE = 3


class _ExportLogFilter(logging.Filter):
    """Keep basic export logs concise without hiding warnings or errors."""

    def __init__(self, include_internal_info: bool) -> None:
        super().__init__()
        self.include_internal_info = include_internal_info

    def filter(self, record: logging.LogRecord) -> bool:
        return (
            self.include_internal_info
            or record.levelno >= logging.WARNING
            or record.name.startswith("czitools.export_tools")
        )


def _set_export_filter(handler: logging.Handler, include_internal_info: bool) -> None:
    """Replace this module's filter while preserving unrelated filters."""
    handler.filters = [item for item in handler.filters if not isinstance(item, _ExportLogFilter)]
    handler.addFilter(_ExportLogFilter(include_internal_info))


def setup_logging(
    log_file_path: str | Path | None = None,
    log_level: int = logging.INFO,
    force_reconfigure: bool = False,
    include_internal_info: bool = False,
    truncate_log_file: bool = False,
) -> logging.Logger:
    """Set up logging consistently across the export functions.

    Args:
        log_file_path (Optional[Union[str, Path]]): Path to a log file. If None,
            only a console handler is added.
        log_level (int): Logging level. Defaults to ``logging.INFO``.
        force_reconfigure (bool): Reconfigure even if logging is already set up.
        include_internal_info (bool): Include informational records from core
            readers and third-party libraries. Defaults to False.
        truncate_log_file (bool): Replace an existing log file when configuring
            its handler. Defaults to False.

    Returns:
        logging.Logger: The configured root logger.
    """
    root_logger = logging.getLogger()
    package_logger = logging.getLogger("czitools")

    # Several core modules configure the package logger during import. Export
    # logging also needs a root file handler, so retaining that package console
    # handler would emit every propagated record twice.
    package_logger.handlers.clear()
    package_logger.propagate = True

    has_file_handler = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root_logger.handlers
    )

    # Reconfigure if the target log file changed (different conversion run).
    if log_file_path is not None and has_file_handler and not force_reconfigure:
        target = str(Path(log_file_path))
        for h in root_logger.handlers:
            if isinstance(h, logging.FileHandler) and h.baseFilename != target:
                force_reconfigure = True
                break

    if has_file_handler and has_console_handler and not force_reconfigure:
        for handler in root_logger.handlers:
            _set_export_filter(handler, include_internal_info)
        return root_logger

    root_logger.setLevel(log_level)

    if force_reconfigure or not (has_file_handler and has_console_handler):
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        _set_export_filter(console_handler, include_internal_info)
        # Force UTF-8 on Windows where the default console encoding is cp1252.
        # This prevents UnicodeEncodeError when log messages contain non-ASCII
        # characters (e.g. the micro sign in "um" or emoji in validation messages).
        if hasattr(console_handler.stream, "reconfigure"):
            try:
                console_handler.stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        root_logger.addHandler(console_handler)

        if log_file_path:
            Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
            file_mode = "w" if truncate_log_file else "a"
            file_handler = logging.FileHandler(str(log_file_path), mode=file_mode, encoding="utf-8")
            file_handler.setFormatter(formatter)
            _set_export_filter(file_handler, include_internal_info)
            root_logger.addHandler(file_handler)

    return root_logger
