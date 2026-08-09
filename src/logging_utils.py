"""
Central logging setup. Import `setup_logging()` once (e.g. at the top of a
notebook or script) and then use `logger.info(...)` instead of `print(...)`
everywhere else.
"""

import logging
from pathlib import Path

from src import config


def setup_logging(log_file=None, level=None):
    """
    Configure and return the shared 'biohub' logger.

    Safe to call multiple times (e.g. on notebook re-run) — it won't
    attach duplicate handlers.
    """
    log_file = Path(log_file) if log_file is not None else Path(config.LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    level = level or config.LOG_LEVEL

    logger = logging.getLogger("biohub")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
