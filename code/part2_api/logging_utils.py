from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    logger.info("Log file: %s", log_file)
    return logger
