"""Shared logger setup for Project Raven agents."""

import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"


def get_agent_logger(agent_name: str) -> logging.Logger:
    """
    Create or return a logger that writes to logs/<agent_name>.log.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(agent_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = LOGS_DIR / f"{agent_name}.log"

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
