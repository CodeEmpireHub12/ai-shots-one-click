"""Shared config loader for Project Raven."""

from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "settings" / "config.yaml"


def load_config() -> Dict[str, Any]:
    """Load settings/config.yaml."""
    if not CONFIG_PATH.exists():
        return {}

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    return config
