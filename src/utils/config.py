"""Configuration helpers for loading experiment settings from YAML files.

This module keeps config loading deliberately small and strict. It reads YAML
files into plain dictionaries so the rest of the code can work with one simple
config shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load one YAML config file and require that the parsed result is a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must parse to a dictionary.")
    return config
