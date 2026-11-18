"""Load config/taxonomy.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.settings import PROJECT_ROOT

TAXONOMY_PATH = PROJECT_ROOT / "config" / "taxonomy.yaml"


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
