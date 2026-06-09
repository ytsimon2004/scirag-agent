"""Central config: loads YAML model/pipeline configs and env vars."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def models_cfg() -> dict[str, Any]:
    return _load_yaml(os.getenv("SCIREG_MODELS_CONFIG", "configs/models.yaml"))


@lru_cache(maxsize=1)
def pipeline_cfg() -> dict[str, Any]:
    return _load_yaml(os.getenv("SCIREG_PIPELINE_CONFIG", "configs/pipeline.yaml"))


def backend_for(agent: str) -> dict[str, Any]:
    """Resolve an agent role (e.g. 'synthesizer') to its concrete backend dict."""
    cfg = models_cfg()
    key = cfg["agents"][agent]
    return cfg["backends"][key]
