"""Central config: loads YAML model/pipeline configs and env vars."""

from __future__ import annotations

import os
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_USER_DIR = Path.home() / ".scirag-agent"
_HOME_ENV = _USER_DIR / ".env"
load_dotenv(_HOME_ENV)  # primary user config
load_dotenv(override=True)  # local .env overrides (dev use)

ROOT = Path(__file__).resolve().parents[2]


def _resolve_config(path: str | Path) -> Path:
    """Locate a config file. Absolute paths are used as-is; relative paths
    (e.g. ``configs/models.yaml``) are searched, in order:

    1. the dev checkout — ``<repo>/configs/...`` (only present when run from source),
    2. a user override — ``~/.scirag-agent/configs/...``,
    3. the default shipped inside the installed package — ``scirag/configs/...``.

    This is what lets the tool-installed `scirag` command run with no checkout
    on disk while a source checkout still picks up edits to ``./configs``.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    dev = ROOT / p
    if dev.exists():
        return dev
    user = _USER_DIR / p
    if user.exists():
        return user
    return Path(resources.files("scirag").joinpath(*p.parts))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(_resolve_config(path)) as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def models_cfg() -> dict[str, Any]:
    return _load_yaml(os.getenv("SCIRAG_MODELS_CONFIG", "configs/models.yaml"))


@lru_cache(maxsize=1)
def pipeline_cfg() -> dict[str, Any]:
    return _load_yaml(os.getenv("SCIRAG_PIPELINE_CONFIG", "configs/pipeline.yaml"))


_runtime_backend: dict[str, str] = {}  # agent -> backend key, overrides models.yaml


def set_agent_backend(agent: str, backend_key: str) -> None:
    """Override the backend for an agent for this session (does not modify models.yaml)."""
    _runtime_backend[agent] = backend_key


def backend_for(agent: str) -> dict[str, Any]:
    """Resolve an agent role (e.g. 'synthesizer') to its concrete backend dict."""
    cfg = models_cfg()
    key = _runtime_backend.get(agent) or cfg["agents"][agent]
    return cfg["backends"][key]


def active_backend_key(agent: str) -> str:
    """Return the backend key currently in use for an agent."""
    cfg = models_cfg()
    return _runtime_backend.get(agent) or cfg["agents"][agent]
