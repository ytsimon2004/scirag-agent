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


_VALID_EFFORT = ("low", "medium", "high")

# --- Settings overlay -------------------------------------------------------
# Three resolution layers, in priority order:
#   1. session override   — set live in the shell (/model, /effort, /rag), volatile
#   2. persistent default — ~/.scirag-agent/settings.yaml, set via the CLI
#                           (`scirag model|effort|rag`); survives restarts
#   3. YAML default       — models.yaml / pipeline.yaml shipped/authored config
_SETTINGS_PATH = _USER_DIR / "settings.yaml"


def _load_settings() -> dict[str, Any]:
    if not _SETTINGS_PATH.exists():
        return {}
    try:
        with open(_SETTINGS_PATH) as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _save_settings(data: dict[str, Any]) -> None:
    _USER_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SETTINGS_PATH, "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


# Layer 2 — persistent defaults, loaded once at startup.
_settings = _load_settings()
_default_effort: str = (
    _settings.get("effort") if _settings.get("effort") in _VALID_EFFORT else "medium"
)
_default_backend: str | None = _settings.get("backend") or None
_default_retrieval: dict[str, Any] = dict(_settings.get("retrieval") or {})

# Layer 1 — session overrides (volatile; reset each launch).
_runtime_backend: dict[str, str] = {}  # agent -> backend key
_runtime_effort: str | None = None  # None = fall back to the persistent default
_runtime_retrieval: dict[str, Any] = {}  # retrieval param -> value


def set_agent_backend(agent: str, backend_key: str) -> None:
    """Override the backend for an agent for this session (does not persist)."""
    _runtime_backend[agent] = backend_key


def set_effort(level: str) -> None:
    """Set the session reasoning effort (low/medium/high). Raises on bad input."""
    if level not in _VALID_EFFORT:
        raise ValueError(f"effort must be one of {_VALID_EFFORT}, got {level!r}")
    global _runtime_effort
    _runtime_effort = level


def get_effort() -> str:
    """Current reasoning effort: session override, else persistent default."""
    return _runtime_effort if _runtime_effort is not None else _default_effort


def set_retrieval_param(key: str, value: Any) -> None:
    """Override one retrieval param for this session (does not persist)."""
    _runtime_retrieval[key] = value


def get_retrieval() -> dict[str, Any]:
    """Retrieval params: pipeline.yaml defaults < persistent defaults < session overrides."""
    cfg = dict(pipeline_cfg()["retrieval"])
    cfg.update(_default_retrieval)
    cfg.update(_runtime_retrieval)
    return cfg


def _resolve_backend_key(agent: str) -> str:
    """Backend key for an agent: session override < persistent default < models.yaml.
    A stale persisted default (backend no longer in models.yaml) is ignored."""
    cfg = models_cfg()
    key = _runtime_backend.get(agent) or _default_backend or cfg["agents"][agent]
    return key if key in cfg["backends"] else cfg["agents"][agent]


def backend_for(agent: str) -> dict[str, Any]:
    """Resolve an agent role (e.g. 'synthesizer') to its concrete backend dict."""
    return models_cfg()["backends"][_resolve_backend_key(agent)]


def active_backend_key(agent: str) -> str:
    """Return the backend key currently in use for an agent."""
    return _resolve_backend_key(agent)


# --- Persistent setters (used by the CLI; write settings.yaml) --------------
def set_default_effort(level: str) -> None:
    """Persist the default reasoning effort to settings.yaml."""
    if level not in _VALID_EFFORT:
        raise ValueError(f"effort must be one of {_VALID_EFFORT}, got {level!r}")
    global _default_effort
    _default_effort = level
    data = _load_settings()
    data["effort"] = level
    _save_settings(data)


def set_default_backend(key: str) -> None:
    """Persist the default LLM backend to settings.yaml. Raises on unknown key."""
    if key not in models_cfg()["backends"]:
        raise ValueError(f"unknown backend {key!r}")
    global _default_backend
    _default_backend = key
    data = _load_settings()
    data["backend"] = key
    _save_settings(data)


def set_default_retrieval_param(key: str, value: Any) -> None:
    """Persist one default retrieval param to settings.yaml."""
    _default_retrieval[key] = value
    data = _load_settings()
    data.setdefault("retrieval", {})[key] = value
    _save_settings(data)
