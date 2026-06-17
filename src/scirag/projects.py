"""Project management — each project has its own isolated LanceDB index.

All data lives under ~/.scirag-agent/:
  lancedb/                  — default global index
  projects/<name>/lancedb/  — per-project indexes
  projects.json             — project registry
  .active_project           — name of the active project (plain text)
"""

from __future__ import annotations

import contextvars
import json
import os
from contextlib import contextmanager
from datetime import date
from pathlib import Path

# Process-local override of the active project, used to scope a single operation
# (e.g. one MCP tool call) to a named project WITHOUT rewriting the shared
# .active_project file. A ContextVar isolates concurrent async tasks / threads, so
# parallel callers don't clobber each other or the interactive shell. `_UNSET`
# distinguishes "no override" from an explicit override to the global index (None).
_UNSET = object()
_project_override: contextvars.ContextVar = contextvars.ContextVar(
    "scirag_project_override", default=_UNSET
)


def _data_dir() -> Path:
    return Path.home() / ".scirag-agent"


def _registry_path() -> Path:
    return _data_dir() / "projects.json"


def _active_path() -> Path:
    return _data_dir() / ".active_project"


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def list_projects() -> list[dict]:
    p = _registry_path()
    return json.loads(p.read_text()) if p.exists() else []


def get_active_project() -> str | None:
    p = _active_path()
    if p.exists():
        name = p.read_text().strip()
        return name or None
    return None


def _resolve_active_name() -> str | None:
    """Resolve the active project name, highest precedence first:
      1. a `using_project()` scope (process-local ContextVar),
      2. the ``SCIRAG_PROJECT`` env var — a name, or empty string for the global
         index (this is how the CLI's ``--project``/``--global`` scope a one-shot run,
         and it crosses the `ui` Chainlit subprocess boundary via inherited env),
      3. the persisted active project (``.active_project``).
    None of these mutate the persisted active project. Returns None for the global index.
    """
    override = _project_override.get()
    if override is not _UNSET:
        return override
    env_proj = os.environ.get("SCIRAG_PROJECT")
    return (env_proj or None) if env_proj is not None else get_active_project()


def get_active_db_uri() -> str:
    """Return the LanceDB URI for the active project, or the global default.

    The active project is resolved by `_resolve_active_name()`; None means the
    global index. Neither path mutates the persisted active project.
    """
    name = _resolve_active_name()
    if name:
        return str(_data_dir() / "projects" / name / "lancedb")
    return str(_data_dir() / "lancedb")


def get_project(name: str) -> dict | None:
    """Return the registry entry for `name`, or None if it doesn't exist."""
    return next((p for p in list_projects() if p["name"] == name), None)


def get_active_system_prompt() -> str:
    """Return the active project's system prompt, or "" (use the built-in default).

    Resolves the active project with the same precedence as `get_active_db_uri()`,
    so CLI ``--project`` and MCP ``using_project()`` scoping pick up the right prompt.
    """
    name = _resolve_active_name()
    if not name:
        return ""
    entry = get_project(name)
    return (entry or {}).get("system_prompt", "")


@contextmanager
def using_project(name: str | None):
    """Temporarily resolve the index to `name` (None = global default) for the
    current execution context only, leaving the persisted active project untouched.

    ContextVar-based, so it's safe under concurrency: parallel async tasks / threads
    each get their own scope and never clobber the .active_project file or each
    other. Only affects `get_active_db_uri()` (and thus all index reads/writes that
    route through it); the shell prompt's `get_active_project()` is unaffected.
    """
    token = _project_override.set(name)
    try:
        yield
    finally:
        _project_override.reset(token)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def set_active_project(name: str | None) -> None:
    p = _active_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if name:
        p.write_text(name)
    elif p.exists():
        p.unlink()


def create_project(name: str, description: str = "", system_prompt: str = "") -> dict:
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Name must be alphanumeric (hyphens/underscores ok): {name!r}")

    projects = list_projects()
    if any(p["name"] == name for p in projects):
        raise ValueError(f"Project {name!r} already exists")

    db_dir = _data_dir() / "projects" / name / "lancedb"
    db_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "name": name,
        "description": description,
        "created": str(date.today()),
        "system_prompt": system_prompt,
    }
    projects.append(entry)
    _registry_path().write_text(json.dumps(projects, indent=2))
    return entry


def set_project_system_prompt(name: str, text: str) -> None:
    """Set (or clear, with "") the system prompt for project `name`."""
    projects = list_projects()
    entry = next((p for p in projects if p["name"] == name), None)
    if entry is None:
        raise ValueError(f"Project {name!r} not found")
    entry["system_prompt"] = text
    _registry_path().write_text(json.dumps(projects, indent=2))


def delete_project(name: str) -> None:
    import shutil

    projects = list_projects()
    if not any(p["name"] == name for p in projects):
        raise ValueError(f"Project {name!r} not found")

    db_path = _data_dir() / "projects" / name
    if db_path.exists():
        shutil.rmtree(db_path)

    _registry_path().write_text(json.dumps([p for p in projects if p["name"] != name], indent=2))
    if get_active_project() == name:
        set_active_project(None)
