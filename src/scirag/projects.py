"""Project management — each project has its own isolated LanceDB index.

All data lives under ~/.scirag-agent/:
  lancedb/                  — default global index
  projects/<name>/lancedb/  — per-project indexes
  projects.json             — project registry
  .active_project           — name of the active project (plain text)
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional


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


def get_active_project() -> Optional[str]:
    p = _active_path()
    if p.exists():
        name = p.read_text().strip()
        return name or None
    return None


def get_active_db_uri() -> str:
    """Return the LanceDB URI for the active project, or the global default."""
    name = get_active_project()
    if name:
        return str(_data_dir() / "projects" / name / "lancedb")
    return str(_data_dir() / "lancedb")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def set_active_project(name: Optional[str]) -> None:
    p = _active_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if name:
        p.write_text(name)
    elif p.exists():
        p.unlink()


def create_project(name: str, description: str = "") -> dict:
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Name must be alphanumeric (hyphens/underscores ok): {name!r}")

    projects = list_projects()
    if any(p["name"] == name for p in projects):
        raise ValueError(f"Project {name!r} already exists")

    db_dir = _data_dir() / "projects" / name / "lancedb"
    db_dir.mkdir(parents=True, exist_ok=True)

    entry = {"name": name, "description": description, "created": str(date.today())}
    projects.append(entry)
    _registry_path().write_text(json.dumps(projects, indent=2))
    return entry


def delete_project(name: str) -> None:
    import shutil
    projects = list_projects()
    if not any(p["name"] == name for p in projects):
        raise ValueError(f"Project {name!r} not found")

    db_path = _data_dir() / "projects" / name
    if db_path.exists():
        shutil.rmtree(db_path)

    _registry_path().write_text(
        json.dumps([p for p in projects if p["name"] != name], indent=2)
    )
    if get_active_project() == name:
        set_active_project(None)
