"""Tests for scirag.config — YAML loading and backend resolution."""

from __future__ import annotations


import pytest

from scirag import config as cfg_module


_MODELS_YAML = """
agents:
  synthesizer: claude-sonnet
  neuro_entity: local-qwen14b
backends:
  claude-sonnet:
    model: anthropic/claude-sonnet-4-6
  local-qwen14b:
    model: ollama/qwen2.5:14b-instruct-q4_K_M
    api_base: http://localhost:11434
embeddings:
  model: bge-m3
  api_base: http://localhost:11434
  dim: 1024
"""

_PIPELINE_YAML = """
retrieval:
  top_k: 20
  bm25_k: 20
  final_k: 10
  hybrid: true
"""


def _patched_load_yaml(yaml_str: str):
    """Return a _load_yaml that always returns the parsed yaml_str."""
    import yaml

    return lambda *_a, **_kw: yaml.safe_load(yaml_str)


def test_backend_for_frontier(monkeypatch):
    monkeypatch.setattr(
        cfg_module, "models_cfg", lambda: __import__("yaml").safe_load(_MODELS_YAML)
    )
    b = cfg_module.backend_for("synthesizer")
    assert b["model"] == "anthropic/claude-sonnet-4-6"
    assert "api_base" not in b


def test_backend_for_local(monkeypatch):
    monkeypatch.setattr(
        cfg_module, "models_cfg", lambda: __import__("yaml").safe_load(_MODELS_YAML)
    )
    b = cfg_module.backend_for("neuro_entity")
    assert b["model"].startswith("ollama/")
    assert b["api_base"] == "http://localhost:11434"


def test_backend_for_unknown_agent(monkeypatch):
    monkeypatch.setattr(
        cfg_module, "models_cfg", lambda: __import__("yaml").safe_load(_MODELS_YAML)
    )
    with pytest.raises(KeyError):
        cfg_module.backend_for("nonexistent_agent")
