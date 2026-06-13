"""Tests for scirag.agents.pipeline — the shared RAG prompt-builder.

Retrieval and index lookups are mocked; these tests lock the relevance-gating
and message-assembly contract that the CLI, UI and MCP server all depend on.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scirag.agents import pipeline
from scirag.agents.pipeline import SYSTEM_GENERAL, SYSTEM_GROUNDED, prepare_answer


def _make_node(pmid: str, score: float, content: str = "passage text") -> MagicMock:
    nws = MagicMock()
    nws.score = score
    nws.node.metadata = {"pmid": pmid, "title": f"Title {pmid}", "year": "2020"}
    nws.node.get_content.return_value = content
    return nws


def _patch(*, nodes=None, indexed=True):
    """Patch the external collaborators of prepare_answer."""
    nodes = nodes if nodes is not None else []
    return (
        patch.object(pipeline, "retrieve", return_value=nodes),
        patch.object(pipeline, "get_indexed_pmids", return_value={"1"} if indexed else set()),
    )


def _run(**kw):
    p_ret, p_idx = _patch(**kw)
    with p_ret, p_idx:
        return prepare_answer(kw.pop("query", "How do place cells remap?"))


class TestGating:
    def test_high_score_grounds_answer(self):
        result = _run(nodes=[_make_node("111", 0.9)])
        assert result.use_rag is True
        assert result.nodes  # passages kept
        assert result.messages[0]["content"] == SYSTEM_GROUNDED
        user_msg = result.messages[-1]["content"]
        assert "[111]" in user_msg
        assert "Sources:" in user_msg

    def test_low_score_falls_back_to_general(self):
        result = _run(nodes=[_make_node("111", 0.05)])
        assert result.use_rag is False
        assert result.nodes == []
        assert result.messages[0]["content"] == SYSTEM_GENERAL
        # User turn is the bare query, no sources block.
        assert result.messages[-1]["content"] == "How do place cells remap?"

    def test_empty_index_skips_retrieval(self):
        p_ret, p_idx = _patch(indexed=False)
        with p_ret as mock_retrieve, p_idx:
            result = prepare_answer("any question")
        mock_retrieve.assert_not_called()
        assert result.use_rag is False

    def test_threshold_read_from_config(self):
        # A node at exactly 0.3 should clear the default threshold (>=).
        result = _run(nodes=[_make_node("1", 0.3)])
        assert result.use_rag is True


class TestMessageAssembly:
    def test_history_is_inserted_between_system_and_user(self):
        history = [
            {"role": "user", "content": "earlier q"},
            {"role": "assistant", "content": "earlier a"},
        ]
        p_ret, p_idx = _patch(nodes=[_make_node("1", 0.9)])
        with p_ret, p_idx:
            result = prepare_answer("new q", history)
        roles = [m["role"] for m in result.messages]
        assert roles == ["system", "user", "assistant", "user"]
        assert result.messages[1]["content"] == "earlier q"
