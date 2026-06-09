"""Tests for scireg.agents.synthesize — LLM call is mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scireg.agents.synthesize import _format_sources, synthesize


def _make_node(pmid: str, title: str, year: str, content: str) -> MagicMock:
    nws = MagicMock()
    nws.node.metadata = {"pmid": pmid, "title": title, "year": year}
    nws.node.get_content.return_value = content
    return nws


class TestFormatSources:
    def test_single_source(self):
        node = _make_node("12345", "A study on memory", "2020", "Memory consolidation text.")
        formatted = _format_sources([node])
        assert "[12345]" in formatted
        assert "A study on memory" in formatted
        assert "2020" in formatted
        assert "Memory consolidation text." in formatted

    def test_multiple_sources_separated(self):
        n1 = _make_node("1", "T1", "2019", "Content1")
        n2 = _make_node("2", "T2", "2021", "Content2")
        formatted = _format_sources([n1, n2])
        assert "---" in formatted
        assert "[1]" in formatted
        assert "[2]" in formatted

    def test_empty_nodes(self):
        assert _format_sources([]) == ""

    def test_missing_pmid_uses_question_mark(self):
        node = MagicMock()
        node.node.metadata = {}
        node.node.get_content.return_value = "text"
        formatted = _format_sources([node])
        assert "[?]" in formatted


class TestSynthesize:
    def test_calls_complete_with_synthesizer_agent(self):
        node = _make_node("999", "Title", "2022", "Source text.")
        with patch("scireg.agents.synthesize.complete", return_value="Cited answer [999].") as mock_complete:
            answer = synthesize("What are place cells?", [node])
        mock_complete.assert_called_once()
        assert mock_complete.call_args[0][0] == "synthesizer"
        assert answer == "Cited answer [999]."

    def test_query_included_in_messages(self):
        node = _make_node("1", "T", "2020", "text")
        with patch("scireg.agents.synthesize.complete", return_value="answer") as mock_complete:
            synthesize("How do grid cells work?", [node])
        messages = mock_complete.call_args[0][1]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "How do grid cells work?" in user_msg

    def test_source_pmids_in_messages(self):
        nodes = [_make_node(str(i), f"T{i}", "2020", f"text {i}") for i in range(3)]
        with patch("scireg.agents.synthesize.complete", return_value="answer") as mock_complete:
            synthesize("query", nodes)
        messages = mock_complete.call_args[0][1]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        for i in range(3):
            assert f"[{i}]" in user_msg
