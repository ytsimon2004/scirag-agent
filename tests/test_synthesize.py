"""Tests for scirag.agents.synthesize._format_sources."""

from __future__ import annotations

from unittest.mock import MagicMock

from scirag.agents.synthesize import _format_sources


def _make_node(
    pmid: str,
    title: str,
    year: str,
    content: str,
    first_author: str = "Smith A",
    authors: str = "",
) -> MagicMock:
    nws = MagicMock()
    nws.node.metadata = {
        "pmid": pmid,
        "title": title,
        "year": year,
        "first_author": first_author,
        "authors": authors or first_author,
    }
    nws.node.get_content.return_value = content
    return nws


class TestFormatSources:
    def test_single_source(self):
        node = _make_node(
            "12345",
            "A study on memory",
            "2020",
            "Memory consolidation text.",
            first_author="Powell A",
            authors="Powell A, Jones B",
        )
        formatted = _format_sources([node])
        assert "Powell et al., 2020" in formatted  # author-year citation marker
        assert "[id: 12345]" in formatted  # id retained for traceability
        assert "A study on memory" in formatted
        assert "Memory consolidation text." in formatted

    def test_multiple_sources_separated(self):
        n1 = _make_node("1", "T1", "2019", "Content1", first_author="Doe J", authors="Doe J")
        n2 = _make_node("2", "T2", "2021", "Content2", first_author="Roe K", authors="Roe K, Q P")
        formatted = _format_sources([n1, n2])
        assert "---" in formatted
        assert "[id: 1]" in formatted
        assert "[id: 2]" in formatted
        assert "Doe, 2019" in formatted  # single author, no "et al."
        assert "Roe et al., 2021" in formatted

    def test_empty_nodes(self):
        assert _format_sources([]) == ""

    def test_missing_metadata_falls_back_to_id(self):
        node = MagicMock()
        node.node.metadata = {}
        node.node.get_content.return_value = "text"
        formatted = _format_sources([node])
        assert "[?]" in formatted  # no author → citation falls back to [pmid]
