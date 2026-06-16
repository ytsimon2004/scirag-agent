"""Tests for scirag.mcp_server.server — the MCP tool layer.

The retrieval/index collaborators are mocked, so these are deterministic and need
neither Ollama nor a populated index (matching tests/test_pipeline.py). They lock
the JSON shapes each tool returns and the per-call project scoping. One live test
at the bottom self-skips unless Ollama is up *and* the active index has content.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scirag.mcp_server import server


def _make_node(pmid: str, score: float, content: str = "passage text") -> MagicMock:
    nws = MagicMock()
    nws.score = score
    nws.node.metadata = {
        "pmid": pmid,
        "title": f"Title {pmid}",
        "text_source": "results",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }
    nws.node.get_content.return_value = content
    return nws


class TestListProjects:
    def test_shape_and_active(self):
        with (
            patch("scirag.projects.get_active_project", return_value="human-rsc"),
            patch(
                "scirag.projects.list_projects",
                return_value=[{"name": "human-rsc", "description": "rsc"}, {"name": "demo"}],
            ),
        ):
            out = server.list_projects()
        assert out["active"] == "human-rsc"
        assert out["projects"] == [
            {"name": "human-rsc", "description": "rsc"},
            {"name": "demo", "description": ""},  # missing description defaults to ""
        ]

    def test_global_default_active_is_null(self):
        with (
            patch("scirag.projects.get_active_project", return_value=None),
            patch("scirag.projects.list_projects", return_value=[]),
        ):
            out = server.list_projects()
        assert out["active"] is None
        assert out["projects"] == []


class TestRetrieveChunks:
    def test_maps_nodes_to_records(self):
        nodes = [_make_node("111", 0.9, "chunk A"), _make_node("222", 0.5, "chunk B")]
        with (
            patch("scirag.retrieval.retriever.retrieve", return_value=nodes),
            patch("scirag.cite.citation", side_effect=lambda md: f"Author ({md['pmid']})"),
        ):
            out = server.retrieve_chunks("place cells")
        assert [r["id"] for r in out] == ["111", "222"]
        assert out[0] == {
            "citation": "Author (111)",
            "id": "111",
            "title": "Title 111",
            "text_source": "results",
            "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
            "score": 0.9,
            "text": "chunk A",
        }

    def test_empty_index_returns_empty_list(self):
        with patch("scirag.retrieval.retriever.retrieve", return_value=[]):
            assert server.retrieve_chunks("anything") == []


class TestGetRecord:
    def test_renames_pmid_to_id(self):
        art = {
            "pmid": "111",
            "title": "T",
            "year": "2020",
            "first_author": "Powell",
            "authors": "Powell, Doe",
            "text_source": "results",
            "chunks": ["c1", "c2"],
        }
        with patch("scirag.ingest.index.get_article_chunks", return_value=art):
            out = server.get_record("  111 ")  # also exercises .strip()
        assert "pmid" not in out
        assert out["id"] == "111"
        assert out["chunks"] == ["c1", "c2"]

    def test_missing_returns_none(self):
        with patch("scirag.ingest.index.get_article_chunks", return_value=None):
            assert server.get_record("999") is None


class TestIndexStatus:
    def test_counts_and_sorting(self):
        articles = [
            {
                "pmid": "1",
                "title": "Old",
                "year": "2018",
                "first_author": "A",
                "text_source": "abstract",
                "origin": "pubmed",
            },
            {
                "pmid": "2",
                "title": "New",
                "year": "2022",
                "first_author": "B",
                "text_source": "results",
                "origin": "pubmed",
            },
            {
                "pmid": "3",
                "title": "Mid",
                "year": "2020",
                "first_author": "C",
                "text_source": "fulltext",
                "origin": "biorxiv",
            },
        ]
        with patch("scirag.ingest.index.get_indexed_articles", return_value=articles):
            out = server.index_status()
        assert out["count"] == 3
        assert out["full_text"] == 2  # results + fulltext (not abstract)
        assert out["abstract_only"] == 1
        # sorted by year descending
        assert [a["id"] for a in out["articles"]] == ["2", "3", "1"]
        assert out["articles"][0]["origin"] == "pubmed"

    def test_empty_index(self):
        with patch("scirag.ingest.index.get_indexed_articles", return_value=[]):
            out = server.index_status()
        assert out == {"count": 0, "full_text": 0, "abstract_only": 0, "articles": []}


class TestScope:
    """_scope validates the project name and routes index reads to it per-call."""

    def test_empty_is_a_no_op(self):
        from scirag.projects import get_active_db_uri

        before = get_active_db_uri()
        with server._scope(""):
            assert get_active_db_uri() == before  # active project unchanged

    def test_named_project_overrides_db_uri(self):
        from scirag.projects import get_active_db_uri

        with patch("scirag.projects.list_projects", return_value=[{"name": "demo"}]):
            with server._scope("demo"):
                assert get_active_db_uri().endswith("projects/demo/lancedb")
        # restored afterwards
        assert not get_active_db_uri().endswith("projects/demo/lancedb")

    def test_unknown_project_raises(self):
        with patch("scirag.projects.list_projects", return_value=[{"name": "demo"}]):
            with pytest.raises(ValueError, match="Unknown project"):
                with server._scope("typo"):
                    pass

    def test_scope_passes_through_to_index_status(self):
        """A bad project name surfaces before any index read happens."""
        with patch("scirag.projects.list_projects", return_value=[{"name": "demo"}]):
            with pytest.raises(ValueError, match="Unknown project"):
                server.index_status(project="nope")


# --------------------------------------------------------------------------- #
# Live integration — exercises the real retrieval stack. Self-skips unless an
# Ollama embedding server is reachable AND the active index has content, so it
# only runs on a real dev box and never breaks CI.
# --------------------------------------------------------------------------- #


def _ollama_up() -> bool:
    import urllib.request

    try:
        from scirag.config import models_cfg

        base = models_cfg()["embeddings"].get("api_base", "http://localhost:11434")
    except Exception:
        base = "http://localhost:11434"
    try:
        urllib.request.urlopen(f"{base}/api/tags", timeout=1)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="Ollama embedding server not reachable")
def test_live_retrieve_chunks():
    from scirag.ingest.index import get_indexed_pmids

    if not get_indexed_pmids():
        pytest.skip("active index is empty")
    out = server.retrieve_chunks("the")
    assert isinstance(out, list)
    if out:
        r = out[0]
        assert {"citation", "id", "title", "text_source", "url", "score", "text"} <= r.keys()
