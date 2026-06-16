"""Tests for scirag.sources.mendeley — offline import from the local library.

Builds a throwaway SQLite database with the same FTS *_content shadow tables the
real Mendeley Reference Manager keeps, points the module at it via MENDELEY_DB_PATH,
and exercises search/fetch end to end. No network, no real Mendeley install.
"""

from __future__ import annotations

import sqlite3

import pytest

from scirag.sources import mendeley


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_split_authors_western_names():
    assert mendeley._split_authors("Andrew S. Alexander Douglas A. Nitz") == [
        "Andrew S. Alexander",
        "Douglas A. Nitz",
    ]


def test_split_authors_no_initials():
    assert mendeley._split_authors("Jane Doe John Smith") == ["Jane Doe", "John Smith"]


def test_split_authors_empty():
    assert mendeley._split_authors("") == []


def test_to_ncbi_surname_initials():
    assert mendeley._to_ncbi("Andrew S. Alexander") == "Alexander AS"
    assert mendeley._to_ncbi("Jane Doe") == "Doe J"


def test_to_ncbi_single_token_passthrough():
    assert mendeley._to_ncbi("Consortium") == "Consortium"


def test_parse_ids_doi_and_pmid():
    blob = "0270-6474 10.1523/JNEUROSCI.1303-21.2021 34876468"
    assert mendeley._parse_ids(blob) == ("10.1523/JNEUROSCI.1303-21.2021", "34876468")


def test_parse_ids_doi_only():
    doi, pmid = mendeley._parse_ids("10.1101/2024.03.01.583000")
    assert doi == "10.1101/2024.03.01.583000"
    assert pmid == ""


def test_parse_ids_empty():
    assert mendeley._parse_ids("") == ("", "")


# ---------------------------------------------------------------------------
# database fixture
# ---------------------------------------------------------------------------

_RESULTS_TEXT = (
    "Abstract\nWe studied the retrosplenial cortex.\n\n"
    "Introduction\nBackground on spatial coding.\n\n"
    "Results\nWe recorded 80 neurons in the RSC.\n"
    "Firing rates rose after learning.\n\n"
    "Discussion\nThis extends prior work.\n"
)


def _make_db(path, docs):
    """Create a Mendeley-shaped SQLite DB. `docs` rows: (id, title, authors, journal,
    year, abstract, identifiers, fulltext)."""
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE documents_fts_content "
        "(c0 TEXT, c1 TEXT, c2 TEXT, c3 TEXT, c4 TEXT, c5 TEXT, c6 TEXT, c7 TEXT, c8 TEXT);"
        "CREATE TABLE files_fts_content (c0 TEXT, c1 TEXT, c2 TEXT);"
    )
    for doc_id, title, authors, journal, year, abstract, ids, fulltext in docs:
        con.execute(
            "INSERT INTO documents_fts_content (c0,c1,c2,c3,c4,c5,c8) VALUES (?,?,?,?,?,?,?)",
            (doc_id, title, authors, journal, year, abstract, ids),
        )
        if fulltext is not None:
            con.execute(
                "INSERT INTO files_fts_content (c0,c1,c2) VALUES (?,?,?)",
                (f"file-{doc_id}", doc_id, fulltext),
            )
    con.commit()
    con.close()


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A small Mendeley DB wired up via MENDELEY_DB_PATH; returns its path."""
    db = tmp_path / "library.db"
    _make_db(
        db,
        [
            (
                "1",
                "Retrosplenial cortex and memory",
                "Andrew S. Alexander Douglas A. Nitz",
                "J Neurosci",
                "2022",
                "An abstract about RSC.",
                "0270-6474 10.1523/jn.2022 34876468",
                _RESULTS_TEXT,
            ),
            (
                "2",
                "A bioRxiv preprint on place cells",
                "Jane Doe",
                "bioRxiv",
                "2024",
                "Preprint abstract.",
                "10.1101/2024.03.01.583000",
                None,
            ),
            (
                "3",
                "A book chapter with no identifiers",
                "John Smith",
                "Some Press",
                "2019",
                "Chapter abstract.",
                "",
                None,
            ),
        ],
    )
    monkeypatch.setenv("MENDELEY_DB_PATH", str(db))
    return db


# ---------------------------------------------------------------------------
# db_path / search / fetch
# ---------------------------------------------------------------------------


def test_db_path_env_override(library):
    assert mendeley.db_path() == library


def test_db_path_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("MENDELEY_DB_PATH", str(tmp_path / "nope.db"))
    assert mendeley.db_path() is None


def test_search_filters_by_words(library):
    assert mendeley.search("retrosplenial") == ["1"]
    assert mendeley.search("place cells") == ["2"]  # matches title words


def test_search_matches_author(library):
    assert mendeley.search("Alexander") == ["1"]


def test_search_empty_query_orders_by_year_desc(library):
    assert mendeley.search("") == ["2", "1", "3"]  # 2024, 2022, 2019


def test_search_no_match(library):
    assert mendeley.search("zzzznotapaper") == []


def test_fetch_pubmed_keyed_by_pmid(library):
    art = mendeley.fetch(["1"])[0]
    assert art.pmid == "34876468"
    assert art.source == "pubmed"
    assert art.doi == "10.1523/jn.2022"
    assert art.authors == ["Alexander AS", "Nitz DA"]
    assert art.year == "2022"
    assert "80 neurons in the RSC" in art.full_text  # Results section isolated
    assert "Background on spatial coding" not in art.full_text  # Introduction excluded


def test_fetch_biorxiv_keyed_by_doi(library):
    art = mendeley.fetch(["2"])[0]
    assert art.pmid == "10.1101/2024.03.01.583000"
    assert art.source == "biorxiv"
    assert art.url == "https://www.biorxiv.org/content/10.1101/2024.03.01.583000"


def test_fetch_fallback_local_key(library):
    art = mendeley.fetch(["3"])[0]
    assert art.pmid == "mendeley-3"
    assert art.source == "mendeley"


def test_search_and_fetch_roundtrip(library):
    arts = mendeley.search_and_fetch("retrosplenial")
    assert [a.pmid for a in arts] == ["34876468"]


def test_fetch_unknown_id_skipped(library):
    assert mendeley.fetch(["999"]) == []
