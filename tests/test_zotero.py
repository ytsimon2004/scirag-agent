"""Tests for scirag.sources.zotero — offline import from the local library.

Builds a throwaway SQLite database with the same normalised schema Zotero keeps
(items / itemData / creators / itemAttachments / deletedItems), points the module
at it via ZOTERO_DB_PATH, and lays attachment text down as a `.zotero-ft-cache`
file under a derived `storage/` dir. PubMed lookups are patched, so no network.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from scirag.sources import zotero


# ---------------------------------------------------------------------------
# pure helper
# ---------------------------------------------------------------------------


def test_to_ncbi_surname_initials():
    assert zotero._to_ncbi("Andrew S.", "Alexander") == "Alexander AS"
    assert zotero._to_ncbi("Jane", "Doe") == "Doe J"


def test_to_ncbi_institution_passthrough():
    # A single-field creator (institution) has no first name.
    assert zotero._to_ncbi("", "Allen Institute") == "Allen Institute"


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

_BIORXIV_FT = (
    "bioRxiv preprint doi: https://doi.org/10.1101/2025.06.24.661247; "
    "this version posted June 24, 2025.\n\n"
    "Abstract\nBrain-wide inputs to the RSC.\n\n"
    "Results\nWe traced 30 input regions.\nLayer 5 neurons dominated.\n\n"
    "Discussion\nInputs are specialized.\n"
)

# typeName -> itemTypeID
_TYPES = {"journalArticle": 1, "preprint": 2, "attachment": 3, "note": 4}


def _make_db(path, refs, deleted=()):
    """Create a Zotero-shaped SQLite DB plus a storage/ dir of ft-cache files.

    Each `refs` entry is a dict: id, type, fields{name:value}, creators[(first,last)],
    fulltext (str|None -> attachment with a .zotero-ft-cache).
    """
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE itemTypes (itemTypeID INTEGER, typeName TEXT);"
        "CREATE TABLE items (itemID INTEGER, itemTypeID INTEGER, key TEXT);"
        "CREATE TABLE fields (fieldID INTEGER, fieldName TEXT);"
        "CREATE TABLE itemDataValues (valueID INTEGER, value TEXT);"
        "CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);"
        "CREATE TABLE creators (creatorID INTEGER, firstName TEXT, lastName TEXT, fieldMode INTEGER);"
        "CREATE TABLE itemCreators (itemID INTEGER, creatorID INTEGER, creatorTypeID INTEGER, orderIndex INTEGER);"
        "CREATE TABLE itemAttachments (itemID INTEGER, parentItemID INTEGER, contentType TEXT, path TEXT);"
        "CREATE TABLE deletedItems (itemID INTEGER);"
    )
    for name, tid in _TYPES.items():
        con.execute("INSERT INTO itemTypes VALUES (?,?)", (tid, name))

    field_ids: dict[str, int] = {}
    vid = 0
    cid = 0
    att_item = 5000
    storage = path.parent / "storage"
    for ref in refs:
        iid = ref["id"]
        con.execute(
            "INSERT INTO items VALUES (?,?,?)",
            (iid, _TYPES[ref.get("type", "journalArticle")], f"KEY{iid}"),
        )
        for fname, fval in ref.get("fields", {}).items():
            if fname not in field_ids:
                field_ids[fname] = len(field_ids) + 1
                con.execute("INSERT INTO fields VALUES (?,?)", (field_ids[fname], fname))
            vid += 1
            con.execute("INSERT INTO itemDataValues VALUES (?,?)", (vid, fval))
            con.execute("INSERT INTO itemData VALUES (?,?,?)", (iid, field_ids[fname], vid))
        for order, (first, last) in enumerate(ref.get("creators", [])):
            cid += 1
            con.execute("INSERT INTO creators VALUES (?,?,?,?)", (cid, first, last, 0))
            con.execute("INSERT INTO itemCreators VALUES (?,?,?,?)", (iid, cid, 1, order))
        if ref.get("fulltext") is not None:
            att_item += 1
            att_key = f"ATT{iid}"
            con.execute(
                "INSERT INTO items VALUES (?,?,?)", (att_item, _TYPES["attachment"], att_key)
            )
            con.execute(
                "INSERT INTO itemAttachments VALUES (?,?,?,?)",
                (att_item, iid, "application/pdf", "storage:paper.pdf"),
            )
            adir = storage / att_key
            adir.mkdir(parents=True, exist_ok=True)
            (adir / "paper.pdf").write_bytes(b"%PDF-1.4 stub")
            (adir / ".zotero-ft-cache").write_text(ref["fulltext"])
    for d in deleted:
        con.execute("INSERT INTO deletedItems VALUES (?)", (d,))
    con.commit()
    con.close()


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A small Zotero DB wired up via ZOTERO_DB_PATH; returns its path."""
    db = tmp_path / "zotero.sqlite"
    _make_db(
        db,
        [
            {  # 10: PMID in extra -> pubmed, full text from ft-cache
                "id": 10,
                "fields": {
                    "title": "Retrosplenial cortex and memory",
                    "abstractNote": "An abstract about RSC.",
                    "publicationTitle": "J Neurosci",
                    "date": "2022-02-02",
                    "DOI": "10.1523/jn.2022",
                    "extra": "PMID: 34876468",
                },
                "creators": [("Andrew S.", "Alexander"), ("Douglas A.", "Nitz")],
                "fulltext": _RESULTS_TEXT,
            },
            {  # 11: bioRxiv DOI in metadata -> biorxiv
                "id": 11,
                "fields": {
                    "title": "A preprint on place cells",
                    "date": "2024-03-01",
                    "DOI": "10.1101/2024.03.01.583000",
                },
                "creators": [("Jane", "Doe")],
                "fulltext": None,
            },
            {  # 12: title only, DOI only in the attached PDF (bioRxiv watermark)
                "id": 12,
                "fields": {
                    "title": "Brain-wide input patterns to the retrosplenial cortex",
                },
                "creators": [("Yu-Ting", "Wei")],
                "fulltext": _BIORXIV_FT,
            },
            {  # 13: publisher DOI, no PMID -> resolved to a PMID (patched)
                "id": 13,
                "fields": {
                    "title": "A Crossref import",
                    "date": "2021",
                    "DOI": "10.1038/s41467-021-00001-0",
                },
                "creators": [("Sam", "Trask")],
                "fulltext": None,
            },
            {  # 14: nothing resolvable -> local zotero-<id>
                "id": 14,
                "fields": {"title": "Unpublished draft notes xyz"},
                "creators": [("Ann", "Author")],
                "fulltext": None,
            },
            {  # 15: trashed -> never surfaced by search
                "id": 15,
                "fields": {"title": "Retrosplenial trashed paper"},
                "creators": [],
                "fulltext": None,
            },
        ],
        deleted=[15],
    )
    monkeypatch.setenv("ZOTERO_DB_PATH", str(db))
    # Default: no PubMed match (offline). Individual tests override as needed.
    with (
        patch("scirag.sources.pubmed.search", return_value=[]),
        patch("scirag.sources.pubmed.fetch", return_value=[]),
    ):
        yield db


# ---------------------------------------------------------------------------
# db_path / search
# ---------------------------------------------------------------------------


def test_db_path_env_override(library):
    assert zotero.db_path() == library


def test_db_path_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOTERO_DB_PATH", str(tmp_path / "nope.sqlite"))
    assert zotero.db_path() is None


def test_search_filters_by_words(library):
    assert zotero.search("memory") == ["10"]


def test_search_matches_author(library):
    assert zotero.search("Alexander") == ["10"]


def test_search_excludes_trashed_item(library):
    # "retrosplenial" appears in items 10, 12, and the trashed item 15; the trashed
    # one is filtered out (10 before 12: 2022 sorts above 12's blank year).
    results = zotero.search("retrosplenial")
    assert results == ["10", "12"]
    assert "15" not in results


def test_search_no_match(library):
    assert zotero.search("zzzznotapaper") == []


# ---------------------------------------------------------------------------
# fetch — key/source resolution per item
# ---------------------------------------------------------------------------


def test_fetch_pmid_from_extra(library):
    art = zotero.fetch(["10"])[0]
    assert art.pmid == "34876468"
    assert art.source == "pubmed"
    assert art.doi == "10.1523/jn.2022"
    assert art.authors == ["Alexander AS", "Nitz DA"]
    assert "80 neurons in the RSC" in art.full_text  # Results isolated from ft-cache
    assert "Background on spatial coding" not in art.full_text


def test_fetch_biorxiv_doi_from_metadata(library):
    art = zotero.fetch(["11"])[0]
    assert art.pmid == "10.1101/2024.03.01.583000"
    assert art.source == "biorxiv"
    assert art.url == "https://www.biorxiv.org/content/10.1101/2024.03.01.583000"


def test_fetch_biorxiv_doi_extracted_from_pdf(library):
    # Item 12 has only a title; the DOI is mined from the attached PDF watermark,
    # and the year is derived from the preprint DOI's embedded date.
    art = zotero.fetch(["12"])[0]
    assert art.pmid == "10.1101/2025.06.24.661247"
    assert art.source == "biorxiv"
    assert art.doi == "10.1101/2025.06.24.661247"
    assert art.year == "2025"
    assert "30 input regions" in art.full_text


def test_fetch_publisher_doi_resolves_to_pmid(library):
    with patch("scirag.sources.pubmed.search", return_value=["33333333"]):
        art = zotero.fetch(["13"])[0]
    assert art.pmid == "33333333"
    assert art.source == "pubmed"
    assert art.doi == "10.1038/s41467-021-00001-0"


def test_fetch_title_resolution_confident_match(library):
    from scirag.sources.pubmed import Article

    real = Article(pmid="44444444", title="Unpublished draft notes xyz", abstract="")
    with (
        patch("scirag.sources.pubmed.search", return_value=["44444444"]),
        patch("scirag.sources.pubmed.fetch", return_value=[real]),
    ):
        art = zotero.fetch(["14"])[0]
    assert art.pmid == "44444444"
    assert art.source == "pubmed"


def test_fetch_local_key_when_unresolvable(library):
    # Default fixture patches PubMed to return nothing -> stays zotero-<id>.
    art = zotero.fetch(["14"])[0]
    assert art.pmid == "zotero-14"
    assert art.source == "zotero"


def test_search_and_fetch_roundtrip(library):
    arts = zotero.search_and_fetch("memory")
    assert [a.pmid for a in arts] == ["34876468"]


def test_fetch_empty_list(library):
    assert zotero.fetch([]) == []
