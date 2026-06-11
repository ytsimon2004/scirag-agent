"""Tests for scirag.sources.pdf — PDF text extraction and Results-section isolation."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch


from scirag.sources.pdf import (
    _extract_doi,
    _extract_year,
    _guess_title,
    _pmid_from_stem,
    extract_results_section,
    load_pdf_as_article,
    load_pdf_directory,
)
from scirag.sources.pubmed import Article


# ---------------------------------------------------------------------------
# extract_results_section — plain text
# ---------------------------------------------------------------------------

_FULL_PAPER = """
Abstract
We studied place cells in the rat hippocampus.

Introduction
Spatial navigation depends on hippocampal circuits.
Grid cells provide metric information.

Results
We recorded 120 place cells from dorsal CA1.
Place fields were stable across multiple sessions.
Remapping occurred after context change.

Discussion
These findings extend previous work on spatial coding.

Methods
Rats were implanted with tetrodes.
"""

_PAPER_RESULTS_AND_DISCUSSION = """
Introduction
Background information here.

Results and Discussion
We found significant changes in firing rate.
Grid spacing increased with environment size.

References
1. O'Keefe, J. (1971).
"""

_PAPER_NO_RESULTS = """
Introduction
We studied something interesting.

Methods
We used patch clamp electrophysiology.

References
Smith et al. 2020.
"""


def test_extract_results_section_standard():
    text = extract_results_section(_FULL_PAPER)
    assert "120 place cells" in text
    assert "stable across multiple sessions" in text
    assert "Remapping occurred" in text


def test_extract_results_section_excludes_other_sections():
    text = extract_results_section(_FULL_PAPER)
    assert "Spatial navigation" not in text  # Introduction
    assert "tetrodes" not in text  # Methods
    assert "previous work" not in text  # Discussion


def test_extract_results_and_discussion_variant():
    text = extract_results_section(_PAPER_RESULTS_AND_DISCUSSION)
    assert "firing rate" in text
    assert "Grid spacing" in text
    assert "Background information" not in text


def test_extract_results_section_not_found():
    assert extract_results_section(_PAPER_NO_RESULTS) == ""


def test_extract_results_section_empty_string():
    assert extract_results_section("") == ""


# ---------------------------------------------------------------------------
# _pmid_from_stem
# ---------------------------------------------------------------------------


def test_pmid_from_stem_numeric():
    assert _pmid_from_stem("12345678") == "12345678"


def test_pmid_from_stem_non_numeric():
    result = _pmid_from_stem("nature_paper_2024")
    assert result.startswith("pdf:")
    assert len(result) == 12  # "pdf:" + 8 hex chars


def test_pmid_from_stem_deterministic():
    assert _pmid_from_stem("foo") == _pmid_from_stem("foo")


# ---------------------------------------------------------------------------
# load_pdf_as_article — pypdf is mocked
# ---------------------------------------------------------------------------


def _make_pdf_reader(text: str) -> MagicMock:
    page = MagicMock()
    page.extract_text.return_value = text
    reader = MagicMock()
    reader.pages = [page]
    return reader


@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_numeric_filename_becomes_pmid(mock_reader_cls, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_FULL_PAPER)
    pdf = tmp_path / "12345678.pdf"
    pdf.touch()
    article = load_pdf_as_article(pdf)
    assert article.pmid == "12345678"


@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_results_section_extracted(mock_reader_cls, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_FULL_PAPER)
    pdf = tmp_path / "paper.pdf"
    pdf.touch()
    article = load_pdf_as_article(pdf)
    assert "120 place cells" in article.full_text
    assert "Spatial navigation" not in article.full_text


@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_falls_back_when_no_results_section(mock_reader_cls, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_PAPER_NO_RESULTS)
    pdf = tmp_path / "paper.pdf"
    pdf.touch()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        article = load_pdf_as_article(pdf)
    assert any("no Results section" in str(w.message) for w in caught)
    assert article.full_text == ""  # no fallback — only Results section allowed


@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_abstract_is_empty(mock_reader_cls, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_FULL_PAPER)
    pdf = tmp_path / "paper.pdf"
    pdf.touch()
    article = load_pdf_as_article(pdf)
    assert article.abstract == ""


# ---------------------------------------------------------------------------
# metadata extraction helpers
# ---------------------------------------------------------------------------

_PAPER_WITH_DOI = """
Article https://doi.org/10.1038/s41467-026-70762-z
Optogenetic inhibition of the retrosplenial cortex disrupts memory retrieval

Results
We found a clear effect.
Discussion
It matters.
"""


def test_extract_doi_found():
    assert _extract_doi(_PAPER_WITH_DOI) == "10.1038/s41467-026-70762-z"


def test_extract_doi_strips_trailing_punctuation():
    assert _extract_doi("see 10.1234/abc.def).") == "10.1234/abc.def"


def test_extract_doi_none():
    assert _extract_doi(_FULL_PAPER) == ""


def test_extract_year():
    assert _extract_year("Published 2021 in Nature") == "2021"


def test_extract_year_none():
    assert _extract_year("no dates here") == ""


def test_guess_title_skips_furniture():
    title = _guess_title(_PAPER_WITH_DOI)
    assert title.startswith("Optogenetic inhibition")


# ---------------------------------------------------------------------------
# load_pdf_as_article — DOI resolution path
# ---------------------------------------------------------------------------


@patch("scirag.sources.pdf.pubmed")
@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_resolves_real_metadata_via_doi(mock_reader_cls, mock_pubmed, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_PAPER_WITH_DOI)
    real = Article(pmid="34592468", title="Real Title", abstract="", year="2021")
    mock_pubmed.search.return_value = ["34592468"]
    mock_pubmed.fetch.return_value = [real]

    pdf = tmp_path / "205920be.pdf"
    pdf.touch()
    article = load_pdf_as_article(pdf)

    mock_pubmed.search.assert_called_once_with("10.1038/s41467-026-70762-z[doi]", retmax=1)
    assert article.pmid == "34592468"  # real PMID, not the pdf: hash
    assert article.title == "Real Title"
    assert article.year == "2021"
    assert "clear effect" in article.full_text  # PDF Results grafted on


@patch("scirag.sources.pdf.pubmed")
@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_falls_back_when_doi_unresolved(mock_reader_cls, mock_pubmed, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_PAPER_WITH_DOI)
    mock_pubmed.search.return_value = []  # no PubMed match

    pdf = tmp_path / "205920be.pdf"
    pdf.touch()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        article = load_pdf_as_article(pdf)

    assert article.pmid.startswith("pdf:")  # synthetic, from filename
    assert article.doi == "10.1038/s41467-026-70762-z"  # still captured locally
    assert article.title.startswith("Optogenetic inhibition")
    assert any("no PubMed match" in str(w.message) for w in caught)


@patch("scirag.sources.pdf.pubmed")
@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_offline_falls_back(mock_reader_cls, mock_pubmed, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_PAPER_WITH_DOI)
    mock_pubmed.search.side_effect = RuntimeError("network down")

    pdf = tmp_path / "205920be.pdf"
    pdf.touch()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        article = load_pdf_as_article(pdf)

    assert article.pmid.startswith("pdf:")
    assert article.doi == "10.1038/s41467-026-70762-z"


# ---------------------------------------------------------------------------
# load_pdf_directory
# ---------------------------------------------------------------------------


@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_directory(mock_reader_cls, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_FULL_PAPER)
    for name in ("11111.pdf", "22222.pdf"):
        (tmp_path / name).touch()
    articles = load_pdf_directory(tmp_path)
    assert len(articles) == 2
    pmids = {a.pmid for a in articles}
    assert pmids == {"11111", "22222"}


def test_load_pdf_directory_empty(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        articles = load_pdf_directory(tmp_path)
    assert articles == []
    assert any("No PDF files" in str(w.message) for w in caught)


@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_directory_skips_broken_file(mock_reader_cls, tmp_path):
    (tmp_path / "good.pdf").touch()
    (tmp_path / "bad.pdf").touch()

    def side_effect(path):
        if "bad" in str(path):
            raise RuntimeError("corrupted")
        return _make_pdf_reader(_FULL_PAPER)

    mock_reader_cls.side_effect = side_effect
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        articles = load_pdf_directory(tmp_path)
    assert len(articles) == 1
    assert any("bad.pdf" in str(w.message) for w in caught)
