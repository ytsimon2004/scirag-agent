"""Tests for scirag.sources.pdf — PDF text extraction and Results-section isolation."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch


from scirag.sources.pdf import (
    _pmid_from_stem,
    extract_results_section,
    load_pdf_as_article,
    load_pdf_directory,
)


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
