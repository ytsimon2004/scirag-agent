"""Tests for scirag.sources.pdf — Results-section isolation and PubMed resolution."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch


from scirag.sources.pdf import (
    _clean_pdf_title,
    _extract_doi,
    _guess_title,
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

_PAPER_WITH_DOI = """
Article https://doi.org/10.1038/s41467-026-70762-z
Optogenetic inhibition of the retrosplenial cortex disrupts memory retrieval

Results
We found a clear effect.
Discussion
It matters.
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


_PAPER_NUMBERED_SECTIONS = """
2. Methods
We implanted electrodes.

3. Results
We recorded 50 neurons in CA1.
Firing rates increased after training.

4. Discussion
This matters a lot.
"""


def test_extract_results_section_numbered_headings():
    text = extract_results_section(_PAPER_NUMBERED_SECTIONS)
    assert "50 neurons" in text
    assert "Firing rates increased" in text
    assert "implanted electrodes" not in text  # 2. Methods (before)
    assert "matters a lot" not in text  # 4. Discussion (after)


def test_extract_results_section_not_found():
    assert extract_results_section(_PAPER_NO_RESULTS) == ""


def test_extract_results_section_empty_string():
    assert extract_results_section("") == ""


# ---------------------------------------------------------------------------
# metadata extraction helpers
# ---------------------------------------------------------------------------


def test_extract_doi_found():
    assert _extract_doi(_PAPER_WITH_DOI) == "10.1038/s41467-026-70762-z"


def test_extract_doi_strips_trailing_punctuation():
    assert _extract_doi("see 10.1234/abc.def).") == "10.1234/abc.def"


def test_extract_doi_none():
    assert _extract_doi(_FULL_PAPER) == ""


def test_extract_doi_strips_elife_component():
    # eLife prints …/eLife.NNNNN.00N; PubMed indexes the bare article DOI.
    assert _extract_doi("doi: 10.7554/eLife.18372.001") == "10.7554/eLife.18372"


def test_clean_pdf_title_valid():
    assert _clean_pdf_title("Organization of feedback projections") == (
        "Organization of feedback projections"
    )


def test_clean_pdf_title_rejects_garbage():
    assert _clean_pdf_title("15209425899234 1..29") == ""  # typesetting artifact
    assert _clean_pdf_title("short") == ""  # too short
    assert _clean_pdf_title(None) == ""  # missing


def test_guess_title_skips_furniture():
    assert _guess_title(_PAPER_WITH_DOI).startswith("Optogenetic inhibition")


def test_guess_title_skips_copyright_and_email():
    text = (
        "© The Author(s) 2020. Published by Oxford University Press.\n"
        "*For correspondence:jackw@alleninstitute.org\n"
        "Stable Encoding of Visual Cues in the Mouse Retrosplenial Cortex\n"
    )
    assert _guess_title(text).startswith("Stable Encoding of Visual Cues")


# ---------------------------------------------------------------------------
# load_pdf_as_article — resolve-or-skip (pypdf + pubmed mocked)
# ---------------------------------------------------------------------------


def _make_pdf_reader(text: str, title=None, author=None) -> MagicMock:
    page = MagicMock()
    page.extract_text.return_value = text
    reader = MagicMock()
    reader.pages = [page]
    meta = MagicMock()
    meta.title = title
    meta.author = author
    reader.metadata = meta
    return reader


@patch("scirag.sources.pdf.pubmed")
@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_numeric_filename_resolves_via_pmid(mock_reader_cls, mock_pubmed, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_FULL_PAPER)
    mock_pubmed.fetch.return_value = [Article(pmid="12345678", title="Place cells", abstract="A")]

    pdf = tmp_path / "12345678.pdf"
    pdf.touch()
    article = load_pdf_as_article(pdf)

    mock_pubmed.fetch.assert_called_once_with(["12345678"])
    assert article.pmid == "12345678"
    assert "120 place cells" in article.full_text  # PDF Results grafted on


@patch("scirag.sources.pdf.pubmed")
@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_resolves_via_doi(mock_reader_cls, mock_pubmed, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_PAPER_WITH_DOI)
    real = Article(pmid="34592468", title="Real Title", abstract="", year="2021")
    mock_pubmed.search.return_value = ["34592468"]
    mock_pubmed.fetch.return_value = [real]

    pdf = tmp_path / "paper.pdf"
    pdf.touch()
    article = load_pdf_as_article(pdf)

    mock_pubmed.search.assert_called_once_with("10.1038/s41467-026-70762-z[doi]", retmax=1)
    assert article.pmid == "34592468"
    assert article.title == "Real Title"
    assert "clear effect" in article.full_text  # PDF Results grafted on


@patch("scirag.sources.pdf.pubmed")
@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_resolves_via_title_when_doi_unresolved(mock_reader_cls, mock_pubmed, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_PAPER_WITH_DOI)
    real = Article(
        pmid="34113813",
        title="Optogenetic inhibition of the retrosplenial cortex disrupts memory retrieval",
        abstract="",
    )
    # DOI lookup misses; title search hits.
    mock_pubmed.search.side_effect = lambda q, retmax=1: [] if q.endswith("[doi]") else ["34113813"]
    mock_pubmed.fetch.return_value = [real]

    pdf = tmp_path / "paper.pdf"
    pdf.touch()
    article = load_pdf_as_article(pdf)

    assert article.pmid == "34113813"
    assert "clear effect" in article.full_text


@patch("scirag.sources.pdf.pubmed")
@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_uses_embedded_title_for_resolution(mock_reader_cls, mock_pubmed, tmp_path):
    # No DOI in text -> title search uses the PDF's embedded /Title.
    title = "Organization of feedback projections to mouse primary visual cortex"
    mock_reader_cls.return_value = _make_pdf_reader(_PAPER_NO_RESULTS, title=title)
    mock_pubmed.search.return_value = ["34113813"]
    mock_pubmed.fetch.return_value = [Article(pmid="34113813", title=title, abstract="")]

    pdf = tmp_path / "morimoto.pdf"
    pdf.touch()
    article = load_pdf_as_article(pdf)

    mock_pubmed.search.assert_called_once_with(f"{title}[title]", retmax=1)
    assert article.pmid == "34113813"


@patch("scirag.sources.pdf.pubmed")
@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_resolved_without_results_has_empty_fulltext(
    mock_reader_cls, mock_pubmed, tmp_path
):
    mock_reader_cls.return_value = _make_pdf_reader(_PAPER_NO_RESULTS)
    mock_pubmed.fetch.return_value = [Article(pmid="999", title="T", abstract="From PubMed")]

    pdf = tmp_path / "999.pdf"
    pdf.touch()
    article = load_pdf_as_article(pdf)

    assert article.full_text == ""  # no Results section in the PDF
    assert article.abstract == "From PubMed"  # but the record still has its abstract


_REVIEW_PAPER = """
Rethinking retrosplenial cortex

Introduction
The retrosplenial cortex is interesting.
We review its many roles.

Conclusion
It does a lot.

References
1. Smith 2020.
"""


@patch("scirag.sources.pdf.pubmed")
@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_review_uses_full_body(mock_reader_cls, mock_pubmed, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_REVIEW_PAPER)
    review = Article(pmid="36460006", title="Rethinking RSC", abstract="", pub_types=["Review"])
    mock_pubmed.fetch.return_value = [review]

    pdf = tmp_path / "36460006.pdf"  # numeric -> resolve via PMID
    pdf.touch()
    article = load_pdf_as_article(pdf)

    assert article.full_text_kind == "review"
    assert article.metadata()["text_source"] == "review"
    assert "review its many roles" in article.full_text  # whole body indexed
    assert "Smith 2020" not in article.full_text  # references trimmed


@patch("scirag.sources.pdf.pubmed")
@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_skips_and_warns_when_unresolved(mock_reader_cls, mock_pubmed, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_PAPER_WITH_DOI)
    mock_pubmed.search.return_value = []  # DOI and title both miss

    pdf = tmp_path / "paper.pdf"
    pdf.touch()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        article = load_pdf_as_article(pdf)

    assert article is None  # NOT imported
    msgs = " ".join(str(w.message) for w in caught)
    assert "NOT imported" in msgs
    assert "pubmed.ncbi.nlm.nih.gov" in msgs  # manual-lookup URL


@patch("scirag.sources.pdf.pubmed")
@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_skips_when_offline(mock_reader_cls, mock_pubmed, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_PAPER_WITH_DOI)
    mock_pubmed.search.side_effect = RuntimeError("network down")

    pdf = tmp_path / "paper.pdf"
    pdf.touch()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        article = load_pdf_as_article(pdf)

    assert article is None


# ---------------------------------------------------------------------------
# load_pdf_directory
# ---------------------------------------------------------------------------


@patch("scirag.sources.pdf.pubmed")
@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_directory(mock_reader_cls, mock_pubmed, tmp_path):
    mock_reader_cls.return_value = _make_pdf_reader(_FULL_PAPER)
    mock_pubmed.fetch.side_effect = lambda pmids: [Article(pmid=pmids[0], title="T", abstract="A")]

    for name in ("11111.pdf", "22222.pdf"):
        (tmp_path / name).touch()
    articles = load_pdf_directory(tmp_path)

    assert {a.pmid for a in articles} == {"11111", "22222"}


@patch("scirag.sources.pdf.pubmed")
@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_directory_skips_unresolved(mock_reader_cls, mock_pubmed, tmp_path):
    # 111.pdf resolves via PMID; junk.pdf (no DOI, title misses) is skipped.
    def reader_for(path):
        return _make_pdf_reader(_FULL_PAPER if "111" in str(path) else _PAPER_NO_RESULTS)

    mock_reader_cls.side_effect = reader_for
    mock_pubmed.fetch.side_effect = lambda pmids: [Article(pmid=pmids[0], title="T", abstract="A")]
    mock_pubmed.search.return_value = []  # title search misses for junk.pdf

    (tmp_path / "111.pdf").touch()
    (tmp_path / "junk.pdf").touch()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        articles = load_pdf_directory(tmp_path)

    assert [a.pmid for a in articles] == ["111"]


def test_load_pdf_directory_empty(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        articles = load_pdf_directory(tmp_path)
    assert articles == []
    assert any("No PDF files" in str(w.message) for w in caught)


@patch("scirag.sources.pdf.pubmed")
@patch("scirag.sources.pdf.pypdf.PdfReader")
def test_load_pdf_directory_skips_broken_file(mock_reader_cls, mock_pubmed, tmp_path):
    (tmp_path / "111.pdf").touch()
    (tmp_path / "bad.pdf").touch()

    def side_effect(path):
        if "bad" in str(path):
            raise RuntimeError("corrupted")
        return _make_pdf_reader(_FULL_PAPER)

    mock_reader_cls.side_effect = side_effect
    mock_pubmed.fetch.side_effect = lambda pmids: [Article(pmid=pmids[0], title="T", abstract="A")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        articles = load_pdf_directory(tmp_path)
    assert len(articles) == 1
    assert any("bad.pdf" in str(w.message) for w in caught)
