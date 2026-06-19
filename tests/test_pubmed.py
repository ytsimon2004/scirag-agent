"""Tests for scirag.sources.pubmed — all network calls are mocked."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree as ET


from scirag.sources.pubmed import (
    Article,
    _extract_results_from_jats,
    _fetch_pmc_fulltext,
    _parse_article,
    _pmids_to_pmcids,
    _unpaywall_pdf_url,
    enrich_with_fulltext,
    fetch,
    search,
    search_and_fetch,
    search_semantic,
)


# ---------------------------------------------------------------------------
# Article dataclass helpers
# ---------------------------------------------------------------------------


def test_article_url():
    a = Article(pmid="12345", title="T", abstract="A")
    assert a.url == "https://pubmed.ncbi.nlm.nih.gov/12345/"


def test_article_to_text_abstract_only():
    a = Article(pmid="1", title="Title", abstract="Abstract text")
    assert a.to_text() == "Title\n\nAbstract text"


def test_article_to_text_prefers_full_text():
    a = Article(pmid="1", title="Title", abstract="Abstract text", full_text="Full body text.")
    assert a.to_text() == "Title\n\nFull body text."


def test_article_to_text_falls_back_when_full_text_empty():
    a = Article(pmid="1", title="Title", abstract="Abstract text", full_text="")
    assert a.to_text() == "Title\n\nAbstract text"


def test_article_to_text_includes_authors():
    a = Article(pmid="1", title="Title", abstract="Body", authors=["Powell K", "Smith J"])
    assert a.to_text() == "Title\n\nAuthors: Powell K, Smith J\n\nBody"


def test_article_metadata_keys():
    a = Article(
        pmid="1",
        title="T",
        abstract="A",
        journal="J",
        year="2024",
        doi="10.1038/s41586-000",
        mesh_terms=["memory", "hippocampus"],
        authors=["Powell K", "Smith J"],
    )
    md = a.metadata()
    assert set(md) == {
        "pmid",
        "title",
        "journal",
        "year",
        "url",
        "doi",
        "mesh",
        "authors",
        "first_author",
        "text_source",
        "source",
    }
    assert md["mesh"] == "memory, hippocampus"
    assert md["doi"] == "10.1038/s41586-000"
    assert md["authors"] == "Powell K, Smith J"
    assert md["first_author"] == "Powell K"


def test_article_metadata_no_authors():
    a = Article(pmid="1", title="T", abstract="A")
    md = a.metadata()
    assert md["authors"] == ""
    assert md["first_author"] == ""


def test_article_is_review():
    assert Article(
        pmid="1", title="T", abstract="A", pub_types=["Journal Article", "Review"]
    ).is_review
    assert not Article(pmid="1", title="T", abstract="A", pub_types=["Journal Article"]).is_review


def test_article_text_source_label():
    # Results section -> "results"; review whole-body -> "review"; neither -> "abstract".
    assert (
        Article(pmid="1", title="T", abstract="A", full_text="R").metadata()["text_source"]
        == "results"
    )
    assert (
        Article(
            pmid="1", title="T", abstract="A", full_text="B", full_text_kind="review"
        ).metadata()["text_source"]
        == "review"
    )
    assert Article(pmid="1", title="T", abstract="A").metadata()["text_source"] == "abstract"


def test_article_metadata_empty_mesh():
    a = Article(pmid="1", title="T", abstract="A")
    assert a.metadata()["mesh"] == ""


# ---------------------------------------------------------------------------
# XML parsing — including DOI
# ---------------------------------------------------------------------------

_ARTICLE_XML = """
<PubmedArticle>
  <MedlineCitation>
    <PMID>99999</PMID>
    <Article>
      <ArticleTitle>Place cells and grid cells</ArticleTitle>
      <Abstract>
        <AbstractText Label="BACKGROUND">Background text.</AbstractText>
        <AbstractText Label="RESULTS">Results text.</AbstractText>
      </Abstract>
      <AuthorList>
        <Author><LastName>O'Keefe</LastName><Initials>J</Initials></Author>
        <Author><LastName>Moser</LastName><Initials>EI</Initials></Author>
      </AuthorList>
      <Journal><Title>Nature Neuroscience</Title></Journal>
      <PublicationTypeList>
        <PublicationType>Journal Article</PublicationType>
        <PublicationType>Review</PublicationType>
      </PublicationTypeList>
    </Article>
    <MeshHeadingList>
      <MeshHeading><DescriptorName>Hippocampus</DescriptorName></MeshHeading>
      <MeshHeading><DescriptorName>Spatial Navigation</DescriptorName></MeshHeading>
    </MeshHeadingList>
  </MedlineCitation>
  <PubmedData>
    <ArticleIdList>
      <ArticleId IdType="doi">10.1038/nn.9999</ArticleId>
      <ArticleId IdType="pubmed">99999</ArticleId>
    </ArticleIdList>
  </PubmedData>
</PubmedArticle>
"""


def test_parse_article_fields():
    root = ET.fromstring(_ARTICLE_XML)
    art = _parse_article(root)
    assert art.pmid == "99999"
    assert art.title == "Place cells and grid cells"
    assert "BACKGROUND: Background text." in art.abstract
    assert "RESULTS: Results text." in art.abstract
    assert "O'Keefe J" in art.authors
    assert "Moser EI" in art.authors
    assert art.journal == "Nature Neuroscience"
    assert "Hippocampus" in art.mesh_terms
    assert "Spatial Navigation" in art.mesh_terms
    assert art.doi == "10.1038/nn.9999"
    assert "Review" in art.pub_types
    assert art.is_review


def test_parse_article_missing_fields():
    """Minimal XML should not raise; missing fields get defaults."""
    xml = "<PubmedArticle><MedlineCitation><PMID>1</PMID></MedlineCitation></PubmedArticle>"
    root = ET.fromstring(xml)
    art = _parse_article(root)
    assert art.pmid == "1"
    assert art.title == ""
    assert art.abstract == ""
    assert art.authors == []
    assert art.doi == ""


def test_parse_article_no_doi():
    xml = """
    <PubmedArticle>
      <MedlineCitation><PMID>2</PMID></MedlineCitation>
      <PubmedData>
        <ArticleIdList>
          <ArticleId IdType="pubmed">2</ArticleId>
        </ArticleIdList>
      </PubmedData>
    </PubmedArticle>"""
    art = _parse_article(ET.fromstring(xml))
    assert art.doi == ""


# ---------------------------------------------------------------------------
# search() / fetch() — mock _get
# ---------------------------------------------------------------------------

_ESEARCH_XML = """
<eSearchResult>
  <IdList>
    <Id>11111</Id>
    <Id>22222</Id>
  </IdList>
</eSearchResult>
"""

_EFETCH_XML = f"""
<PubmedArticleSet>
  {_ARTICLE_XML}
</PubmedArticleSet>
"""


def _mock_response(text: str) -> MagicMock:
    r = MagicMock()
    r.text = text
    r.raise_for_status = MagicMock()
    return r


@patch("scirag.sources.pubmed._get")
def test_search_returns_pmids(mock_get):
    mock_get.return_value = _mock_response(_ESEARCH_XML)
    pmids = search("hippocampal place cells")
    assert pmids == ["11111", "22222"]
    mock_get.assert_called_once()
    call_path = mock_get.call_args[0][0]
    assert call_path == "esearch.fcgi"


@patch("scirag.sources.pubmed._get")
def test_fetch_returns_articles(mock_get):
    mock_get.return_value = _mock_response(_EFETCH_XML)
    articles = fetch(["99999"])
    assert len(articles) == 1
    assert articles[0].pmid == "99999"


@patch("scirag.sources.pubmed._get")
def test_fetch_empty_list(mock_get):
    articles = fetch([])
    assert articles == []
    mock_get.assert_not_called()


@patch("scirag.sources.pubmed._get")
def test_search_and_fetch(mock_get):
    mock_get.side_effect = [
        _mock_response(_ESEARCH_XML),
        _mock_response(_EFETCH_XML),
    ]
    articles = search_and_fetch("grid cells entorhinal cortex", retmax=5)
    assert isinstance(articles, list)
    assert mock_get.call_count == 2


def _mock_json_response(payload: dict) -> MagicMock:
    r = MagicMock()
    r.json = MagicMock(return_value=payload)
    r.raise_for_status = MagicMock()
    return r


@patch("scirag.sources.pubmed.httpx.get")
def test_search_semantic_returns_pmids(mock_get):
    mock_get.return_value = _mock_json_response(
        {"resultList": {"result": [{"pmid": "11111"}, {"pmid": "22222"}, {"pmid": "11111"}]}}
    )
    pmids = search_semantic("retrosplenial cortex related disorder in human", retmax=5)
    assert pmids == ["11111", "22222"]  # deduped, relevance order preserved
    url = mock_get.call_args[0][0]
    query = mock_get.call_args.kwargs["params"]["query"]
    assert "europepmc" in url
    assert "SRC:MED" in query


@patch("scirag.sources.pubmed.httpx.get")
def test_search_semantic_year_filter(mock_get):
    mock_get.return_value = _mock_json_response({"resultList": {"result": []}})
    search_semantic("place cells", min_year="2018", max_year="2024")
    query = mock_get.call_args.kwargs["params"]["query"]
    assert "FIRST_PDATE:[2018-01-01 TO 2024-12-31]" in query


@patch("scirag.sources.pubmed.httpx.get")
def test_search_semantic_skips_records_without_pmid(mock_get):
    mock_get.return_value = _mock_json_response(
        {"resultList": {"result": [{"id": "PPR1"}, {"pmid": "33333"}]}}
    )
    assert search_semantic("query") == ["33333"]


# ---------------------------------------------------------------------------
# JATS Results-section extraction
# ---------------------------------------------------------------------------

_JATS_XML = """
<article>
  <body>
    <sec>
      <title>Introduction</title>
      <p>Place cells fire when the animal occupies a specific location.</p>
      <p>Grid cells form a hexagonal pattern.</p>
    </sec>
    <sec>
      <title>Results</title>
      <p>We recorded from CA1 neurons during navigation.</p>
      <p>Place fields were stable across sessions.</p>
    </sec>
    <sec>
      <title>Discussion</title>
      <p>These findings suggest a remapping mechanism.</p>
    </sec>
  </body>
</article>
"""

_JATS_NO_RESULTS_XML = """
<article>
  <body>
    <sec><title>Introduction</title><p>Background.</p></sec>
    <sec><title>Methods</title><p>We used patch clamp.</p></sec>
  </body>
</article>
"""


def test_extract_results_from_jats_returns_results_only():
    root = ET.fromstring(_JATS_XML)
    text = _extract_results_from_jats(root)
    assert "CA1 neurons" in text
    assert "Place fields" in text
    # Introduction and Discussion should be excluded
    assert "Place cells fire" not in text
    assert "remapping mechanism" not in text


def test_extract_results_from_jats_no_results_section():
    root = ET.fromstring(_JATS_NO_RESULTS_XML)
    assert _extract_results_from_jats(root) == ""


@patch("scirag.sources.pubmed._get")
def test_fetch_pmc_fulltext_returns_results_only(mock_get):
    mock_get.return_value = _mock_response(_JATS_XML)
    text = _fetch_pmc_fulltext("7654321")
    assert "CA1 neurons" in text
    assert "Place cells fire" not in text  # Introduction excluded


@patch("scirag.sources.pubmed._get")
def test_fetch_pmc_fulltext_returns_empty_on_error(mock_get):
    mock_get.side_effect = Exception("network error")
    assert _fetch_pmc_fulltext("bad_id") == ""


# ---------------------------------------------------------------------------
# PMC elink
# ---------------------------------------------------------------------------

_ELINK_XML = """
<eLinkResult>
  <LinkSet>
    <IdList><Id>99999</Id></IdList>
    <LinkSetDb>
      <DbTo>pmc</DbTo>
      <LinkName>pubmed_pmc</LinkName>
      <Link><Id>7654321</Id></Link>
    </LinkSetDb>
  </LinkSet>
</eLinkResult>
"""

_ELINK_NO_RESULTS_XML = """
<eLinkResult>
  <LinkSet>
    <IdList><Id>99999</Id></IdList>
  </LinkSet>
</eLinkResult>
"""


@patch("scirag.sources.pubmed._get")
def test_pmids_to_pmcids_maps_correctly(mock_get):
    mock_get.return_value = _mock_response(_ELINK_XML)
    result = _pmids_to_pmcids(["99999"])
    assert result == {"99999": "7654321"}
    assert mock_get.call_args[0][0] == "elink.fcgi"


@patch("scirag.sources.pubmed._get")
def test_pmids_to_pmcids_no_pmc_entry(mock_get):
    mock_get.return_value = _mock_response(_ELINK_NO_RESULTS_XML)
    result = _pmids_to_pmcids(["99999"])
    assert result == {}


# ---------------------------------------------------------------------------
# Unpaywall
# ---------------------------------------------------------------------------

_UNPAYWALL_JSON_WITH_PDF = '{"best_oa_location": {"url_for_pdf": "https://example.com/paper.pdf", "url": "https://example.com/paper"}}'
_UNPAYWALL_JSON_NO_OA = '{"best_oa_location": null}'


def _mock_httpx_response(text: str, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.text = text
    r.json.return_value = __import__("json").loads(text)
    r.status_code = status
    r.raise_for_status = MagicMock()
    return r


@patch("scirag.sources.pubmed.httpx.get")
def test_unpaywall_pdf_url_found(mock_get):
    mock_get.return_value = _mock_httpx_response(_UNPAYWALL_JSON_WITH_PDF)
    url = _unpaywall_pdf_url("10.1038/nn.9999")
    assert url == "https://example.com/paper.pdf"


@patch("scirag.sources.pubmed.httpx.get")
def test_unpaywall_pdf_url_no_oa(mock_get):
    mock_get.return_value = _mock_httpx_response(_UNPAYWALL_JSON_NO_OA)
    assert _unpaywall_pdf_url("10.1038/nn.9999") == ""


def test_unpaywall_pdf_url_empty_doi():
    assert _unpaywall_pdf_url("") == ""


@patch("scirag.sources.pubmed.httpx.get")
def test_unpaywall_returns_empty_on_error(mock_get):
    mock_get.side_effect = Exception("network error")
    assert _unpaywall_pdf_url("10.1038/nn.9999") == ""


@patch("scirag.sources.pubmed._download_pdf_results", return_value="Results from PDF.")
@patch("scirag.sources.pubmed._unpaywall_pdf_url", return_value="https://example.com/paper.pdf")
@patch("scirag.sources.pubmed._pmids_to_pmcids", return_value={})
def test_enrich_falls_back_to_unpaywall(mock_pmc, mock_unp, mock_dl):
    article = Article(pmid="99999", title="T", abstract="A", doi="10.1038/nn.9999")
    enrich_with_fulltext([article])
    assert article.full_text == "Results from PDF."


# ---------------------------------------------------------------------------
# enrich_with_fulltext — PMC path + warning
# ---------------------------------------------------------------------------


@patch("scirag.sources.pubmed._fetch_pmc_fulltext", return_value="Results section text.")
@patch("scirag.sources.pubmed._pmids_to_pmcids", return_value={"99999": "7654321"})
def test_enrich_with_fulltext_sets_fields(mock_pmcids, mock_fulltext):
    article = Article(pmid="99999", title="T", abstract="A")
    enrich_with_fulltext([article])
    assert article.pmc_id == "7654321"
    assert article.full_text == "Results section text."


@patch("scirag.sources.pubmed._unpaywall_pdf_url", return_value="")
@patch("scirag.sources.pubmed._pmids_to_pmcids", return_value={})
def test_enrich_warns_when_no_full_text(mock_pmcids, mock_unp):
    article = Article(pmid="99999", title="T", abstract="A")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        enrich_with_fulltext([article])
    assert any("no retrievable full text" in str(w.message) for w in caught)
    assert any("scirag import" in str(w.message) for w in caught)


def test_enrich_with_fulltext_empty_list():
    enrich_with_fulltext([])  # must not raise


# ---------------------------------------------------------------------------
# Full pipeline: search → fetch → enrich
# ---------------------------------------------------------------------------


@patch("scirag.sources.pubmed._download_pdf_results", return_value="")
@patch("scirag.sources.pubmed._unpaywall_pdf_url", return_value="")
@patch("scirag.sources.pubmed._get")
def test_full_pipeline_search_fetch_enrich(mock_get, mock_unp, mock_dl):
    mock_get.side_effect = [
        _mock_response(_ESEARCH_XML),  # search
        _mock_response(_EFETCH_XML),  # fetch
        _mock_response(_ELINK_XML),  # pmids_to_pmcids
        _mock_response(_JATS_XML),  # fetch_pmc_fulltext
    ]
    articles = search_and_fetch("place cells", retmax=2)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        enrich_with_fulltext(articles)
    assert articles[0].pmc_id == "7654321"
    assert "CA1 neurons" in articles[0].full_text
