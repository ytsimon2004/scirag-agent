"""Tests for scireg.sources.pubmed — all network calls are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from xml.etree import ElementTree as ET

import pytest

from scireg.sources.pubmed import (
    Article,
    _parse_article,
    fetch,
    search,
    search_and_fetch,
)


# ---------------------------------------------------------------------------
# Article dataclass helpers
# ---------------------------------------------------------------------------

def test_article_url():
    a = Article(pmid="12345", title="T", abstract="A")
    assert a.url == "https://pubmed.ncbi.nlm.nih.gov/12345/"


def test_article_to_text():
    a = Article(pmid="1", title="Title", abstract="Abstract text")
    assert a.to_text() == "Title\n\nAbstract text"


def test_article_metadata_keys():
    a = Article(pmid="1", title="T", abstract="A", journal="J", year="2024",
                mesh_terms=["memory", "hippocampus"])
    md = a.metadata()
    assert set(md) == {"pmid", "title", "journal", "year", "url", "mesh"}
    assert md["mesh"] == "memory, hippocampus"


def test_article_metadata_empty_mesh():
    a = Article(pmid="1", title="T", abstract="A")
    assert a.metadata()["mesh"] == ""


# ---------------------------------------------------------------------------
# XML parsing
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
    </Article>
    <MeshHeadingList>
      <MeshHeading><DescriptorName>Hippocampus</DescriptorName></MeshHeading>
      <MeshHeading><DescriptorName>Spatial Navigation</DescriptorName></MeshHeading>
    </MeshHeadingList>
  </MedlineCitation>
  <PubmedData>
    <History>
      <PubMedPubDate PubStatus="pubmed">
        <Year>2005</Year>
      </PubMedPubDate>
    </History>
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


def test_parse_article_missing_fields():
    """Minimal XML should not raise; missing fields get defaults."""
    xml = "<PubmedArticle><MedlineCitation><PMID>1</PMID></MedlineCitation></PubmedArticle>"
    root = ET.fromstring(xml)
    art = _parse_article(root)
    assert art.pmid == "1"
    assert art.title == ""
    assert art.abstract == ""
    assert art.authors == []


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


@patch("scireg.sources.pubmed._get")
def test_search_returns_pmids(mock_get):
    mock_get.return_value = _mock_response(_ESEARCH_XML)
    pmids = search("hippocampal place cells")
    assert pmids == ["11111", "22222"]
    mock_get.assert_called_once()
    call_path = mock_get.call_args[0][0]
    assert call_path == "esearch.fcgi"


@patch("scireg.sources.pubmed._get")
def test_fetch_returns_articles(mock_get):
    mock_get.return_value = _mock_response(_EFETCH_XML)
    articles = fetch(["99999"])
    assert len(articles) == 1
    assert articles[0].pmid == "99999"


@patch("scireg.sources.pubmed._get")
def test_fetch_empty_list(mock_get):
    articles = fetch([])
    assert articles == []
    mock_get.assert_not_called()


@patch("scireg.sources.pubmed._get")
def test_search_and_fetch(mock_get):
    mock_get.side_effect = [
        _mock_response(_ESEARCH_XML),
        _mock_response(_EFETCH_XML),
    ]
    articles = search_and_fetch("grid cells entorhinal cortex", retmax=5)
    assert isinstance(articles, list)
    assert mock_get.call_count == 2
