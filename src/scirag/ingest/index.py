"""Indexing layer (LlamaIndex). Parses + chunks Articles, embeds with a local
Ollama model, and persists to an embedded LanceDB store.

All DB paths route through scirag.projects.get_active_db_uri() so that
multiple research projects can each have their own isolated index.
"""

from __future__ import annotations

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from scirag.config import models_cfg, pipeline_cfg
from scirag.sources.pubmed import Article


def _embed_model() -> OllamaEmbedding:
    emb = models_cfg()["embeddings"]
    return OllamaEmbedding(model_name=emb["model"], base_url=emb["api_base"])


def _vector_store() -> LanceDBVectorStore:
    from scirag.projects import get_active_db_uri

    return LanceDBVectorStore(
        uri=get_active_db_uri(),
        table_name=pipeline_cfg()["index"]["table"],
    )


def build_index(articles: list[Article]) -> VectorStoreIndex:
    """Chunk + embed + persist a batch of articles. Idempotent-append."""
    idx = pipeline_cfg()["index"]
    docs = [Document(text=a.to_text(), metadata=a.metadata(), doc_id=a.pmid) for a in articles]
    splitter = SentenceSplitter(chunk_size=idx["chunk_size"], chunk_overlap=idx["chunk_overlap"])
    storage = StorageContext.from_defaults(vector_store=_vector_store())
    return VectorStoreIndex.from_documents(
        docs,
        storage_context=storage,
        embed_model=_embed_model(),
        transformations=[splitter],
        show_progress=True,
    )


def get_indexed_pmids() -> set[str]:
    """Return PMIDs in the active project's index. Empty set if index doesn't exist."""
    return {a["pmid"] for a in get_indexed_articles()}


def get_indexed_articles() -> list[dict]:
    """Return deduplicated article metadata (pmid, title, year, first_author, text_source).

    `text_source` is "results" when the chunk came from full-text Results, else
    "abstract". `first_author` may be "" for articles indexed before authors were
    stored, or for PDF imports without resolved metadata.
    """
    import lancedb
    from scirag.projects import get_active_db_uri

    try:
        db = lancedb.connect(get_active_db_uri())
        tbl = db.open_table(pipeline_cfg()["index"]["table"])
        arrow_tbl = tbl.to_lance().to_table(columns=["metadata"])
        metadata_col = arrow_tbl.column("metadata").combine_chunks()
        fields = {f.name for f in metadata_col.type}

        def _col(name: str) -> list:
            if name in fields:
                return metadata_col.field(name).to_pylist()
            return [""] * len(metadata_col)

        seen: set[str] = set()
        articles: list[dict] = []
        for pmid, title, year, first_author, source in zip(
            metadata_col.field("pmid").to_pylist(),
            metadata_col.field("title").to_pylist(),
            metadata_col.field("year").to_pylist(),
            _col("first_author"),
            _col("text_source"),
        ):
            if pmid and pmid not in seen:
                seen.add(pmid)
                articles.append(
                    {
                        "pmid": pmid,
                        "title": title or "",
                        "year": year or "",
                        "first_author": first_author or "",
                        "text_source": source or "",
                    }
                )
        return articles
    except Exception:
        return []


def remove_articles(pmids: list[str]) -> int:
    """Delete all chunks for the given PMIDs. Returns count of PMIDs removed."""
    import lancedb
    from scirag.projects import get_active_db_uri

    if not pmids:
        return 0
    db = lancedb.connect(get_active_db_uri())
    tbl = db.open_table(pipeline_cfg()["index"]["table"])
    ids = ", ".join(f"'{p}'" for p in pmids)
    tbl.delete(f"metadata.pmid IN ({ids})")
    return len(pmids)


def load_index() -> VectorStoreIndex:
    """Open the active project's LanceDB index for querying."""
    return VectorStoreIndex.from_vector_store(_vector_store(), embed_model=_embed_model())
