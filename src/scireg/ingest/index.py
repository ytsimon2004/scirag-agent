"""Indexing layer (LlamaIndex). Parses + chunks Articles, embeds with a local
Ollama model, and persists to an embedded LanceDB store.

All DB paths route through scireg.projects.get_active_db_uri() so that
multiple research projects can each have their own isolated index.
"""
from __future__ import annotations

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from scireg.config import models_cfg, pipeline_cfg
from scireg.sources.pubmed import Article


def _embed_model() -> OllamaEmbedding:
    emb = models_cfg()["embeddings"]
    return OllamaEmbedding(model_name=emb["model"], base_url=emb["api_base"])


def _vector_store() -> LanceDBVectorStore:
    from scireg.projects import get_active_db_uri
    return LanceDBVectorStore(
        uri=get_active_db_uri(),
        table_name=pipeline_cfg()["index"]["table"],
    )


def build_index(articles: list[Article]) -> VectorStoreIndex:
    """Chunk + embed + persist a batch of articles. Idempotent-append."""
    idx = pipeline_cfg()["index"]
    docs = [
        Document(text=a.to_text(), metadata=a.metadata(), doc_id=a.pmid)
        for a in articles
    ]
    splitter = SentenceSplitter(
        chunk_size=idx["chunk_size"], chunk_overlap=idx["chunk_overlap"]
    )
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
    import lancedb
    from scireg.projects import get_active_db_uri
    try:
        db = lancedb.connect(get_active_db_uri())
        tbl = db.open_table(pipeline_cfg()["index"]["table"])
        arrow_tbl = tbl.to_lance().to_table(columns=["metadata"])
        metadata_col = arrow_tbl.column("metadata").combine_chunks()
        pmids = metadata_col.field("pmid").to_pylist()
        return {p for p in pmids if p}
    except Exception:
        return set()


def load_index() -> VectorStoreIndex:
    """Open the active project's LanceDB index for querying."""
    return VectorStoreIndex.from_vector_store(
        _vector_store(), embed_model=_embed_model()
    )
