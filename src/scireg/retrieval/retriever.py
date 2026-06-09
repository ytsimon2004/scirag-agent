"""Hybrid retrieval: dense (vector) + sparse (BM25) with reciprocal-rank
fusion. BM25 is built over whatever nodes the vector store returns plus the
docstore, so it stays in-process (no separate search server for the first cut).
"""
from __future__ import annotations

from llama_index.core.schema import NodeWithScore

from scireg.config import pipeline_cfg
from scireg.ingest.index import load_index


def _rrf(rankings: list[list[NodeWithScore]], k: int = 60) -> list[NodeWithScore]:
    """Reciprocal-rank fusion across multiple ranked node lists."""
    scores: dict[str, float] = {}
    nodes: dict[str, NodeWithScore] = {}
    for ranking in rankings:
        for rank, nws in enumerate(ranking):
            nid = nws.node.node_id
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
            nodes[nid] = nws
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [nodes[nid] for nid in ranked]


def retrieve(query: str) -> list[NodeWithScore]:
    cfg = pipeline_cfg()["retrieval"]
    index = load_index()

    dense = index.as_retriever(similarity_top_k=cfg["top_k"]).retrieve(query)

    if not cfg.get("hybrid"):
        return dense[: cfg["final_k"]]

    # Sparse pass over the same docstore.
    try:
        from llama_index.retrievers.bm25 import BM25Retriever

        bm25 = BM25Retriever.from_defaults(
            docstore=index.docstore, similarity_top_k=cfg["bm25_k"]
        )
        sparse = bm25.retrieve(query)
    except Exception:
        sparse = []  # BM25 optional; degrade to dense-only gracefully

    fused = _rrf([dense, sparse]) if sparse else dense
    return fused[: cfg["final_k"]]
