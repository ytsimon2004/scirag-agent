"""Hybrid retrieval: dense (vector) + sparse (BM25) with reciprocal-rank
fusion, then an optional cross-encoder rerank.

Flow: pull `top_k` dense + `bm25_k` sparse candidates, fuse them with RRF, then
(when `rerank` is on) re-score the fused pool with a cross-encoder and keep the
best `final_k`. Retrieving wide and reranking raises both recall and precision
versus trimming the RRF order straight to `final_k`.

BM25 is built over the docstore so it stays in-process (no search server).
"""

from __future__ import annotations

import sys

from llama_index.core.schema import NodeWithScore

from scirag.config import get_retrieval
from scirag.ingest.index import load_index

_DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
_reranker_cache: dict[str, object] = {}
_rerank_warned = False


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


def _get_reranker(model_name: str):
    """Lazily load + cache a sentence-transformers CrossEncoder."""
    if model_name not in _reranker_cache:
        from sentence_transformers import CrossEncoder

        _reranker_cache[model_name] = CrossEncoder(model_name)
    return _reranker_cache[model_name]


def _rerank(query: str, nodes: list[NodeWithScore], model_name: str) -> list[NodeWithScore] | None:
    """Reorder `nodes` by cross-encoder relevance to `query`. Returns the same
    NodeWithScore objects in a new order (their `.score` is left untouched, so the
    cosine-based grounding gate keeps working). Returns None — caller keeps the
    fused order — if the reranker can't be used."""
    global _rerank_warned
    if not nodes:
        return nodes
    try:
        reranker = _get_reranker(model_name)
    except Exception as exc:  # missing extra, model download/load failure, etc.
        if not _rerank_warned:
            _rerank_warned = True
            print(
                f"[scirag] reranking disabled: {exc}. "
                "Install with `uv sync --extra rerank` to enable cross-encoder reranking.",
                file=sys.stderr,
            )
        return None
    scores = reranker.predict([(query, n.node.get_content()) for n in nodes])
    order = sorted(range(len(nodes)), key=lambda i: scores[i], reverse=True)
    return [nodes[i] for i in order]


def retrieve(query: str) -> list[NodeWithScore]:
    cfg = get_retrieval()
    index = load_index()

    dense = index.as_retriever(similarity_top_k=cfg["top_k"]).retrieve(query)

    if cfg.get("hybrid"):
        # Sparse pass over the same docstore.
        try:
            import warnings
            from llama_index.retrievers.bm25 import BM25Retriever

            with warnings.catch_warnings():
                # An empty docstore makes bm25s emit "Mean of empty slice" / divide
                # warnings while building; suppress both build and query noise.
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="bm25s")
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
                bm25 = BM25Retriever.from_defaults(
                    docstore=index.docstore, similarity_top_k=cfg["bm25_k"]
                )
                sparse = bm25.retrieve(query)
        except Exception:
            sparse = []  # BM25 optional; degrade to dense-only gracefully
        candidates = _rrf([dense, sparse]) if sparse else dense
    else:
        candidates = dense

    if cfg.get("rerank"):
        reranked = _rerank(query, candidates, cfg.get("rerank_model") or _DEFAULT_RERANK_MODEL)
        if reranked is not None:
            candidates = reranked

    return candidates[: cfg["final_k"]]
