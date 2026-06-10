"""Neuroscience entity grounding (extension point).

First cut: an LLM-based extractor that pulls candidate neuro entities (brain
regions, neurotransmitters, genes/proteins, methods) from a query so the
retriever can expand/filter. Replace the `_ONTOLOGIES` hooks with real lookups
(Allen Brain Atlas, NCBI Gene, UniProt, ChEBI, MeSH) as you build them out.
"""
from __future__ import annotations

import json

from scirag.llm.router import complete

# Wire these to real resolvers next:
#   brain_region -> Allen Brain Atlas / NeuroNames / NIFSTD
#   gene/protein -> NCBI Gene, UniProt
#   neurotransmitter/drug -> ChEBI, MeSH pharmacological actions
#   method -> controlled vocab (patch-clamp, 2P imaging, optogenetics, ...)
_ONTOLOGIES = ("brain_region", "neurotransmitter", "gene_protein", "method", "species")

_PROMPT = (
    "Extract neuroscience entities from the query. Return strict JSON with keys "
    f"{list(_ONTOLOGIES)}, each a list of strings (empty if none). No prose."
)


def extract_entities(query: str) -> dict[str, list[str]]:
    messages = [
        {"role": "system", "content": _PROMPT},
        {"role": "user", "content": query},
    ]
    raw = complete("neuro_entity", messages, temperature=0.0, max_tokens=400)
    try:
        data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return {k: [] for k in _ONTOLOGIES}
    return {k: data.get(k, []) for k in _ONTOLOGIES}


def expand_query(query: str, entities: dict[str, list[str]]) -> str:
    """Append grounded terms to the query to sharpen PubMed/dense retrieval."""
    terms = [t for vals in entities.values() for t in vals]
    return f"{query} {' '.join(terms)}".strip() if terms else query
