"""Synthesis agent: turns retrieved passages into a grounded, cited answer.
Citations use [PMID] markers tied to the source metadata.
"""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore

from scirag.llm.router import complete

SYSTEM = (
    "You are a scientific literature assistant. Answer ONLY from the provided "
    "sources. Cite every claim with its [PMID] or [DOI] marker. If the sources "
    "do not support an answer, say so explicitly."
)


def _format_sources(nodes: list[NodeWithScore]) -> str:
    blocks = []
    for n in nodes:
        md = n.node.metadata
        pmid = md.get("pmid", "?")
        title = md.get("title", "")
        header = f"[{pmid}] {title} ({md.get('year', 'n.d.')})"
        authors = md.get("authors") or ""
        if authors:
            header += f" — {authors}"
        blocks.append(f"{header}\n{n.node.get_content()}")
    return "\n\n---\n\n".join(blocks)


def synthesize(query: str, nodes: list[NodeWithScore]) -> str:
    sources = _format_sources(nodes)
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Question: {query}\n\nSources:\n{sources}\n\n"
            "Write a concise, cited answer.",
        },
    ]
    return complete("synthesizer", messages, max_tokens=1200)
