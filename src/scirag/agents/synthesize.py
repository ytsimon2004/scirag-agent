"""Synthesis agent: turns retrieved passages into a grounded, cited answer.
The answer cites with author-year markers (e.g. Powell et al., 2020); each source
block still carries its PMID/DOI id (the system-wide primary key) for traceability.
"""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore

from scirag.llm.router import complete

SYSTEM = (
    "You are a scientific literature assistant. Answer ONLY from the provided "
    "sources. Cite every claim with the author-year marker shown in parentheses "
    "for each source, e.g. (Powell et al., 2020), placed right after the claim. "
    "When several sources support a claim, cite each, e.g. "
    "(Powell et al., 2020; Alexander, 2023). Use the marker exactly as given; do "
    "not cite raw PMIDs or DOIs. If the sources do not support an answer, say so "
    "explicitly."
)


def _format_sources(nodes: list[NodeWithScore]) -> str:
    from scirag.cite import citation

    blocks = []
    for n in nodes:
        md = n.node.metadata
        # The author-year citation is the marker the model must use in the answer;
        # the id (PMID/DOI) stays in the block for traceability, not for citing.
        header = f"({citation(md)}) {md.get('title', '')} [id: {md.get('pmid', '?')}]"
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
