# scirag

Multi-agent **retrieval-augmented generation** over the neuroscience literature —
grounded answers with citations, built on PubMed, PMC, and bioRxiv.

## What this does

Ask a natural-language question and scirag retrieves the most relevant passages
from your indexed papers, then synthesizes a **cited answer** where every claim is
tied to an author-year marker (e.g. `(Powell et al., 2020)`).

- **Hybrid retrieval** — dense (bge-m3) + BM25 fused with RRF, optional cross-encoder reranking.
- **Grounded synthesis** — answers are constrained to the retrieved sources; no source, no claim.
- **Local-first** — runs fully offline on Ollama (Qwen3-14B) by default; frontier backends optional.

## Tips

- Ask specific, mechanistic questions ("How do place cells remap across environments?")
  rather than broad keyword queries — semantic retrieval rewards natural language.
- Each answer lists its sources; the `[id: …]` is the PMID (PubMed) or DOI (bioRxiv)
  primary key for traceability.
- **Import a paper:** use the 📎 attach button (or drag-and-drop) to add a **PDF** to
  this project's index. Only PDFs are accepted — other file types are skipped.
- Manage the index, projects, and models from the `scirag` shell — see the project README.
