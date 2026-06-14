# scirag — Multi-agent RAG for scientific literature

Neuroscience-focused retrieval-augmented generation over PubMed/PMC, built with
**LlamaIndex** (indexing/retrieval) and **LiteLLM** (LLM routing). LLM backends —
open-source (Ollama: Qwen/Llama/DeepSeek) and frontier (Claude/OpenAI, via API or
the `claude`/`codex` CLIs) — sit behind one LiteLLM router, selectable per agent.

## Framework division of labor
- **LlamaIndex** — ingestion, chunking, embeddings, LanceDB vector store, retrievers.
- **LiteLLM** — unified LLM router across local and frontier backends.

## Environment & setup
- **uv** manages everything. Python 3.11.
- `uv sync` — install. `uv sync --extra mcp --extra eval --extra ui --extra rerank`
  for extras (or `--extra all`). `rerank` pulls sentence-transformers + torch for
  cross-encoder reranking; without it, retrieval falls back to RRF order.
- API keys live in `~/.scirag-agent/.env` (manage with `scirag env`). Set
  `NCBI_API_KEY` (raises PubMed rate limit to 10 req/s) and
  `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` only if a frontier backend is selected.
- **Ollama must be running** for local models + embeddings: `ollama serve` then
  `ollama pull qwen3:14b-q4_K_M && ollama pull bge-m3`.

## Models & reasoning
- Default backend for every agent is `local-qwen3-14b` (`ollama/qwen3:14b-q4_K_M`),
  a **hybrid-thinking** model: by default it emits a `<think>…</think>` chain before
  answering. That reasoning is the main latency cost in the pipeline. Disable it for
  speed by passing Ollama's `think: False` (or appending `/no_think` to the prompt);
  prefer a non-thinking backend (`local-llama4-scout`) for high-frequency agents and
  reserve a reasoning model (`local-deepseek-r1-32b`) for where it earns its cost.
- Swap models per agent in `configs/models.yaml` (or at runtime with `scirag model`),
  never in code.
- `scirag effort low|medium|high` (or `/effort` in the shell) tunes reasoning depth vs.
  speed for the session. The router maps it per backend (`llm/router.py`): Ollama thinking
  models toggle `think` (low = off); Claude/OpenAI APIs use litellm's `reasoning_effort`;
  the CLI backends pass it through too (`claude -p --effort`, `codex -c
  model_reasoning_effort`). Effort also scales the answer token budget. Session-only,
  defaults to `medium`.
- `scirag rag` (or `/rag` in the shell) tunes retrieval params live — `final_k`, `top_k`,
  `bm25_k`, `hybrid`, `rag_score_threshold`, `rerank` — via a picker with per-param help, or
  `/rag final_k 4` shorthand. Session-only overrides merged in `config.get_retrieval()`
  (which `retriever.py` and `agents/pipeline.py` read); `chunk_size`/`chunk_overlap` are
  index-time only and intentionally excluded.
- Retrieval (`retrieval/retriever.py`) = dense (bge-m3) + BM25 → RRF fusion → optional
  **cross-encoder rerank** (`bge-reranker-v2-m3` via sentence-transformers) → top `final_k`.
  Retrieve wide (`top_k`/`bm25_k` ~30) and let the reranker pick the best `final_k` — raises
  recall *and* precision. Reranking only reorders nodes; their cosine `.score` is preserved so
  the `rag_score_threshold` gate still works. **Off by default** (opt-in: `/rag rerank on`);
  needs `--extra rerank`; degrades to RRF order if absent. Model is lazy-loaded and cached on
  first reranked query.

## Hardware
- **Mac (36 GB unified)** is the primary local inference box — run Qwen3-14B (q4) +
  bge-m3 embeddings via Ollama. This is the default in `configs/models.yaml`.
- **Linux RTX 4060 (8 GB)** only fits 7–8B at 4-bit; best used as a remote
  embedding/reranker server. To use it, point `embeddings.api_base` (and any
  `local-*` backend `api_base`) in `models.yaml` at `http://<linux-ip>:11434`.
- To run **fully offline**, keep every agent on `local-qwen3-14b` (the default) —
  no API keys needed.

## Layout
- `src/scirag/config.py` — loads `configs/models.yaml` + `configs/pipeline.yaml` + `.env`.
- `src/scirag/projects.py` — project management; each project is an isolated LanceDB index.
- `src/scirag/llm/router.py` — `complete(agent, messages)`; the ONLY place LLMs are called.
- `src/scirag/agents/pipeline.py` — canonical RAG pipeline: entity extraction ->
  retrieval -> relevance gating -> grounded-prompt assembly.
- `src/scirag/agents/synthesize.py` — cited-answer synthesis agent.
- `src/scirag/sources/pubmed.py` — NCBI E-utilities client (`Article` dataclass).
- `src/scirag/sources/biorxiv.py` — bioRxiv source; keyword search via Europe PMC
  (the bioRxiv API has no search endpoint), direct-DOI metadata via the bioRxiv API,
  and full-text Results from the JATS XML — fetched with `curl_cffi` browser
  impersonation since biorxiv.org's full-text host is Cloudflare-gated. Builds the
  same `Article` with the preprint DOI in the `pmid` slot and `source="biorxiv"`.
- `src/scirag/sources/pdf.py` — PDF ingestion: resolve a PDF to its source record +
  isolate the Results section.
- `src/scirag/ingest/index.py` — LlamaIndex -> LanceDB (embedded, at `data/lancedb`).
- `src/scirag/retrieval/retriever.py` — hybrid dense + BM25 with RRF fusion.
- `src/scirag/shell.py` — interactive REPL, launched by `scirag` with no arguments.
- `src/scirag/ui.py` — Chainlit web UI (`scirag llm-ui`, needs `--extra ui`).
- `src/scirag/mcp_server/server.py` — exposes retrieval as MCP tools (optional extra).
- `src/scirag/cli.py` — Typer CLI entry point (`scirag = scirag.cli:app`).

## Run
```
scirag                                          # interactive shell (no args)
scirag index --retmax 30                        # interactively fetch/select/index PubMed
scirag bindex --days-back 180 --full-text       # interactively index bioRxiv preprints
scirag retrieve "place cells remapping"         # show retrieved chunks, no LLM
scirag llm "How do place cells remap across environments?"   # grounded, cited answer
scirag llm-ui                                   # Chainlit web UI (needs --extra ui)
scirag import path/to/paper.pdf                 # index a PDF (or a dir of PDFs)
scirag model                                    # list backends; pass a key to switch
scirag effort high                              # set reasoning effort (low/medium/high)
scirag rag                                      # tune retrieval params (final_k, top_k, …) — picker
scirag show <pmid|doi>                           # print a stored record's text
scirag env set NCBI_API_KEY <key>               # manage API keys in ~/.scirag-agent/.env
uv run python -m scirag.mcp_server.server       # MCP server (needs --extra mcp)
```
(`scirag` is the installed console script; prefix with `uv run` if not on PATH.)

## Conventions
- **All LLM calls go through `scirag.llm.router.complete(agent, ...)`** — never call
  `litellm`/provider SDKs directly. Add/swap models in `configs/models.yaml`, not in code.
- Agent role names in `models.yaml` (`planner`, `retriever`, `synthesizer`, `critic`)
  are the contract between config and code. Only `synthesizer` is wired to an LLM today;
  the others are configured ahead of the multi-agent buildout (see Roadmap).
- Synthesis must cite every claim with the `[id]` marker from source metadata —
  a PMID for PubMed records, a DOI (`10.1101/…`) for bioRxiv preprints. Both live
  in the metadata `pmid` field, which is the system-wide primary key (dedup,
  `show`, citations).
- PubMed and bioRxiv are the data sources; the MCP server is how agents reach them.
  Don't bypass `sources/` with ad-hoc HTTP calls elsewhere. The bioRxiv API has no
  keyword-search endpoint, so `biorxiv.search` queries Europe PMC
  (`SRC:PPR AND PUBLISHER:"bioRxiv"`) for relevance-ranked results.

## Roadmap (multi-agent buildout)
Current pipeline is linear (`extract_entities -> retrieve -> synthesize`). Next:
wire the `planner`/`retriever`/`critic` agents; LangGraph for conditional routing —
critic/verifier loop that scores citation grounding and routes back to `retrieve` on
failure; supervisor/router node; query-planner (MeSH expansion); real ontology
resolvers (Allen Brain Atlas, NCBI Gene, UniProt, ChEBI, MeSH).
(Done: hybrid dense+BM25 retrieval with RRF; cross-encoder reranking via `bge-reranker-v2-m3`.)
