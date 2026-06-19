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
- `uv sync` — install. `--extra all` bundles the light extras (mcp, ui).
  `rerank` is **separate** (`uv sync --extra all --extra rerank`) because it pulls
  sentence-transformers + torch — torch is pinned CPU-only in `[tool.uv.sources]`
  (the reranker just re-scores candidates; all inference is via Ollama), keeping it
  ~1 GB not the ~6 GB CUDA stack. Without `rerank`, retrieval falls back to RRF order.
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
- **model / effort / rag have two surfaces** (`config.py`): the **CLI** (`scirag model|effort|rag`)
  persists a *default* to `~/.scirag-agent/settings.yaml`; the **shell** (`/model`,`/effort`,`/rag`)
  sets a *session* override. Resolution order: session override → `settings.yaml` default →
  shipped YAML config. Never hardcode model choices in code — swap in `configs/models.yaml`
  or via these.
- Reasoning effort (`/effort low|medium|high`, or `scirag effort` for the default) tunes
  reasoning depth vs. speed. The router maps it per backend (`llm/router.py`): Ollama thinking
  models toggle `think` (low = off); Claude/OpenAI APIs use litellm's `reasoning_effort`;
  the CLI backends pass it through too (`claude -p --effort`, `codex -c
  model_reasoning_effort`). Effort also scales the answer token budget. Defaults to `medium`.
- Retrieval params (`/rag`, or `scirag rag` for the default) — `final_k`, `top_k`,
  `bm25_k`, `hybrid`, `rag_score_threshold`, `rerank` — set via a picker with per-param help, or
  `/rag final_k 4` shorthand. Merged in `config.get_retrieval()` (which `retriever.py` and
  `agents/pipeline.py` read); `chunk_size`/`chunk_overlap` are index-time only and
  intentionally excluded.
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
- `src/scirag/config.py` — loads `configs/models.yaml` + `configs/pipeline.yaml` + `.env`,
  plus the `~/.scirag-agent/settings.yaml` overlay (persistent model/effort/rag defaults
  set via the CLI; session overrides layer on top).
- `src/scirag/projects.py` — project management; each project is an isolated LanceDB index.
  Index access routes through `get_active_db_uri()`, resolved as: `using_project()` scope
  → `SCIRAG_PROJECT` env → persisted `.active_project`. The active project is shared across
  shell and CLI; the CLI's `ask`/`export`/`ui` accept `--project <name>`/`--global` to scope
  one run (via `SCIRAG_PROJECT`, which also crosses the `ui` Chainlit subprocess) without
  mutating it. Each project also carries an optional `system_prompt` (stored in
  `projects.json`); `get_active_system_prompt()` resolves it with the same precedence and
  `pipeline.py` appends it to the synthesis system prompt (blank = built-in default). Set it
  at `/create-project` (interactive single line) or later with `/system-prompt --edit` ($EDITOR);
  the global index has none.
- `src/scirag/llm/router.py` — `complete(agent, messages)`; the ONLY place LLMs are called.
- `src/scirag/agents/pipeline.py` — canonical RAG pipeline: entity extraction ->
  retrieval -> relevance gating -> grounded-prompt assembly.
- `src/scirag/agents/synthesize.py` — cited-answer synthesis agent.
- `src/scirag/sources/pubmed.py` — NCBI E-utilities client (`Article` dataclass).
  `search()` is keyword esearch (Boolean/field syntax); `search_semantic()` ranks the
  same PubMed corpus by relevance via Europe PMC (`SRC:MED`) so a natural-language
  question works (esearch would mangle it — e.g. read "in human" as an `[Author]`).
  Both return PMIDs, so `fetch()` and everything downstream are identical.
- `src/scirag/sources/biorxiv.py` — bioRxiv source; keyword search via Europe PMC
  (the bioRxiv API has no search endpoint), direct-DOI metadata via the bioRxiv API,
  and full-text Results from the JATS XML — fetched with `curl_cffi` browser
  impersonation since biorxiv.org's full-text host is Cloudflare-gated. Builds the
  same `Article` with the preprint DOI in the `pmid` slot and `source="biorxiv"`.
  Because that keyword search *is* Europe PMC relevance ranking, `bindex` accepts a
  natural-language question directly (no Boolean syntax) — it never had PubMed's
  esearch problem, so there's no `--semantic` flag to add (it's effectively always on).
- `src/scirag/sources/pdf.py` — PDF ingestion: resolve a PDF to its source record +
  isolate the Results section.
- `src/scirag/sources/mendeley.py` — import from the local Mendeley Reference Manager
  library (offline): reads its SQLite store + already-extracted PDF text, reusing
  `pdf.extract_results_section`. Keys by PMID (else bioRxiv DOI, else `mendeley-<id>`)
  so imports dedup against PubMed/bioRxiv. Auto-detects the per-OS install
  (macOS/Windows/Linux); override the DB/PDF location via `sources.mendeley.db_path` /
  `userfiles_path` in `pipeline.yaml` (or the `MENDELEY_DB_PATH` env var, which wins).
- `src/scirag/sources/zotero.py` — import from the local Zotero library (offline),
  mirroring the Mendeley source. Reads `zotero.sqlite` (a normalised relational
  schema, not Mendeley's FTS tables) + Zotero's own full-text cache
  (`storage/<key>/.zotero-ft-cache`, else the attached PDF), reusing
  `pdf.extract_results_section`. Mines the PMID from the CSL `extra` field. Keys by
  PMID (else bioRxiv DOI, else `zotero-<id>`) so imports dedup against
  PubMed/bioRxiv/Mendeley. Defaults to `~/Zotero`; override the DB/storage location
  via `sources.zotero.db_path` / `storage_path` in `pipeline.yaml` (or the
  `ZOTERO_DB_PATH` env var, which wins).
- `src/scirag/ingest/index.py` — LlamaIndex -> LanceDB (embedded, at `data/lancedb`).
- `src/scirag/retrieval/retriever.py` — hybrid dense + BM25 with RRF fusion.
- `src/scirag/shell.py` — interactive REPL, launched by `scirag` with no arguments.
- `src/scirag/ui.py` — Chainlit web UI (`scirag ui`, needs `--extra ui`).
- `src/scirag/mcp_server/server.py` — exposes retrieval as MCP tools (optional extra).
- `src/scirag/cli.py` — Typer CLI entry point (`scirag = scirag.cli:app`).

## Run
The CLI is deliberately small — launch, configure, and the two scriptable ops.
Everything operational (index/bindex/retrieve/show/import*/remove/clear-db/projects)
is interactive and lives only in the **shell**.
```
# CLI (outside the shell)
scirag                                          # interactive shell (no args)
scirag ui                                       # Chainlit web UI (needs --extra ui)
scirag ask "How do place cells remap across environments?"   # one-shot grounded answer
scirag export [path]                            # export indexed papers' metadata to CSV
scirag env set NCBI_API_KEY <key>               # manage API keys in ~/.scirag-agent/.env
scirag model claude-sonnet                      # persist default backend → settings.yaml
scirag effort high                              # persist default reasoning effort
scirag rag final_k 12                           # persist default retrieval param
uv run python -m scirag.mcp_server.server       # MCP server (needs --extra mcp)

# Shell commands (inside `scirag`)
/index "retrosplenial cortex" --semantic        # fetch/select/index PubMed (sentences ok)
/bindex "how do place cells remap"              # index bioRxiv (always relevance-ranked)
/retrieve "place cells remapping"  ·  /show <pmid|doi>  ·  /import path/to.pdf
/import-mendeley · /import-zotero · /import-text  ·  /model · /effort · /rag (session-only)
/create-project <name>                          # prompts for an optional system prompt
/system-prompt [--edit] [--default]             # view the project's prompt (--edit → $EDITOR, --default resets)
```
(`scirag` is the installed console script; prefix with `uv run` if not on PATH.)
`scirag model|effort|rag` persist a **default** to `~/.scirag-agent/settings.yaml`; the
shell `/model`,`/effort`,`/rag` are **session** overrides on top (see Models & reasoning).

## Conventions
- **All LLM calls go through `scirag.llm.router.complete(agent, ...)`** — never call
  `litellm`/provider SDKs directly. Add/swap models in `configs/models.yaml`, not in code.
- Agent role names in `models.yaml` (`planner`, `retriever`, `synthesizer`, `critic`)
  are the contract between config and code. Only `synthesizer` is wired to an LLM today;
  the others are configured ahead of the multi-agent buildout (see Roadmap).
- Synthesis cites every claim with a human-readable **author-year** marker
  (e.g. `(Powell et al., 2020)`), built by `scirag.cite.citation()` from the
  source metadata and used in the answer + all source listings (shell answers,
  `/retrieve`, web UI). The PMID (PubMed) / DOI (`10.1101/…`, bioRxiv) still lives
  in the metadata `pmid` field and remains the system-wide **primary key** (dedup,
  `show`/`remove`, and the `[id: …]` shown in each source block for traceability) —
  it's just no longer the citation marker the model emits.
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
