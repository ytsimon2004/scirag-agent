# scirag-agent

Scientific RAG · PubMed/PMC — interactive shell for literature retrieval,
indexing, and LLM-grounded Q&A.

Built with **LlamaIndex** (indexing/retrieval), **LangGraph** (orchestration),
and **LiteLLM** (LLM router). Runs fully local via Ollama or with frontier
models (Claude/OpenAI) — swappable per agent in `configs/models.yaml`.

---

## Install

```
git clone <repo> && cd scireg
uv sync
```

### Local models (Ollama)

```
ollama serve
ollama pull qwen2.5:14b-instruct-q4_K_M   # LLM (~9 GB)
ollama pull bge-m3                         # embeddings (~1.2 GB)
```

### Web UI (optional)

```
uv sync --extra ui
```

### API keys (optional — needed only for frontier backends)

```
scirag
scirag ❯ /env set NCBI_API_KEY    <key>     # raises PubMed rate limit to 10 req/s
scirag ❯ /env set ANTHROPIC_API_KEY <key>   # for claude-sonnet / claude-opus
scirag ❯ /env set OPENAI_API_KEY  <key>     # for gpt-4o
```

Keys are stored in `~/.scirag-agent/.env` — never in the repo.

---

## Interactive shell

```
scirag
```

```
╭──────────────────────────────────────────────────────╮
│  scirag-agent  scientific RAG · PubMed/PMC           │
│──────────────────────────────────────────────────────│
│  llm:          ollama/qwen2.5:14b-instruct-q4_K_M    │
│  embedding:    bge-m3                                │
│  ollama:       running                               │
│  project:      none (global)                         │
│  index:        0 article(s)                          │
│  directory:    ~/code/scireg                         │
╰──────────────────────────────────────────────────────╯

scirag ❯
```

---

## Core workflow

### 1 — Search PubMed (no index needed)

```
scirag ❯ /search "anterior posterior retrosplenial cortex" --retmax 10
```

Shows each result with availability badges:
- `PMC✓` — full Results-section text retrievable
- `DOI✓` — Unpaywall can find a free PDF
- `ABS✓` — abstract available as fallback

### 2 — Index papers interactively

```
scirag ❯ /index "anterior posterior retrosplenial cortex" --retmax 10 --full-text
```

Fetches articles, shows a numbered list with status and clickable URLs, then
opens a checkbox prompt — select which papers to embed and store.

`--full-text` enriches each selected article through:

```
PMC full text (Results section only)
  → Unpaywall PDF (Results section only)
    → abstract fallback
      → warning + URL for manual download
```

Manual PDF import when automatic retrieval fails:

```
scirag ❯ /import-pdf ~/Downloads/41881980.pdf
scirag ❯ /import-dir ~/Downloads/papers/
```

### 3 — Retrieve (no LLM)

```
scirag ❯ /retrieve "visuospatial coding retrosplenial"
```

Shows matching chunks with PMID, title, year, `results`/`abstract` source
badge, and a clickable PubMed URL.

### 4 — Ask with LLM (RAG)

```
scirag ❯ /llm "What distinguishes anterior from posterior RSC?"
```

Shows retrieved sources first, then a cited answer. Remembers conversation
history within the session — follow-up questions work naturally.

```
scirag ❯ /llm "Which cell types mediate this?"   # follow-up
scirag ❯ /llm --reset                            # clear history
```

### 5 — Web UI

```
scirag ❯ /llm-ui              # opens http://localhost:8000
scirag ❯ /llm-ui --port 8080  # custom port
```

Requires `uv sync --extra ui` first. Features:

- Chat interface with streaming responses
- Retrieved sources shown inline before each answer (title, year, source type, snippet)
- Full source text accessible in the sidebar
- General questions answered from LLM knowledge; research questions trigger RAG automatically
- ⚙️ settings panel to switch LLM backend mid-conversation
- `/reset` in chat to clear conversation history

The web UI and shell share the same index and project — index papers in the
shell, ask questions in the browser, or mix both freely.

---

## Projects

Separate indexes per research topic, stored in `~/.scirag-agent/projects/`.

```
scirag ❯ /create-project rsc "Retrosplenial cortex circuits"
scirag ❯ /create-project place-cells "Hippocampal navigation"

scirag ❯ /project              # list all
  ● rsc          Retrosplenial cortex circuits   created 2026-06-10
  ○ place-cells  Hippocampal navigation          created 2026-06-10

scirag ❯ /project place-cells  # switch
scirag ❯ /project --default    # back to global index
scirag ❯ /delete-project rsc   # asks for confirmation
```

Prompt shows the active project: `scirag[rsc] ❯`

---

## LLM backends

```
scirag ❯ /model
```

```
Key              Model                                Needs
local-qwen14b    ollama/qwen2.5:14b-instruct-q4_K_M               ← active
local-llama8b    ollama/llama3.1:8b-instruct-q4_K_M
local-deepseek   ollama/deepseek-r1:14b
claude-sonnet    anthropic/claude-sonnet-4-6          ANTHROPIC_API_KEY
claude-opus      anthropic/claude-opus-4-8            ANTHROPIC_API_KEY
gpt              openai/gpt-4o                        OPENAI_API_KEY
```

Arrow keys to select, or `scirag ❯ /model claude-sonnet` directly.
Switch applies to all agents for the current session — `configs/models.yaml`
is never modified.

---

## All shell commands

| Command | Description |
|---|---|
| `/search <query> [--retmax N]` | PubMed search with availability indicators |
| `/index <query> [--retmax N] [--full-text]` | Fetch, preview, select, embed |
| `/retrieve <query>` | Query local index (no LLM) |
| `/llm <question> [--reset]` | RAG answer with sources + conversation memory |
| `/llm-ui [--port N]` | Open Chainlit web UI in browser |
| `/model [backend-key]` | List or switch LLM backend |
| `/import-pdf <path>` | Index a single PDF (Results section only) |
| `/import-dir <path>` | Index all PDFs in a directory |
| `/env [set\|unset <KEY> <val>]` | Manage API keys in `~/.scirag-agent/.env` |
| `/status` | Index statistics |
| `/clear-db [--force]` | Delete the active index |
| `/create-project <name> [desc]` | Create project and switch to it |
| `/project [name\|--default]` | List or switch projects |
| `/delete-project <name> [--force]` | Delete a project and its index |
| `/help` | Show all commands |
| `/clear` | Clear the screen |
| `/exit` | Exit |

All commands also available as subcommands: `scirag index "..."`, `scirag llm "..."`, etc.

---

## Data layout

```
~/.scirag-agent/
  .env                        # API keys (managed by /env)
  lancedb/                    # global index
  projects/
    rsc/lancedb/              # per-project indexes
    place-cells/lancedb/
  projects.json               # project registry
  .active_project             # active project name

configs/
  models.yaml                 # LLM backends + embeddings
  pipeline.yaml               # chunk sizes, retrieval parameters
```

---

## Full-text retrieval notes

Only the **Results section** is indexed — not introduction, methods, or
discussion. Coverage per source:

| Source | Coverage | Requires |
|---|---|---|
| PMC full text | ~40 % of PubMed | Open-access articles only |
| Unpaywall PDF | +20–30 % | DOI present; legal free copy exists |
| Manual PDF import | anything | `/import-pdf` or `/import-dir` |
| Abstract fallback | 100 % | Always available |

For papers with no free full text, `scirag` warns and prints the PubMed URL
for manual download.
