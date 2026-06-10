# scirag-agent

Scientific RAG · PubMed/PMC — interactive shell for literature retrieval,
indexing, and LLM-grounded Q&A.

Built with **LlamaIndex** (indexing/retrieval), **LangGraph** (orchestration),
and **LiteLLM** (LLM router). Runs fully local via Ollama or with frontier
models (Claude/OpenAI) — swappable per agent in `configs/models.yaml`.

---

## Install

```bash
git clone <repo> && cd scireg
uv sync
```

### Local models (Ollama)

```bash
ollama serve
ollama pull qwen2.5:14b-instruct-q4_K_M   # LLM (~9 GB)
ollama pull bge-m3                         # embeddings (~1.2 GB)
```

### API keys (optional — needed only for frontier backends)

```ansi
[1;32mscirag[0m [36m❯[0m [36m/env[0m set NCBI_API_KEY     <key>   [2m# raises PubMed rate limit to 10 req/s[0m
[1;32mscirag[0m [36m❯[0m [36m/env[0m set ANTHROPIC_API_KEY <key>   [2m# for claude-sonnet / claude-opus[0m
[1;32mscirag[0m [36m❯[0m [36m/env[0m set OPENAI_API_KEY     <key>   [2m# for gpt-4o[0m
```

Keys are stored in `~/.scirag-agent/.env` — never in the repo.

---

## Interactive shell

```bash
scirag
```

```ansi
╭──────────────────────────────────────────────────────╮
│  [1;36mscirag-agent[0m  [2mscientific RAG · PubMed/PMC[0m           │
│──────────────────────────────────────────────────────│
│  [2mllm:[0m          [36mollama/qwen2.5:14b-instruct-q4_K_M[0m   [2m/model to change[0m
│  [2membedding:[0m    [2mbge-m3[0m
│  [2mollama:[0m       [32mrunning[0m
│  [2mproject:[0m      [2mnone (global)[0m
│  [2mindex:[0m        [2m0 article(s)[0m
│  [2mdirectory:[0m    [2m~/code/scireg[0m
╰──────────────────────────────────────────────────────╯

[1;32mscirag[0m [36m❯[0m
```

---

## Core workflow

### 1 — Search PubMed

```ansi
[1;32mscirag[0m [36m❯[0m /search "anterior posterior retrosplenial cortex" --retmax 5
```

```ansi
[1;36m41881980[0m Anterior and posterior retrosplenial cortex... [2m(2026)[0m
  [32mPMC✓[0m [32mDOI✓[0m [32mABS✓[0m  [2mNature communications[0m
  [2mhttps://pubmed.ncbi.nlm.nih.gov/41881980/[0m

[1;36m35029643[0m Grid cells and spatial coding in the RSC...   [2m(2022)[0m
  [31mPMC✗[0m [32mDOI✓[0m [32mABS✓[0m  [2mNeuron[0m
  [2mhttps://pubmed.ncbi.nlm.nih.gov/35029643/[0m

[32m5 articles[0m  —  PMC: 1  |  DOI/Unpaywall: 4  |  abstract-only: 4
```

Badges: `PMC✓` full Results-section text · `DOI✓` Unpaywall free PDF · `ABS✓` abstract fallback

### 2 — Index papers interactively

```ansi
[1;32mscirag[0m [36m❯[0m /index "retrosplenial cortex" --retmax 10 [36m--full-text[0m
```

```ansi
 1. [1;36m41881980[0m  Anterior and posterior retrosplenial cortex...  [2m(2026)[0m  [32mPMC✓[0m [32mDOI✓[0m
     [2mhttps://pubmed.ncbi.nlm.nih.gov/41881980/[0m
 2. [2m[yellow] [indexed] [0m[2m[strike]35029643  Grid cells and spatial coding[0m  [2m(2022)[0m  [32mDOI✓[0m
     [2mhttps://pubmed.ncbi.nlm.nih.gov/35029643/[0m

? Select articles to index  (space = toggle, a = all, i = invert, enter = confirm):
 » [32m●[0m 41881980  Anterior and posterior retrosplenial...  (2026)  [PMC✓, DOI✓]
   [2m○ 35029643  Grid cells...  (2022)  [indexed, DOI✓][0m
```

`--full-text` enriches each selected article through:

```
PMC full text (Results section only)
  → Unpaywall PDF  (Results section only)
    → abstract fallback
      → warning + URL for manual download
```

Manual PDF import when automatic retrieval fails:

```ansi
[1;32mscirag[0m [36m❯[0m /import-pdf ~/Downloads/41881980.pdf
[1;32mscirag[0m [36m❯[0m /import-dir ~/Downloads/papers/
```

### 3 — Retrieve (no LLM)

```ansi
[1;32mscirag[0m [36m❯[0m /retrieve "visuospatial coding retrosplenial"
```

```ansi
[1;36m41881980[0m Anterior and posterior retrosplenial cortex...  [2m(2026)[0m  [32mresults[0m
  [2mhttps://pubmed.ncbi.nlm.nih.gov/41881980/[0m
  [2mThe anterior RSC receives strong visual drive while the posterior RSC…[0m

[32m6 chunks retrieved[0m
```

### 4 — Ask with LLM (RAG)

```ansi
[1;32mscirag[0m [36m❯[0m /llm "What distinguishes anterior from posterior RSC?"
```

```ansi
─────────────────────────── Sources ───────────────────────────
  [1;36m41881980[0m  Anterior and posterior retrosplenial cortex...  [2m(2026)[0m  [32mresults[0m
  [2mThe anterior RSC projects to visual areas while the posterior RSC…[0m
  [2mhttps://pubmed.ncbi.nlm.nih.gov/41881980/[0m
─────────────────────────── Answer ────────────────────────────
The anterior RSC is predominantly driven by visual input [41881980],
whereas the posterior RSC is more strongly connected to spatial and
navigational circuits [41881980].

[2m(conversation turn 1 — /llm --reset to clear)[0m
```

Follow-up questions work within the same session:

```ansi
[1;32mscirag[0m [36m❯[0m /llm "Which cell types mediate this?"   [2m# uses prior context[0m
[1;32mscirag[0m [36m❯[0m /llm --reset                            [2m# clear history[0m
```

---

## Projects

Separate indexes per research topic, stored in `~/.scirag-agent/projects/`.

```ansi
[1;32mscirag[0m [36m❯[0m /create-project rsc "Retrosplenial cortex circuits"
Created project [1;36mrsc[0m and switched to it.

[1;32mscirag[36m[rsc][0m [36m❯[0m /create-project place-cells "Hippocampal navigation"
Created project [1;36mplace-cells[0m and switched to it.

[1;32mscirag[36m[place-cells][0m [36m❯[0m /project
[1;36m● place-cells[0m  Hippocampal navigation          [2mcreated 2026-06-10[0m
[2m○ rsc          Retrosplenial cortex circuits  created 2026-06-10[0m

[1;32mscirag[36m[place-cells][0m [36m❯[0m /project rsc
Switched to project [1;36mrsc[0m.

[1;32mscirag[36m[rsc][0m [36m❯[0m /project --default
[2mSwitched to default global index.[0m
```

---

## LLM backends

```ansi
[1;32mscirag[0m [36m❯[0m /model
```

```ansi
? Select LLM backend  (↑↓ to move, enter to confirm):
 » [1;36mlocal-qwen14b [0m   ollama/qwen2.5:14b-instruct-q4_K_M            ← active
   local-llama8b    ollama/llama3.1:8b-instruct-q4_K_M
   local-deepseek   ollama/deepseek-r1:14b
   claude-sonnet    anthropic/claude-sonnet-4-6   [33m[ANTHROPIC_API_KEY][0m
   claude-opus      anthropic/claude-opus-4-8     [33m[ANTHROPIC_API_KEY][0m
   gpt              openai/gpt-4o                 [33m[OPENAI_API_KEY][0m
```

Switch applies to all agents for the current session — `configs/models.yaml` is never modified.

---

## All shell commands

| Command | Description |
|---|---|
| `/search <query> [--retmax N]` | PubMed search with availability indicators |
| `/index <query> [--retmax N] [--full-text]` | Fetch, preview, select, embed |
| `/retrieve <query>` | Query local index (no LLM) |
| `/llm <question> [--reset]` | RAG answer with sources + conversation memory |
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

## Full-text retrieval

Only the **Results section** is indexed — not introduction, methods, or discussion.

| Source | Coverage | Notes |
|---|---|---|
| PMC full text | ~40 % of PubMed | Open-access articles only |
| Unpaywall PDF | +20–30 % | Requires DOI; legal free copy |
| Manual PDF import | anything | `/import-pdf` or `/import-dir` |
| Abstract fallback | 100 % | Always available |
