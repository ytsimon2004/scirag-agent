# Shell commands

scirag has two surfaces: a small **CLI** for launching/configuring and the two
scriptable operations, and an interactive **shell** where everything operational
lives.

## CLI (outside the shell)

```bash
scirag                                # interactive shell (no args)
scirag ui                             # Chainlit web UI (needs --extra ui)
scirag ask "How do place cells remap?"  # one-shot grounded answer
scirag export [path]                  # export indexed papers' metadata to CSV
scirag env set NCBI_API_KEY <key>     # manage API keys in ~/.scirag-agent/.env
scirag model claude-sonnet            # persist default backend -> settings.yaml
scirag effort high                    # persist default reasoning effort
scirag rag final_k 12                 # persist default retrieval param
```

`ask`, `export`, and `ui` accept `--project <name>` / `--global` to scope a single
run without changing the active project.

## Shell commands (inside `scirag`)

### Indexing & sources

Type `/help` inside the shell to see the same list with live flag completion.

### Indexing & sources

| Command | Args | What it does |
|---------|------|--------------|
| `/index` | `<query> [--retmax N] [--full-text] [--semantic] [--year-from YYYY] [--year-to YYYY]` | Fetch + select + index PubMed articles (`--semantic` = relevance search, accepts sentences) |
| `/bindex` | `<query> [--retmax N] [--days-back N] [--full-text] [--year-from YYYY] [--year-to YYYY]` | Fetch + select + index bioRxiv preprints (relevance search, accepts sentences) |
| `/import` | `<path>` | Index a PDF file, or every PDF in a directory |
| `/import-mendeley` | `<query> [--retmax N]` | Search + select + index Mendeley library papers (offline) |
| `/import-zotero` | `<query> [--retmax N]` | Search + select + index Zotero library papers (offline) |
| `/import-text` | | Index free-form text (prompts for title, identifier, origin, year, author) |

```{admonition} --full-text
:class: note

`/index` and `/bindex` index abstracts by default. Add `--full-text` to also fetch
and index each paper's Results section (slower, but far richer for grounding).
```

### Inspecting the index

| Command | Args | What it does |
|---------|------|--------------|
| `/retrieve` | `<query>` | Query the local index and show ranked chunks + scores (no LLM) |
| `/show` | `<pmid>` | Print a paper's stored abstract/results text |
| `/status` | | Show index statistics |
| `/export` | `[path]` | Export indexed papers' metadata to CSV |
| `/remove` | `[pmid ...]` | Remove article(s) from the index (interactive if no args) |
| `/clear-db` | `[--force]` | Delete the active index |

### Configuration (session overrides)

| Command | Args | What it does |
|---------|------|--------------|
| `/model` | `[backend-key]` | List or switch the LLM backend |
| `/effort` | `[low\|medium\|high]` | Set reasoning effort (speed vs. accuracy) |
| `/rag` | `[<param> <value>]` | Tune retrieval params (`final_k`, `top_k`, …); no args = picker |

```{admonition} CLI default vs. shell session override
:class: note

`scirag model|effort|rag` persist a **default** to `~/.scirag-agent/settings.yaml`.
The shell `/model`, `/effort`, `/rag` set a **session** override on top. See
[Configuration](configuration.md) for the full resolution order.
```

### Projects

| Command | Args | What it does |
|---------|------|--------------|
| `/create-project` | `<name> [description]` | Create a new project and switch to it (prompts for an optional system prompt) |
| `/project` | `[name\|--default]` | List projects, or switch to one (`--default` = the global index) |
| `/delete-project` | `<name> [--force]` | Delete a project and its index |
| `/system-prompt` | `[--edit] [--default]` | View the active project's system prompt (`--edit` opens `$EDITOR`, `--default` resets it) |

### Session & utilities

| Command | Args | What it does |
|---------|------|--------------|
| `/ui` | `[--port N]` | Open the Chainlit web UI in the browser (click-to-expand sources) |
| `/env` | `[set <KEY> <val> \| unset <KEY>]` | Manage API keys in `~/.scirag-agent/.env` |
| `/help` | | Show the full command list |
| `/clear` | | Clear the conversation context |
| `/exit` | | Exit scirag (alias: `/quit`) |
