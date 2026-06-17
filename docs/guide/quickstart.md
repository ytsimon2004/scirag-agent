# Quickstart

This walks through the core loop: create a project, index a few papers, inspect what
was stored, and ask a grounded question.

Launch the interactive shell (the `scirag` console script with no arguments):

```bash
scirag
```

Everything operational lives in the shell. The CLI outside it is deliberately small
(see [Shell commands](shell-commands.md)).

## 1 — Create a project

Each project is an isolated LanceDB index, so you can keep separate literature scopes.

```text
/create-project place-cells
```

You'll be prompted for an optional **system prompt** that is appended to the synthesis
prompt for this project (blank = built-in default). Edit it later with
`/system-prompt --edit`.

## 2 — Choose a model (optional)

The default backend for every agent is `local-qwen3-14b` (fully offline). To switch
for this session:

```text
/model claude-sonnet
```

To persist a default across sessions, use the CLI: `scirag model claude-sonnet`.
See [Configuration](configuration.md) for the full override hierarchy.

## 3 — Index papers

```text
/index "retrosplenial cortex head direction" --semantic
/bindex "how do place cells remap across environments"
```

- `/index` searches **PubMed**. Add `--semantic` to rank by relevance via Europe PMC
  so a natural-language phrase works (plain esearch would mis-parse it). You then
  select which results to index.
- `/bindex` searches **bioRxiv** (always relevance-ranked, so plain questions are fine).

You can also import local material:

```text
/import path/to/paper.pdf
/import-mendeley
/import-zotero
/import-text
```

## 4 — Check what's stored

```text
/retrieve "place cells remapping"     # show ranked chunks + scores
/show 31112130                         # show stored text for a PMID/DOI
```

## 5 — Ask, grounded in your papers

```text
How do place cells remap across environments?
```

The answer cites every claim with an author-year marker (e.g. `(Powell et al., 2020)`)
and lists the source passages it used. If nothing in the index clears the relevance
threshold, the model says so explicitly and falls back to general knowledge.

### One-shot from the CLI

For scripting, skip the shell:

```bash
scirag ask "How do place cells remap across environments?"
scirag ask "..." --project place-cells   # scope a single run
```
