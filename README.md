# Memory Optimizer

This project provides a lightweight implementation of a dynamic context
optimization system for language models.  It includes SQLite-backed storage of
"memories", scoring for relevance, and context assembly within a token budget.

The code in `ai_memory/` includes:

* A lightweight `MemoryDatabase` built on SQLite with a normalized schema,
  Write-Ahead Logging enabled and automatic indexes/triggers for consistency.
* An improved regex based entity extractor capable of identifying emails,
  phone numbers, URLs, dates and other common entities.
* A heuristic `TokenCounter` that estimates token usage for multiple model
  types without external dependencies.
* Utilities for importing existing conversation logs via JSON with validation
  and batch support.
* 2025-07-16 → switched to SQLite backend, added conversation session layer.
* A Flask-based API server for adding memories, building context, and exporting
  data (`python -m ai_memory.api`).
* Built-in model configuration with context budgets via `model_config.py`.
* Convenience setup script `setup_memory_system.sh` to install all
  dependencies from `requirements.txt` (including Huggingface packages) and
  provide usage examples.

## Quick start

Install dependencies from `requirements.txt` and create a helpful alias:

```bash
./setup_memory_system.sh
```

Add a memory and list it back:

```bash
aimem add "hello world" -c demo
aimem list -n 1 -c demo
```

Run the API server and fetch context via HTTP:

```bash
python -m ai_memory.api &
# In another terminal
curl -XPOST localhost:5678/context -d '{"query":"hello"}' \
     -H "Content-Type: application/json"
```

The `aimem list` command also supports filtering by detected entity:

```bash
aimem list --entity foo@example.com
```

Import a structured JSON log into the memory database:

```bash
aimem import ./chatlogs/claude_memory.json
```

To compact old memories into a summary, run:

```bash
aimem vacuum
```

Embed a chat log into a vector index. For ChatGPT exports use
`--json-extract messages` so only the conversation text is embedded:

```bash
aimem vectorize ./chatlogs/claude_memory.json --vector-index ./index.faiss \
  --json-extract messages
```

### Embedding & Metadata

```bash
aimem vectorize export.json --json-extract auto \
      --vector-index ~/aimemorysystem/memory_store.index
# Produces memory_store.index + memory_store.pkl
```

Integration example:

```bash
export LUNA_VECTOR_DIR="$HOME/aimemorysystem"
aimem vectorize data.json --json-extract all
luna "Summarize the memories"
chat-with-luna "Why did we choose FAISS?"
# Chat-with-Luna automatically pulls relevant context from your
# previous interactions before sending the query to Luna.
```

Environment variables:

- `LUNA_VECTOR_DIR` – base directory for vector index and metadata
- `LUNA_VECTOR_INDEX` – override full path to index file
- `AIMEM_MAX_MEMORIES` – max fragments before compaction (default 1000)
- `AIMEM_SUMMARY_TOKENS` – token limit for summaries (default 120)
- `AIMEM_COMPRESS_BATCH` – number of fragments to compress at once (default 500)

`--json-extract` controls how JSON logs are parsed:

* `auto` – default, tries `messages` then all strings
* `messages` – expects `{conversations:[{messages:[{content:...}]}]}`
* `all` – recursively embed all string values up to 2 kB
* `none` – disable JSON parsing

## Automated ZIP ingestion

The `ingest-zip` command extracts chat log archives and imports any `*memory.json` files it finds:

```bash
aimem ingest-zip --src ~/Downloads --dest ~/chatlogs
# add -v for progress output
# aimem ingest-zip --src ~/Downloads --dest ~/chatlogs --verbose
```

You can also run `ai_memory/ingest/process_new_memories.sh` which calls the same
utility with sensible defaults.

## Troubleshooting

* **No vector memory found**: check that `LUNA_VECTOR_DIR` points to the folder
  containing `memory_store.index` and metadata files. Run `aimem vectorize` to
  create them if needed.
* **Index/metadata mismatch**: verify `memory_store.index`, `memory_store.pkl`
  and `memory_store.memories.pkl` exist and contain the same number of entries.
* Use `LUNA_VECTOR_INDEX` to specify a custom path when running from a different
  working directory.
