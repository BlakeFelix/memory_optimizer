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
* Convenience setup script `setup_memory_system.sh` to install dependencies
  and provide usage examples.

## Quick start

Install dependencies and create a helpful alias:

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
