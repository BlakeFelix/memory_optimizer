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
