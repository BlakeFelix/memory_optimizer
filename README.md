# Memory Optimizer

This project provides a lightweight implementation of a dynamic context
optimization system for language models.  It includes in-memory storage of
"memories", scoring for relevance, and context assembly within a token budget.

The code in `aimemory/` includes:

* A lightweight `MemoryDatabase` built on SQLite with a normalized schema,
  Write-Ahead Logging enabled and automatic indexes/triggers for consistency.
* An improved regex based entity extractor capable of identifying emails,
  phone numbers, URLs, dates and other common entities.
* A heuristic `TokenCounter` that estimates token usage for multiple model
  types without external dependencies.
* Utilities for importing existing conversation logs via JSON with validation
  and batch support.
