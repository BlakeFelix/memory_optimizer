#!/usr/bin/env bash
# -------------------------------------------------------------------
# Rebuild Luna’s FAISS index from SQLite (works with legacy ai_memory)
# -------------------------------------------------------------------
set -euo pipefail

# 0. Activate the venv Luna uses
source ~/aimemory_cpu/bin/activate

# 1. Paths
cd ~/memory_optimizer
STORE_DIR=".ai_memory"
INDEX="$STORE_DIR/memory_store.index"
TMP="$(mktemp "${STORE_DIR}/export_XXXX.json")"

# 2. Export conversations → JSON -----------------------------------
echo "[luna‑revector] exporting …"
if ! python -m ai_memory.cli export -o "$TMP" -c conversations ; then
  echo "[luna‑revector] export failed – abort" >&2
  rm -f "$TMP"
  exit 1
fi

# 3. Vectorise ------------------------------------------------------
echo "[luna‑revector] vectorising → $INDEX"
python -m ai_memory.cli vectorize "$TMP" --vector-index "$INDEX" --json-extract all

# 4. Tidy
rm -f "$TMP"
echo "✓ Rebuilt at $(date)"
