#!/usr/bin/env bash
# Wrapper to ingest new chat log ZIPs
set -e
SRC="${SRC:-$HOME/Downloads}"
DEST="${DEST:-$HOME/chatlogs}"
INDEX="${INDEX:-}"
MODEL="${MODEL:-}"

ARGS=(--src "$SRC" --dest "$DEST")
if [ -n "$INDEX" ]; then
    ARGS+=(--index "$INDEX")
fi
if [ -n "$MODEL" ]; then
    ARGS+=(--model "$MODEL")
fi

python -m ai_memory.ingest.zip_watcher "${ARGS[@]}" "$@"
