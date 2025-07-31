#!/usr/bin/env bash
# ---------------------------------------------------------------
# ~/memory_optimizer/setup_luna.sh
# Provision venv, dependencies, models, symlinks, and systemd jobs
# ---------------------------------------------------------------
set -euo pipefail
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENVDIR="$HOME/aimemory_cpu"
BIN="$HOME/bin"

echo "▶ Creating/activating venv: $VENVDIR"
python3 -m venv "$VENVDIR"
source "$VENVDIR/bin/activate"
pip -q install --upgrade pip

echo "▶ Installing Python deps"
pip install -q "ai-memory>=0.4" faiss-cpu sentence-transformers torch \
                 scipy flask click

echo "▶ Pulling Ollama base & vision models"
ollama pull llama3:70b-instruct-q4_K_M
ollama pull llava:13b

echo "▶ Re‑creating Luna alias if missing"
if ! ollama show luna &>/dev/null; then
  printf 'FROM llama3:70b-instruct-q4_K_M\n' > "$ROOT/luna.modelfile"
  ollama create luna -f "$ROOT/luna.modelfile"
fi

echo "▶ Symlinking helper CLIs into $BIN"
mkdir -p "$BIN"
ln -sf "$ROOT/ai_memory/chat_with_luna.py"      "$BIN/luna"
ln -sf "$ROOT/maintenance/luna_revector.sh"     "$BIN/luna-revector"

echo "▶ Installing revector systemd timer"
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"
cp "$ROOT/maintenance/luna-revector.service" "$UNIT_DIR/"
cp "$ROOT/maintenance/luna-revector.timer"   "$UNIT_DIR/"
systemctl --user daemon-reload
systemctl --user enable --now luna-revector.timer

echo "✅  Setup complete.  Try:  luna \"hello\""
