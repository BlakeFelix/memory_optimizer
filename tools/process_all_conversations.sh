#!/bin/bash
# Process all ChatGPT conversation exports

set -e  # Exit on error
cd "$(dirname "$0")/.."

# Backup existing vector store
echo "Backing up existing vector store..."
mkdir -p ~/aimemorysystem/backup
mv ~/aimemorysystem/memory_store.* ~/aimemorysystem/backup/ 2>/dev/null || true

# Process each conversations.json file
echo "Processing ChatGPT conversations..."
find ~/chatlogs -name "conversations.json" -type f | while read -r file; do
    echo "Processing: $file"
    python tools/extract_chatgpt_messages.py "$file"
done

# Check results
echo -e "\nChecking results..."
python - <<'PY'
import faiss
from pathlib import Path
try:
    idx = faiss.read_index(str(Path.home() / 'aimemorysystem' / 'memory_store.index'))
    print(f'Total vectors: {idx.ntotal}')
except Exception as e:
    print(f'Error: {e}')
PY
