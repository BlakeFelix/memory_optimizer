#!/bin/bash
# Process all ChatGPT conversation exports with CPU mode

set -e  # Exit on error
cd "$(dirname "$0")/.."

# Setup environment
export CUDA_VISIBLE_DEVICES=""
export LUNA_VECTOR_DIR="$HOME/aimemorysystem"

# Backup existing data
if [ -d "$LUNA_VECTOR_DIR" ]; then
    echo "Backing up existing vector store..."
    backup_dir="$LUNA_VECTOR_DIR/backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    cp "$LUNA_VECTOR_DIR"/memory_store.* "$backup_dir/" 2>/dev/null || true
fi

# Clean start
echo "Starting fresh..."
rm -f "$LUNA_VECTOR_DIR"/memory_store.*

# Process each conversations.json file
echo "Processing ChatGPT conversations..."
find ~/chatlogs -name "conversations.json" -type f | while read -r file; do
    echo "Processing: $file"
    python tools/extract_chatgpt_messages.py "$file"
    
    # Small delay to avoid overwhelming the system
    sleep 0.5
done

# Check results
echo -e "\nChecking results..."
python - <<'PY'
import faiss, pickle, os

vec_dir = os.getenv('LUNA_VECTOR_DIR', os.path.expanduser('~/aimemorysystem'))
idx_path = os.path.join(vec_dir, 'memory_store.index')
pkl_path = os.path.join(vec_dir, 'memory_store.pkl')

try:
    idx = faiss.read_index(idx_path)
    print(f'Total vectors: {idx.ntotal}')
    
    with open(pkl_path, 'rb') as f:
        meta = pickle.load(f)
    print(f'Metadata entries: {len(meta)}')
    
    if meta:
        entries = list(meta.values()) if isinstance(meta, dict) else meta
        print('\nFirst 3 entries:')
        for i, entry in enumerate(entries[:3], 1):
            text = entry.get('text', '') if isinstance(entry, dict) else str(entry)
            print(f"{i}. {text[:100]}...")
except Exception as e:
    print(f'Error: {e}')
PY

echo -e "\nDone! Test with: luna-cpu \"What have we discussed about memory systems?\""
