#!/usr/bin/env bash
# Setup script for AI Memory system

# 1. Install required Python packages
echo "Installing Python packages listed in requirements.txt..."
pip install -r memory_optimizer/requirements.txt

# 2. Create shell alias for convenience
ALIAS_CMD="alias aimem='python -m ai_memory.cli'"
if ! grep -qxF "$ALIAS_CMD" ~/.bashrc; then
    echo "$ALIAS_CMD" >> ~/.bashrc
    echo "Added 'aimem' alias to ~/.bashrc (reload your shell to use it)."
else
    echo "Alias 'aimem' already present in your shell config."
fi

# 3. Display usage examples for the CLI
cat <<'USAGE'

✅ Installation complete. Here are some usage examples:
  aimem add "Sample memory content" -i 5.0 -c my_conversation
  aimem list -n 5 --conversation-id my_conversation
  aimem context "What do we know about sample content?" --model gpt-4
  aimem export --output backup_memories.json

To start the API server, run:
  python -m ai_memory.api

USAGE

# 4. (Optional) Print a systemd service template for running API server
cat <<'SYSTEMD'
----- Below is a sample systemd service unit (for reference only) -----
[Unit]
Description=AI Memory API Service
After=network.target

[Service]
ExecStart=$(which python) -m ai_memory.api
WorkingDirectory=$(pwd)
Restart=always
User=$USER
Environment=AI_MEMORY_ROOT=$HOME/ai_memory

[Install]
WantedBy=multi-user.target
-----------------------------------------------------------------------
(Note: Copy the above into a file like /etc/systemd/system/aimemory.service and adjust as needed.)
SYSTEMD
