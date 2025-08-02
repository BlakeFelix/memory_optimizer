#!/usr/bin/env bash
# Robust Luna CLI: joins all args, or reads stdin if '-'

source ~/aimemory_cpu/bin/activate

if [[ "$1" == "-" ]]; then
  prompt="$(cat -)"
elif [[ $# -eq 0 ]]; then
  echo "Usage: luna <prompt>  |  echo 'multi line' | luna -" >&2
  exit 1
else
  prompt="$*"
fi

python ~/memory_optimizer/ai_memory/chat_with_luna.py "$prompt"

