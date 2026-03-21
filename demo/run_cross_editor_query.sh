#!/usr/bin/env bash
set -euo pipefail

tool="${1:-}"
if [[ -z "$tool" ]]; then
  echo "usage: $0 <claude|codex|opencode>" >&2
  exit 1
fi

prompt="Use recall_quick only. Call it with exactly this query: 'default consolidation model used by this project and why it was chosen'. Then answer in one short line using only that result."

case "$tool" in
  claude)
    exec claude -p \
      --bare \
      --strict-mcp-config \
      --mcp-config '{"mcpServers":{"synapt":{"command":"synapt","args":["server"]}}}' \
      --dangerously-skip-permissions \
      --allowedTools mcp__synapt__recall_quick \
      -- "$prompt"
    ;;
  codex)
    exec codex exec \
      --skip-git-repo-check \
      -m gpt-5.4-mini \
      -c 'mcp_servers.gitgrip.enabled=false' \
      -c 'mcp_servers.supabase.enabled=false' \
      -c 'model_reasoning_effort="low"' \
      "$prompt"
    ;;
  opencode)
    exec opencode run \
      -m openai/gpt-5.4-mini \
      "$prompt"
    ;;
  *)
    echo "unknown tool: $tool" >&2
    exit 1
    ;;
esac
