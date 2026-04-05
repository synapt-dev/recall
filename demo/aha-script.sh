#!/bin/bash
# Scripted demo for asciinema recording
# Shows recall finding past decisions instantly

slowtype() {
    local text="$1"
    for ((i=0; i<${#text}; i++)); do
        printf '%s' "${text:$i:1}"
        sleep 0.04
    done
    echo
}

clear
echo ""
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  synapt recall — persistent memory for AI agents"
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
sleep 2

echo "  # New session. Agent has no context. But recall does."
echo ""
sleep 2

echo -n "  \$ "
slowtype "python3 -c \"from synapt.recall.server import recall_quick; print(recall_quick('what did we decide about the benchmark score'))\""
echo ""

# Run the actual command
python3 -c "
from synapt.recall.server import recall_quick
result = recall_quick('what did we decide about the benchmark score')
# Indent and truncate for clean display
for line in result.split('\n')[:15]:
    print(f'  {line}')
if len(result.split('\n')) > 15:
    print(f'  ... ({len(result.split(chr(10)))} lines total)')
" 2>/dev/null

echo ""
sleep 4

echo "  # What's pending from the last session?"
echo ""
sleep 1

echo -n "  \$ "
slowtype "python3 -c \"from synapt.recall.server import recall_journal; print(recall_journal(action='read'))\""
echo ""

python3 -c "
from synapt.recall.server import recall_journal
result = recall_journal(action='read')
for line in result.split('\n')[:12]:
    print(f'  {line}')
if len(result.split('\n')) > 12:
    print(f'  ... ({len(result.split(chr(10)))} lines total)')
" 2>/dev/null

echo ""
sleep 4

echo "  # 48,700 chunks. 156 sessions. Sub-second via MCP."
echo ""
sleep 2

echo "  pip install synapt"
echo "  claude mcp add synapt -- synapt server"
echo ""
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  github.com/synapt-dev/synapt"
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sleep 4
