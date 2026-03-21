# Cross-Editor Memory Demo
# Shows synapt recall working across Claude Code, Codex CLI, and OpenCode
#
# Prerequisites:
#   1. Build demo project + index: python3 demo/build_demo_index.py
#      Fast local check: python3 demo/build_demo_index.py --no-embeddings
#   2. Run: vhs demo/cross-editor-memory.tape
#
# Uses CodeMemo project_03 (synapt development sessions) as the data source.

Output demo/cross-editor-memory.gif
Output demo/cross-editor-memory.mp4

Set FontSize 16
Set Width 1000
Set Height 700
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms
Set Padding 20

# Start in demo project directory so synapt server finds the index
Type "cd /tmp/synapt-demo"
Enter
Sleep 500ms

# Title
Type "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
Enter
Type "echo '  synapt — keep your memory'"
Enter
Type "echo '  One memory. Three editors.'"
Enter
Type "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
Enter
Sleep 2s

# === Claude Code ===
Type "echo ''"
Enter
Type "echo '🟣 Claude Code'"
Enter
Sleep 1s
Type@80ms "claude -p 'use recall to tell me: what embedding model does this project use and why was it chosen? be brief.'"
Enter
Sleep 25s

# === Codex CLI ===
Type "echo ''"
Enter
Type "echo '🟢 Codex CLI (OpenAI)'"
Enter
Sleep 1s
Type@80ms "codex exec 'use recall to tell me: what embedding model does this project use and why was it chosen? be brief.'"
Enter
Sleep 25s

# === OpenCode ===
Type "echo ''"
Enter
Type "echo '🔵 OpenCode'"
Enter
Sleep 1s
Type@80ms "opencode run 'use recall to tell me: what embedding model does this project use and why was it chosen? be brief.'"
Enter
Sleep 25s

# Closing
Type "echo ''"
Enter
Type "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
Enter
Type "echo '  Same project. Same memory. Any editor.'"
Enter
Type "echo '  pip install synapt    synapt.dev'"
Enter
Type "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
Enter
Sleep 4s
