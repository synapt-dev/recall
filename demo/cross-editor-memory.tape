# Cross-Editor Memory Demo
# Run: cd /Users/layne/Development/synapt-dev && vhs synapt/demo/cross-editor-memory.tape
Output synapt/demo/cross-editor-memory.gif
Output synapt/demo/cross-editor-memory.mp4

Set FontSize 16
Set Width 1000
Set Height 700
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms
Set Padding 20

# cd to a directory with the recall index so MCP tools work
Type "cd /Users/layne/Development/synapt-dev"
Enter
Sleep 1s

Type "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
Enter
Type "echo '  synapt — persistent memory for AI coding agents'"
Enter
Type "echo '  One memory. Three editors. Same answer.'"
Enter
Type "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
Enter
Sleep 3s

Type "echo ''"
Enter
Type "echo '🟣 Claude Code'"
Enter
Sleep 1s
Type@80ms "claude -p 'use recall to tell me: what embedding model does this project use and why was it chosen? be brief.' --allowedTools 'mcp__synapt__*'"
Enter
Sleep 60s

Type "echo ''"
Enter
Type "echo '🟢 Codex CLI (OpenAI)'"
Enter
Sleep 1s
Type@80ms "codex exec --skip-git-repo-check 'use recall to tell me: what embedding model does this project use and why was it chosen? be brief.'"
Enter
Sleep 60s

Type "echo ''"
Enter
Type "echo '🔵 OpenCode'"
Enter
Sleep 1s
Type@80ms "opencode run 'use recall to tell me: what embedding model does this project use and why was it chosen? be brief.'"
Enter
Sleep 60s

Type "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
Enter
Type "echo '  Same project. Same memory. Any editor.'"
Enter
Type "echo '  pip install synapt    github.com/laynepenney/synapt'"
Enter
Type "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
Enter
Sleep 5s
