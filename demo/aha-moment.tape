# The Aha Moment
# "My agent forgot everything. One command brought it all back."
#
# Run: vhs demo/aha-moment.tape

Output demo/aha-moment.gif
Output demo/aha-moment.mp4

Set FontSize 16
Set Width 900
Set Height 500
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms
Set Padding 24

# Scene 1: The problem
Type "# New Claude Code session. Fresh context. Agent knows nothing."
Enter
Sleep 2s

Type "# But I need to know: what did we decide about auth last week?"
Enter
Sleep 2s

# Scene 2: The search
Type@30ms "synapt recall search 'auth decision'"
Enter
Sleep 15s

# Scene 3: The journal
Type ""
Enter
Type "# What about what's still pending?"
Enter
Sleep 1s
Type@30ms "synapt recall journal --read"
Enter
Sleep 15s

# Scene 4: The punchline
Type ""
Enter
Type "# 48,700 chunks. 156 sessions. Searchable in <30ms."
Enter
Sleep 1s
Type@30ms "synapt recall stats"
Enter
Sleep 10s

# CTA
Type ""
Enter
Type "# pip install synapt && claude mcp add synapt -- synapt server"
Enter
Sleep 4s
