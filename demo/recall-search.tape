# Recall Search Demo
# Shows synapt recall search finding past session context
#
# Run: vhs demo/recall-search.tape

Output demo/recall-search.gif
Output demo/recall-search.mp4

Set FontSize 15
Set Width 1000
Set Height 600
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set Padding 20

# Title
Type "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
Enter
Type "echo '  synapt recall — persistent memory for AI'"
Enter
Type "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
Enter
Sleep 2s

# Stats
Type "echo ''"
Enter
Type@50ms "echo '📊 What does recall know?'"
Enter
Sleep 500ms
Type@50ms "synapt recall stats"
Enter
Sleep 3s

# Search
Type "echo ''"
Enter
Type@50ms "echo '🔍 Searching past sessions...'"
Enter
Sleep 500ms
Type@50ms "synapt recall search 'dedup regression root cause'"
Enter
Sleep 5s

# Journal
Type "echo ''"
Enter
Type@50ms "echo '📓 Last session journal:'"
Enter
Sleep 500ms
Type@50ms "synapt recall journal --read"
Enter
Sleep 4s

# Quick check
Type "echo ''"
Enter
Type@50ms "echo '⚡ Quick knowledge check:'"
Enter
Sleep 500ms
Type@50ms "synapt recall search --depth summary 'what benchmark scores do we have'"
Enter
Sleep 5s

# Closing
Type "echo ''"
Enter
Type "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
Enter
Type "echo '  pip install synapt    github.com/synapt-dev/synapt'"
Enter
Type "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
Enter
Sleep 4s
