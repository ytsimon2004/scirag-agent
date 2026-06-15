Output docs/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 700
Set Theme "Catppuccin Mocha"
Set Padding 24
Set TypingSpeed 50ms
Set PlaybackSpeed 1.25

# ── Launch ──────────────────────────────────────────────────────────────────
Type "scirag"
Enter
Sleep 5s

# ── Index ──────────────────────────────────────────────────────────────────
Type `/index "anterior and posterior differences in rodent retrosplenial cortex" --retmax 5`
Enter
Sleep 10s

# ── Select first two papers ──────────────────────────────────────────────────
# All 5 papers start pre-checked; invert to clear all, then pick #1 and #2
Type "i"
Sleep 300ms
Space
Sleep 300ms
Down
Sleep 300ms
Space
Sleep 300ms
Enter
Sleep 10s

# ── Status ───────────────────────────────────────────────────────────────────────
Type "/status"
Enter
Sleep 3s

# ── Selection ───────────────────────────────────────────────────────────────────────
Type "/model"
Enter
Sleep 2s
Down
Sleep 300ms
Down
Sleep 300ms
Down
Sleep 300ms
Enter
Sleep 3s


# ── LLM ───────────────────────────────────────────────────────────────────────
Type `/llm "What distinguishes anterior from posterior RSC?"`
Enter
Sleep 20s
