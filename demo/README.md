# Demo Assets

This directory holds reproducible terminal demos for synapt.

## Cross-editor memory demo

Build the temp project and its recall index:

```bash
python demo/build_demo_index.py
```

For a faster local smoke check:

```bash
python demo/build_demo_index.py --no-embeddings
```

That creates `/tmp/synapt-demo` from CodeMemo `project_03_memory_system` and
builds a real synapt recall index there.

Probe the three editor CLIs before a full render:

```bash
python demo/run_cross_editor_probe.py
```

That runs bounded smoke checks for `claude`, `codex`, and `opencode` in the
demo project so obvious auth, trust, or startup problems fail fast. The Codex
probe includes `--skip-git-repo-check` because `/tmp/synapt-demo` is outside a
git worktree.

Then render the tape:

```bash
vhs demo/cross-editor-memory.tape
```

Outputs:

- `demo/cross-editor-memory.gif`
- `demo/cross-editor-memory.mp4`

Prerequisites:

- `vhs`
- `ffmpeg`
- local access to the CodeMemo dataset via `evaluation/codememo/eval.py`
- editor CLIs used in the tape (`claude`, `codex`, `opencode`)

Notes:

- The tape uses a tighter `recall_quick` prompt and longer sleeps than the
  first draft because the live editor CLIs can take close to a minute to finish
  a bounded memory query.
- If a probe times out, increase `--timeout` before rendering instead of
  blindly running VHS.
