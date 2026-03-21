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
builds a real synapt recall index there. The `--no-embeddings` path skips the
heavier cluster/LLM upgrade work so setup stays laptop-friendly; use the
default full build before a final tape render.

Probe the three editor CLIs before a full render:

```bash
python demo/run_cross_editor_probe.py
```

That runs bounded smoke checks for `claude`, `codex`, and `opencode` in the
demo project so obvious auth, trust, or startup problems fail fast.

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

- The tape uses `demo/run_cross_editor_query.sh` inside `/tmp/synapt-demo` so
  the repo controls the exact per-editor flags. On this machine that means:
  Claude runs in bare mode with a synapt-only MCP config, Codex disables
  unrelated MCP servers, and OpenCode uses `openai/gpt-5.4-mini`.
- The query is about the default consolidation model, not the embedding model.
  That prompt hit the right memory path consistently in all three editors,
  while the earlier embedding-model wording was ambiguous.
- If a probe times out, increase `--timeout` before rendering instead of
  blindly running VHS.
