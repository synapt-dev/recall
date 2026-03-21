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

Then render the tape:

```bash
vhs demo/cross-editor-memory.tape
```

Outputs:

- `demo/cross-editor-memory.gif`
- `demo/cross-editor-memory.mp4`

Prerequisites:

- `vhs`
- local access to the CodeMemo dataset via `evaluation/codememo/eval.py`
- editor CLIs used in the tape (`claude`, `codex`, `opencode`)
