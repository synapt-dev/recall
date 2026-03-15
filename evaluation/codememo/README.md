# CodeMemo Benchmark

A coding-assistant memory benchmark for evaluating retrieval and recall over
multi-session development histories.  Think of it as LOCOMO for coding sessions:
the system ingests a series of Claude Code transcripts (JSONL) from a realistic
software project and must answer questions that require remembering decisions,
debugging steps, conventions, and temporal ordering across sessions.

## Categories

| Cat | Name           | Description                                                  |
|-----|----------------|--------------------------------------------------------------|
| 1   | Factual        | Single-fact recall (library version, config value, file path)|
| 2   | Debug          | Remembering how a bug was diagnosed and fixed                |
| 3   | Architecture   | Design decisions, trade-offs, component relationships        |
| 4   | Temporal        | When something happened or ordering of events                |
| 5   | Convention     | Project-specific patterns, style rules, naming conventions   |
| 6   | Cross-session  | Connecting information scattered across multiple sessions    |

## Data layout

```
evaluation/codememo/data/
  project_01_cli_tool/
    manifest.json          # project metadata
    questions.json         # QA pairs with evidence links
    sessions/              # Claude Code JSONL transcripts (one per session)
      session_001.jsonl
      session_002.jsonl
      ...
```

## Running the eval

```bash
# Full eval (requires OPENAI_API_KEY):
python evaluation/codememo/eval.py --project project_01_cli_tool

# RecallDB mode (FTS5 + embeddings, no MLX):
python evaluation/codememo/eval.py --project project_01_cli_tool --recalldb

# Full pipeline (RecallDB + enrich + consolidate + knowledge graph):
python evaluation/codememo/eval.py --project project_01_cli_tool --full-pipeline

# Retrieval-only (no API key needed):
python evaluation/codememo/eval.py --project project_01_cli_tool --retrieval-only

# All projects:
python evaluation/codememo/eval.py

# Limit knowledge nodes:
python evaluation/codememo/eval.py --max-knowledge 5
```

## Competitor comparison

Run the same benchmark against Mem0 or Memobase to compare apples-to-apples.
Same questions, same judge model (gpt-4o-mini), same scoring — only the memory
system differs.

### Mem0

```bash
# Install mem0 open-source
pip install mem0ai

# Requires OPENAI_API_KEY (mem0 uses OpenAI for fact extraction + embeddings)
export OPENAI_API_KEY=sk-...

# Run on all projects:
python -m evaluation.codememo.competitor_eval --system mem0

# Single project:
python -m evaluation.codememo.competitor_eval --system mem0 --project project_01_cli_tool

# Retrieval-only (skip answer generation / judging):
python -m evaluation.codememo.competitor_eval --system mem0 --retrieval-only

# Results saved to evaluation/codememo/results/mem0/
```

Mem0 uses local Qdrant for vector storage (no cloud signup needed) but calls
OpenAI for LLM-based fact extraction (`infer=True`) and embeddings
(`text-embedding-3-small`). Each project gets a fresh Qdrant collection.

### Memobase

Requires a running Memobase server. See
[Memobase server setup](https://github.com/memodb-io/memobase) for Docker instructions.

```bash
pip install memobase

export MEMOBASE_URL=http://localhost:8019
export MEMOBASE_API_KEY=your-token

python -m evaluation.codememo.competitor_eval --system memobase
```

### synapt (for comparison)

```bash
# Install synapt
pip install -e .  # from synapt repo root

# RecallDB mode (local 3B model, no cloud API for memory):
python -m evaluation.codememo.eval --recalldb

# Full pipeline (local 3B enrichment + consolidation):
python -m evaluation.codememo.eval --recalldb --full-pipeline

# Requires OPENAI_API_KEY only for answer generation + judging (gpt-4o-mini)
```

synapt's indexing, embedding, and retrieval are fully local — runs on a laptop
with no cloud dependencies. The only API calls are for answer generation and
judging, which are identical across all systems in this benchmark.

### Published results (2026-03-14)

| System | Factual | Debug | Architecture | Temporal | Convention | Cross-Session | **Overall** |
|--------|---------|-------|-------------|----------|------------|---------------|-------------|
| **synapt v0.5.2** (local 3B) | **97.14** | **100.0** | 92.86 | **90.91** | **80.0** | **86.36** | **90.51** |
| Mem0 (OSS + OpenAI) | 72.73 | 77.78 | **100.0** | 87.50 | 42.86 | 71.43 | **76.0** |

## Pluggable systems

The eval defines a `SystemUnderTest` protocol (`eval.py`). To benchmark a
different memory system, implement `ingest(session_paths)` and `query(question)`
and pass it to `run_evaluation()`. See `competitor_eval.py` for examples.

```python
class SystemUnderTest(Protocol):
    def ingest(self, session_paths: list[Path]) -> None:
        """Ingest session transcripts into the memory system."""
        ...

    def query(self, question: str, max_chunks: int = 20) -> str:
        """Retrieve context for a question. Returns formatted text."""
        ...

    def stats(self) -> dict:
        """Return {"chunk_count": int, "knowledge_count": int}."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...
```

## Metrics

- **J-score**: LLM-as-judge accuracy (CORRECT/WRONG via gpt-4o-mini), per category and overall.
- **F1**: Token-level F1 between generated and gold answers.
- **Retrieval recall@k**: Fraction of evidence turns that appear in retrieved context.
