<p align="center">
  <img src="assets/banner.png" alt="Synapt" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/synapt/"><img src="https://img.shields.io/pypi/v/synapt?color=7c5cbf" alt="PyPI"></a>
  <a href="https://pypi.org/project/synapt/"><img src="https://img.shields.io/pypi/pyversions/synapt?color=00e5cc" alt="Python"></a>
  <a href="https://github.com/laynepenney/synapt/blob/main/LICENSE"><img src="https://img.shields.io/github/license/laynepenney/synapt" alt="License"></a>
</p>

<p align="center">
  Persistent conversational memory for AI coding assistants.<br>
  Indexes your past sessions and makes them searchable — so your AI assistant<br>
  remembers what you worked on, decisions you made, and patterns you established.
</p>

---

Works as an [MCP server](https://modelcontextprotocol.io/) for Claude Code and other MCP-compatible tools.

## Install

```bash
pip install synapt
```

## Quick start

### 1. Build the index

Synapt discovers Claude Code transcripts automatically:

```bash
synapt recall build
```

### 2. Search past sessions

```bash
synapt recall search "how did we fix the auth bug"
```

### 3. Use as an MCP server

Add to your Claude Code config (`~/.claude/mcp.json`):

```json
{
  "mcpServers": {
    "synapt": {
      "type": "stdio",
      "command": "synapt",
      "args": ["server"]
    }
  }
}
```

This gives your AI assistant 13 tools for searching past sessions, managing a journal, setting reminders, and building a durable knowledge base.

## Features

- **Hybrid search** — BM25 full-text search fused with semantic embeddings via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). Surfaces results that keyword search alone would miss.
- **Query intent routing** — Classifies queries as factual, temporal, debug, decision, aggregation, exploratory, or procedural and adjusts search parameters (recency decay, knowledge boost, embedding weight) automatically.
- **Knowledge embeddings** — Durable knowledge nodes get 384-dim embeddings for semantic retrieval, built at index time.
- **Topic clustering** — Jaccard token-overlap clustering groups related chunks across sessions.
- **Session journal** — Rich entries with focus, decisions, done items, and next steps.
- **Reminders** — Cross-session sticky reminders that surface at session start.
- **Timeline** — Chronological work arcs showing project narrative.
- **Working memory** — Frequency-boosted search results for active topics.
- **LLM enrichment** — Optional LLM-powered summaries and cluster upgrades.
- **Knowledge consolidation** — Extracts durable knowledge from session journals.
- **Plugin system** — Extend with additional tools via entry-point discovery.

## MCP tools

| Tool | Description |
|------|-------------|
| `recall_search` | Search past sessions by query |
| `recall_context` | Get context for the current session |
| `recall_files` | Find sessions that touched specific files |
| `recall_sessions` | List indexed sessions |
| `recall_timeline` | View chronological work arcs |
| `recall_build` | Build or rebuild the transcript index |
| `recall_setup` | Auto-configure hooks and MCP integration |
| `recall_stats` | Index statistics |
| `recall_journal` | Write rich session journal entries |
| `recall_remind` | Set cross-session reminders |
| `recall_enrich` | LLM-powered chunk summarization |
| `recall_consolidate` | Extract knowledge from journals |
| `recall_contradict` | Flag contradictions in knowledge |

## CLI reference

```bash
synapt recall build              # Build index (discovers transcripts automatically)
synapt recall build --incremental # Skip already-indexed files
synapt recall search "query"     # Search past sessions
synapt recall stats              # Show index statistics
synapt recall journal --write    # Write a session journal entry
synapt recall setup              # Auto-configure hooks
synapt server                    # Start MCP server
```

## Benchmarks

Evaluated on [LOCOMO](https://snap-research.github.io/locomo/) (Long Conversational Memory) — 10 conversations, 1540 QA pairs — following [Mem0's methodology](https://arxiv.org/abs/2504.19413) (J-score via LLM-as-Judge).

| System | Multi-Hop | Temporal | Single-Hop | Open-Domain | **Overall** | Infra |
|--------|-----------|----------|------------|-------------|-------------|-------|
| **synapt** | **70.21** | **61.68** | **62.50** | **80.14** | **73.38** | **local 3B model** |
| Full-Context | — | — | — | — | 72.90 | upper bound |
| Mem0+Graph | 47.19 | 58.13 | 65.71 | 75.71 | 68.44 | cloud GPT-4 |
| Mem0 | 51.15 | 55.51 | 67.13 | 72.93 | 66.88 | cloud GPT-4 |
| Zep | 41.35 | 49.31 | 61.70 | 76.60 | 65.99 | cloud service |
| LangMem | 47.92 | 23.43 | 62.23 | 71.12 | 58.10 | cloud |
| OpenAI Memory | 42.92 | 21.71 | 63.79 | 62.29 | 52.90 | cloud |

Synapt scores **73.38% overall** — beating the Full-Context upper bound (72.90%), Mem0+Graph (68.44%), Mem0 (66.88%), Zep (65.99%), and all other tested systems — using Ministral 3B locally for enrichment. No cloud API calls.

> **What is Full-Context?** The entire conversation history is passed directly to GPT-4 as context — no retrieval, no memory extraction. It represents the theoretical upper bound: the LLM has access to every fact. Synapt beats it because focused retrieval surfaces only what's relevant, reducing noise for the answer model.

**Best-in-class**: Multi-hop (70.21%) and open-domain (80.14%) — highest of any system tested, including those using GPT-4 for memory extraction.

## How search works

Synapt runs three retrieval paths and merges them:

1. **BM25/FTS5** — Full-text search with configurable recency decay
2. **Embeddings** — Cosine similarity over 384-dim vectors ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2))
3. **Knowledge** — Durable facts extracted from session journals, searched via FTS5 + embeddings with confidence-weighted boosting

Chunk results are merged via **Reciprocal Rank Fusion** (RRF), which combines rankings rather than raw scores. This means a result that BM25 missed entirely can still surface if it's semantically similar to the query. Knowledge nodes are boosted by confidence and entity overlap, then interleaved with chunk results.

Query intent classification adjusts parameters automatically — debug queries weight recent sessions heavily, factual queries prioritize knowledge nodes, temporal queries enable entity-focused search, exploratory queries boost semantic matching.

## Models and dependencies

Synapt uses **two types of models** for different purposes. All models are fetched from HuggingFace on first use and cached locally. No API token is required — all default models are public.

### Search (included by default)

`pip install synapt` installs everything needed for hybrid search:

| Model | Purpose | Size | Library |
|-------|---------|------|---------|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Embedding vectors for semantic search | ~90 MB | sentence-transformers |
| [flan-t5-base](https://huggingface.co/google/flan-t5-base) | Encoder-decoder summarization | ~1 GB | transformers |

These are **encoder models** (not chat LLMs). They run locally on CPU, require no server, and are downloaded to `~/.cache/huggingface/` on first use.

`sentence-transformers` is a default dependency. It transitively installs `transformers` and `torch`, which makes flan-t5-base available for summarization tasks automatically.

### Enrichment (optional LLM backend)

The `recall_enrich` and `recall_consolidate` tools use a **decoder-only chat LLM** to generate journal summaries and extract knowledge nodes. These are optional — core search works without them.

Synapt auto-selects the best available backend:

| Priority | Backend | Model | Install |
|----------|---------|-------|---------|
| 1st | **MLX** (Apple Silicon) | [Ministral-3B-4bit](https://huggingface.co/mlx-community/Ministral-3-3B-Instruct-2512-4bit) (~1.7 GB) | Automatic on Apple Silicon |
| 2nd | **Ollama** | ministral:3b (~1.7 GB) | [ollama.com](https://ollama.com), then `ollama pull ministral:3b` |

On Apple Silicon Macs, `mlx-lm` is installed automatically as a default dependency. It runs in-process with no server — just works. On Linux/Windows, install Ollama as the backend.

If neither is installed, enrichment tools return a message explaining what to install. Search, journal, reminders, and all other features work normally without an LLM backend.

## Plugins

Synapt discovers plugins via Python entry points. To create a plugin:

1. Create a module with a `register_tools(mcp)` function
2. Register it in your `pyproject.toml`:

```toml
[project.entry-points."synapt.plugins"]
my_plugin = "my_package.server"
```

The MCP server automatically discovers and loads plugins at startup.

## Development

```bash
git clone https://github.com/laynepenney/synapt.git
cd synapt
pip install -e ".[test]"
pytest tests/ -v
```

## License

MIT
