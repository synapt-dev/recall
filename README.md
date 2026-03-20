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

<p align="center">
  <a href="https://synapt.dev">Website</a> &middot;
  <a href="https://synapt.dev/blog/">Blog</a> &middot;
  <a href="https://x.com/synapt_dev">@synapt_dev</a>
</p>

---

**#2 on LOCOMO** (76.04%, within 1.5pp of Engram) and **+14.51pp over Mem0** on CodeMemo (90.51% vs 76.0%). Local-first — runs on a laptop, no cloud dependency for memory.

Works as an [MCP server](https://modelcontextprotocol.io/) for Claude Code, Codex CLI, OpenCode, and other MCP-compatible tools.

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

This gives your AI assistant tools for searching past sessions, managing a journal, setting reminders, and building a durable knowledge base.

### Codex CLI

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.synapt]
command = "synapt"
args = ["server"]
```

Synapt automatically discovers and indexes Codex transcripts from `~/.codex/sessions/`.

If you want Codex to re-check `#dev` or continue autonomous work on a timer, the repo includes a simple loop wrapper:

```bash
./scripts/codex-loop.sh \
  --interval 60 \
  --prompt "check #dev, review fresh PRs, or pick up the next unowned task. Post what you're doing in #dev." \
  -- --full-auto
```

This launches a fresh `codex exec` each iteration. It does not wake an already-idle interactive Codex session.

### OpenCode

Add to `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "synapt": {
      "type": "local",
      "command": ["synapt", "server"],
      "enabled": true
    }
  }
}
```

## Features

- **FTS5 + embedding hybrid search** — BM25 full-text search fused with semantic embeddings via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), plus cross-encoder reranking. Surfaces results that keyword search alone would miss.
- **Sub-chunk splitting** — Splits transcripts at tool-use boundaries so each chunk captures a coherent action (code edit, test run, error trace) rather than arbitrary fixed-length windows.
- **Cross-session link expansion** — When retrieving a chunk, automatically surfaces related chunks from other sessions, enabling multi-hop reasoning across your project history.
- **Content-aware adaptive filtering** — Classifies conversations as code/personal/mixed and adjusts consolidation filters and retrieval parameters per content type.
- **Query intent routing** — Classifies queries as factual, temporal, debug, decision, aggregation, exploratory, or procedural and adjusts search parameters (recency decay, knowledge boost, embedding weight) automatically.
- **Enrichment + consolidation + knowledge graph** — Optional LLM-powered session summaries, durable knowledge extraction, and a knowledge graph that connects facts across sessions.
- **Knowledge embeddings** — Durable knowledge nodes get 384-dim embeddings for semantic retrieval, built at index time.
- **Topic clustering** — Jaccard token-overlap clustering groups related chunks across sessions.
- **Session journal** — Rich entries with focus, decisions, done items, and next steps.
- **Reminders** — Cross-session sticky reminders that surface at session start.
- **Timeline** — Chronological work arcs showing project narrative.
- **Working memory** — Frequency-boosted search results for active topics.
- **Local-first** — Runs entirely on your laptop. Indexing, embedding, and retrieval are all local — no cloud dependency for memory.
- **MCP server** — 18 tools for Claude Code integration: search, journal, channels, reminders, portable archive export/import, knowledge, and more.
- **Agent channels** — Cross-session communication via append-only channels. Agents can post messages, send directives, and coordinate work across worktrees.
- **Directive notifications** — Targeted directives are automatically surfaced in MCP tool responses. Broadcast directives (`to="*"`) reach all agents.
- **Contradiction flagging** — Flag conflicting information from free text or search results. Auto-matches existing knowledge nodes via FTS, creates new nodes on resolution.
- **Codex CLI support** — Indexes Codex transcripts from `~/.codex/sessions/` automatically. Cross-editor memory between Claude Code and Codex.
- **Agent-aware consolidation** — Journal entries capture agent identity (griptree, agent_id). Consolidation detects concurrent sessions and annotates them for the LLM.
- **Plugin system** — Extend with additional tools via entry-point discovery.

## MCP tools

| Tool | Description |
|------|-------------|
| `recall_search` | Search past sessions by query |
| `recall_context` | Get context for the current session |
| `recall_files` | Find file history and prior context for a specific path |
| `recall_sessions` | List indexed sessions |
| `recall_timeline` | View chronological work arcs |
| `recall_build` | Build or rebuild the transcript index |
| `recall_setup` | Auto-configure hooks and MCP integration |
| `recall_export` | Export a portable `.synapt-archive` backup |
| `recall_import` | Import a portable recall archive (merge or replace) |
| `recall_stats` | Index statistics |
| `recall_journal` | Write rich session journal entries |
| `recall_remind` | Set cross-session reminders |
| `recall_enrich` | LLM-powered chunk summarization |
| `recall_consolidate` | Extract knowledge from journals |
| `recall_contradict` | Flag contradictions in knowledge (supports free-text claims) |
| `recall_channel` | Cross-session agent communication (post, read, directives, who) |
| `recall_quick` | Fast knowledge check (no transcript chunks) |
| `recall_reload` | Restart MCP server to pick up code changes |

## CLI reference

```bash
synapt recall build              # Build index (discovers transcripts automatically)
synapt recall build --incremental # Skip already-indexed files
synapt recall search "query"     # Search past sessions
synapt recall export backup.synapt-archive  # Create a portable backup
synapt recall import backup.synapt-archive --merge  # Merge a backup into local recall state
synapt recall stats              # Show index statistics
synapt recall journal --write    # Write a session journal entry
synapt recall setup              # Auto-configure hooks
synapt server                    # Start MCP server
```

## Benchmarks

### LOCOMO — Conversational Memory

Evaluated on [LOCOMO](https://snap-research.github.io/locomo/) (Long Conversational Memory) — 10 conversations, 1540 QA pairs — following [Mem0's methodology](https://arxiv.org/abs/2504.19413) (J-score via LLM-as-Judge). Competitor data from the [Mem0 paper](https://arxiv.org/abs/2504.19413) and [Memobase benchmark](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md).

All systems use gpt-4o-mini as shared backbone (generation + judge) for fair comparison. Competitor data from the [Engram paper](https://arxiv.org/abs/2511.12960) (3 runs, stddev reported) and [Mem0 paper](https://arxiv.org/abs/2504.19413).

| System | **Overall** | Multi-Hop | Temporal | Infra |
|--------|-------------|-----------|----------|-------|
| Engram | 77.55 ± 0.13 | — | — | cloud (BM25+ColBERT+KG) |
| **synapt v0.6.1 (8B)** | **76.04** | **70.92** | 66.36 | Ministral 8B cloud enrich |
| Memobase | 75.78 | 46.88 | **85.05** | cloud |
| **synapt v0.6.1 (3B)** | **73.38** | 70.21 | 61.68 | **local 3B on M2 Air** |
| memOS | 72.99 ± 0.14 | — | — | cloud |
| Full-Context | 72.90 | — | — | upper bound |
| Mem0 | 64.73 ± 0.17 | 51.15 | 55.51 | cloud GPT-4 |
| Zep | 42.29 ± 0.18 | — | — | cloud |

Synapt is **#2 on LOCOMO** at 76.04% — 1.51pp behind Engram (within their stddev of ±0.13) and ahead of Memobase (75.78%), the Full-Context upper bound (72.90%), and all other systems. The 3B local configuration (73.38%) beats the Full-Context upper bound using only a Ministral 3B model running on an M2 MacBook Air.

**Best-in-class multi-hop**: 70.92% — highest of any system tested, including those using GPT-4 for memory extraction. Engram is cloud-only; synapt runs entirely on a laptop.

> **What is Full-Context?** The entire conversation history is passed directly to the LLM as context — no retrieval, no memory extraction. It represents the theoretical upper bound: the LLM has access to every fact. Synapt beats it because focused retrieval surfaces only what's relevant, reducing noise for the answer model.

### CodeMemo — Coding Memory

First benchmark specifically testing coding session memory — 158 questions across 3 projects, 6 categories. Same gpt-4o-mini judge and answer model for both systems.

| System | Factual | Debug | Architecture | Temporal | Convention | Cross-Session | **Overall** |
|--------|---------|-------|-------------|----------|------------|---------------|-------------|
| **synapt v0.6.2** | **97.14** | **100.0** | 92.86 | **90.91** | **80.0** | **86.36** | **90.51** |
| Mem0 (OSS) | 72.73 | 77.78 | **100.0** | 87.50 | 42.86 | 71.43 | 76.0 |

Synapt leads by **+14.51pp overall**. The biggest gaps are in convention (+37pp), factual (+24pp), and debug (+22pp) — categories that depend on raw evidence preservation. Synapt runs entirely locally; Mem0 requires OpenAI API calls for memory extraction, embedding, and search.

## How search works

Synapt runs three retrieval paths and merges them:

1. **BM25/FTS5** — Full-text search with configurable recency decay
2. **Embeddings** — Cosine similarity over 384-dim vectors ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2))
3. **Knowledge** — Durable facts extracted from session journals, searched via FTS5 + embeddings with confidence-weighted boosting

Chunk results are merged via **Reciprocal Rank Fusion** (RRF), which combines rankings rather than raw scores. This means a result that BM25 missed entirely can still surface if it's semantically similar to the query. Knowledge nodes are boosted by confidence and entity overlap, then interleaved with chunk results.

Query intent classification adjusts parameters automatically — debug queries weight recent sessions heavily, factual queries prioritize knowledge nodes, temporal queries enable entity-focused search, exploratory queries boost semantic matching.

**Why knowledge nodes matter:** During a project, your assistant might discuss multiple options across sessions — approach A in session 3, approach B in session 5, then settle on B-with-modifications in session 7. Raw transcripts contain all three discussions equally. The knowledge layer extracts the final decision as a durable fact, so when you search "what did we decide?", the decision surfaces first — not the earlier deliberation that was superseded.

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

## Reproducible Evals

Long-running evals should run from an isolated git worktree with a dedicated
venv so code changes on `main` do not mutate the run mid-flight.

```bash
./scripts/eval-worktree.sh              # from HEAD
./scripts/eval-worktree.sh abc1234      # from a specific commit/tag

source /tmp/synapt-eval-<ref>/.venv/bin/activate
cd /tmp/synapt-eval-<ref>
python -m evaluation.codememo.eval --recalldb --model gpt-4o-mini
python -m evaluation.locomo_eval --recalldb --batch
```

The helper creates:

- a detached worktree at `/tmp/synapt-eval-<ref>`
- a dedicated venv at `/tmp/synapt-eval-<ref>/.venv`
- a non-editable install frozen to that ref

Clean up with:

```bash
./scripts/eval-worktree.sh --cleanup <ref>
```

## Links

- [synapt.dev](https://synapt.dev) — Website
- [synapt.dev/blog](https://synapt.dev/blog/) — Blog
- [@synapt_dev](https://x.com/synapt_dev) — X / Twitter
- [PyPI](https://pypi.org/project/synapt/) — `pip install synapt`

## License

MIT
