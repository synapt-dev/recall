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

Add to your Claude Code config (`.mcp.json`):

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
- **Query intent routing** — Classifies queries as factual, debug, exploratory, or procedural and adjusts search parameters (recency decay, knowledge boost, embedding weight) automatically.
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

## How search works

Synapt runs two retrieval paths in parallel and merges them:

1. **BM25** — Full-text search with recency decay over session chunks
2. **Embeddings** — Cosine similarity over 384-dim vectors ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2))

Results are merged via **Reciprocal Rank Fusion** (RRF), which combines rankings rather than raw scores. This means a result that BM25 missed entirely can still surface if it's semantically similar to the query.

Query intent classification then adjusts parameters — debug queries weight recent sessions heavily, factual queries prioritize knowledge nodes, exploratory queries boost semantic matching.

## Optional backends

Synapt uses local LLMs for enrichment and summarization. Install a backend if you want LLM-powered features (`recall_enrich`, `recall_consolidate`):

```bash
# Ollama (recommended)
# Install from https://ollama.com, then:
ollama pull ministral:3b

# MLX (Apple Silicon)
pip install mlx-lm
```

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
