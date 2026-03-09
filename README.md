# synapt

Persistent conversational memory for AI coding assistants. Synapt indexes your past coding sessions and makes them searchable — so your AI assistant remembers what you worked on, decisions you made, and patterns you established.

Works as an [MCP server](https://modelcontextprotocol.io/) for Claude Code and other MCP-compatible tools.

## Install

```bash
pip install synapt

# With MCP server support (recommended)
pip install 'synapt[mcp]'
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

- **Transcript indexing** — BM25 full-text search over past coding sessions
- **Topic clustering** — Jaccard token-overlap clustering groups related chunks
- **Knowledge consolidation** — Extracts durable knowledge from session journals
- **Session journal** — Rich entries with focus, decisions, done items, and next steps
- **Reminders** — Cross-session sticky reminders that surface at session start
- **Timeline** — Chronological work arcs showing project narrative
- **LLM enrichment** — Optional LLM-powered summaries and cluster upgrades
- **Working memory** — Frequency-boosted search results for active topics
- **Plugin system** — Extend with additional tools via entry-point discovery

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

## Optional backends

Synapt uses local LLMs for enrichment and summarization. Install optional backends:

```bash
# MLX (Apple Silicon)
pip install mlx-lm

# Ollama
# Install from https://ollama.com, then:
ollama pull qwen2.5:3b

# Transformers (GPU/CPU)
pip install 'synapt[transformers]'
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
