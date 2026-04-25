# MCP Registry Listing Prep

Prep material for [recall#761](https://github.com/synapt-dev/recall/issues/761).

This document is intentionally submission-oriented. The goal is to make the
official Anthropic registry handoff and community directory listings a copy/paste
operation rather than an open-ended writing task.

## Canonical server identity

- Name: `synapt`
- Package: `synapt`
- Repository: `https://github.com/synapt-dev/recall`
- Website: `https://synapt.dev`
- Install: `pip install synapt`
- Run command: `synapt server`
- Protocol: stdio MCP server
- License: MIT

## Short description

Persistent memory and coordination for AI coding assistants. `synapt` gives
Claude Code, Codex CLI, Cursor, Windsurf, and other MCP-compatible clients
search, journals, reminders, channels, directives, and durable knowledge nodes
from one local memory system.

## One-line tagline

Persistent memory and coordination MCP server for AI coding assistants.

## Core capabilities

- Search prior sessions and decisions with `recall_search`
- Save durable knowledge and conventions with `recall_save`
- Journal sprint focus, done items, and next steps with `recall_journal`
- Coordinate teams through append-only channels, unread counts, directives, and claims
- Index Claude Code and Codex transcripts into one shared local memory store

## Supported clients

- Claude Code
- Codex CLI
- Cursor
- Windsurf
- OpenCode
- Any MCP client that supports stdio server definitions

## Install snippets

### Claude Code

```bash
pip install synapt
claude mcp add synapt -- synapt server
synapt init
```

### Codex CLI

```toml
[mcp_servers.synapt]
command = "synapt"
args = ["server"]
```

### Cursor

Project-local `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "synapt": {
      "command": "synapt",
      "args": ["server"]
    }
  }
}
```

### Windsurf

`~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "synapt": {
      "command": "synapt",
      "args": ["server"]
    }
  }
}
```

## Anthropic MCP registry handoff draft

Use this as the maintainer-facing handoff for Layne rather than submitting it
automatically from this branch.

### Summary

`synapt` is a local-first MCP server for persistent memory and team
coordination. It gives Claude Code durable recall across sessions: hybrid search,
journals, reminders, channels, directives, and knowledge nodes from a local
SQLite-backed memory store.

### Why it belongs in the registry

- It is installable from PyPI with one command.
- It already works in Claude Code via stdio MCP.
- It is open source and MIT-licensed.
- It is not a toy demo server; it is used as the operational memory layer for
  multi-agent software work.

### Submission payload

- Server name: `synapt`
- Repository URL: `https://github.com/synapt-dev/recall`
- Package/install URL: `https://pypi.org/project/synapt/`
- Install command: `pip install synapt`
- Start command: `synapt server`
- Transport: stdio
- License: MIT
- Category: memory / productivity / developer tools
- Description:
  `Persistent memory and coordination for AI coding assistants. Search prior sessions, save durable knowledge, journal progress, and coordinate agents through channels and directives from one local MCP server.`

## awesome-mcp-servers draft

Suggested entry:

```md
- [synapt](https://github.com/synapt-dev/recall) - Persistent memory and coordination MCP server for AI coding assistants. Search prior sessions, save durable knowledge, journal progress, and coordinate multi-agent work through shared channels and directives. Install with `pip install synapt`, then run via `synapt server`.
```

## Directory-review checklist

Before final submission:

- confirm the package version on PyPI matches current README instructions
- confirm install command still works in a clean venv
- confirm Cursor and Windsurf config snippets still match current client docs
- confirm the repository description and homepage are current
- have Layne submit the official Anthropic registry entry from the canonical maintainer path
