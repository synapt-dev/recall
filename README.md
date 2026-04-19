<p align="center">
  <img src="assets/banner.png" alt="Synapt" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/synapt/"><img src="https://img.shields.io/pypi/v/synapt?color=7c5cbf" alt="PyPI"></a>
  <a href="https://pypi.org/project/synapt/"><img src="https://img.shields.io/pypi/pyversions/synapt?color=00e5cc" alt="Python"></a>
  <a href="https://github.com/synapt-dev/recall/blob/main/LICENSE"><img src="https://img.shields.io/github/license/synapt-dev/recall" alt="License"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LangChain-ready-1C3C3C?logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/CrewAI-ready-FF6B35?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0id2hpdGUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iOCIgY3k9IjgiIHI9IjYiLz48L3N2Zz4=&logoColor=white" alt="CrewAI">
  <img src="https://img.shields.io/badge/OpenAI_Agents-ready-412991?logo=openai&logoColor=white" alt="OpenAI Agents">
  <img src="https://img.shields.io/badge/Claude_Code-ready-D97706?logo=anthropic&logoColor=white" alt="Claude Code">
  <img src="https://img.shields.io/badge/Codex_CLI-ready-412991?logo=openai&logoColor=white" alt="Codex CLI">
</p>

<p align="center">
  Memory and coordination infrastructure for AI coding teams.<br>
  Search past sessions, preserve decisions, coordinate agents, and ship from one persistent system of record.
</p>

<p align="center">
  <a href="https://synapt.dev">Website</a> &middot;
  <a href="https://synapt.dev/guide.html">Guide</a> &middot;
  <a href="https://synapt.dev/blog/">Blog</a> &middot;
  <a href="https://x.com/synapt_dev">@synapt_dev</a>
</p>

---

**synapt** gives Claude Code, Codex CLI, OpenCode, and other MCP-compatible assistants persistent operational memory.

It closes the gap between a one-shot assistant and a real working team:
- recall prior sessions, file history, decisions, and unresolved work
- preserve context in journals, reminders, and knowledge nodes
- coordinate multiple agents through shared channels, directives, and task claims
- scale from solo recall on a laptop to multi-agent operational memory

Agent skill files for repository-native use live in:
- `.codex/skills/synapt/SKILL.md`
- `.claude/skills/synapt/SKILL.md`

## Three-command quickstart

For the default Claude Code path:

```bash
pip install synapt
claude mcp add synapt -- synapt server
synapt init
```

That gives you:
- a project-local `.synapt/` memory store
- indexed Claude Code and Codex transcripts
- Claude hooks for automatic archive/build flow
- the published Codex `dev-loop` skill installed into `${CODEX_HOME:-~/.codex}/skills/dev-loop/`

## Platform setup

### Claude Code

Recommended:

```bash
pip install synapt
claude mcp add synapt -- synapt server
synapt init
```

### Codex CLI

Install:

```bash
pip install synapt
```

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.synapt]
command = "synapt"
args = ["server"]
```

Then initialize the project:

```bash
synapt init
```

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

Then run:

```bash
synapt init
```

### Manual MCP config

If your client accepts stdio MCP definitions directly, use:

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

## Framework integrations

synapt plugs into popular agent frameworks as a drop-in memory backend. Each adapter wraps recall's search and save API for the framework's native session interface.

### LangChain

`SynaptChatMessageHistory` implements LangChain's `BaseChatMessageHistory`. Messages are stored in SQLite with WAL mode; recall search and knowledge persistence are one method call away.

```bash
pip install synapt langchain-core
```

```python
from synapt.integrations.langchain import SynaptChatMessageHistory

history = SynaptChatMessageHistory(session_id="user-123")
history.add_messages([HumanMessage(content="hello")])
print(history.messages)

# Semantic search across all indexed sessions
results = history.search("deployment config")

# Persist a decision as a durable knowledge node
history.save_to_recall("Always use UTC timestamps", category="convention")
```

Works with `RunnableWithMessageHistory`, `ConversationChain`, and any LangChain component that accepts a `BaseChatMessageHistory`.

### OpenAI Agents SDK

`SynaptSession` implements the Agents SDK `Session` protocol. Items are stored as JSON dicts in SQLite; async throughout.

```bash
pip install synapt openai-agents
```

```python
from synapt.integrations.openai_agents import SynaptSession

session = SynaptSession(session_id="agent-run-42")
await session.add_items([{"role": "user", "content": "hello"}])
items = await session.get_items(limit=10)

# Bridge to recall search
results = session.search("prior error handling decisions")
```

### CrewAI

`SynaptMemory` provides long-term memory storage for CrewAI crews, backed by recall's hybrid search.

```bash
pip install synapt crewai
```

```python
from synapt.integrations.crewai import SynaptMemory

memory = SynaptMemory()
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    memory=memory,
)
crew.kickoff()
```

### Claude Code (plugin)

Install the synapt plugin to give Claude Code persistent memory across sessions. The plugin auto-starts the recall MCP server and installs recall skills.

```bash
claude plugin add synapt-recall
```

Once installed, Claude Code gains `recall_search`, `recall_save`, `recall_journal`, and 20+ other recall tools automatically. No manual MCP configuration needed.

### Codex CLI

Codex CLI connects to synapt via MCP. Add to `~/.codex/config.toml`:

```toml
[mcp_servers.synapt]
command = "synapt"
args = ["server"]
```

Then initialize:

```bash
pip install synapt
synapt init
```

The `dev-loop` skill is installed automatically, giving Codex recall search, channel coordination, and journal access.

## What `synapt init` does

Run from a project root:

```bash
synapt init
```

It will:
1. archive project-relevant Claude Code and Codex transcripts
2. build the `.synapt/recall/index/recall.db` search index
3. register the Synapt MCP server in Claude Code when the `claude` CLI is available
4. install Claude hooks for `SessionStart`, `SessionEnd`, and `PreCompact`
5. deploy the packaged Codex `dev-loop` skill
6. add `.synapt/` to `.gitignore`

`synapt recall setup` remains available as the explicit recall-scoped equivalent.

## Product tiers

synapt is one memory system with a clear adoption ladder:

### 1. Solo recall

Search prior sessions, file history, timelines, and journals on one machine.

```bash
synapt recall search "how did we fix auth"
synapt recall files "src/auth.py"
synapt recall timeline
```

### 2. Multi-agent memory

Add channels, directives, reminders, and task claims for coordinated execution across worktrees and agents.

```python
recall_channel(action="join", channel="dev", name="Atlas")
recall_channel(action="intent", channel="dev", message="reviewing PR #403")
recall_channel(action="claim", channel="dev", message="m_abc123")
```

### 3. Dashboard

Expose the same shared operational memory in a browser-facing mission-control surface.

### 4. Spawn / orchestration

Use synapt as the memory and coordination substrate beneath higher-level agent orchestration.

## Core features

- **Hybrid search**: BM25 + embeddings + reciprocal rank fusion + reranking
- **File-aware recall**: find where a file, bug, issue, or decision was handled before
- **Journal + knowledge**: durable summaries, extracted facts, contradictions, and timeline arcs
- **Agent channels**: shared append-only coordination across sessions and worktrees
- **Cross-client memory**: Claude Code and Codex transcripts converge into one searchable system
- **Portable archive**: export/import `.synapt-archive` state between machines
- **Plugin system**: extend MCP tools and CLI commands through Python entry points

## Benchmarks

### LOCOMO

LOCOMO evaluates long conversational memory over 10 conversations and 1540 QA pairs.

All systems use gpt-4o-mini as the shared generation + judge backbone for fair comparison. Competitor data comes from the [Engram paper](https://arxiv.org/abs/2511.12960) and [Mem0 paper](https://arxiv.org/abs/2504.19413).

| System | **Overall** | Multi-Hop | Temporal | Infra |
|--------|-------------|-----------|----------|-------|
| Engram | 77.55 ± 0.13 | — | — | cloud (BM25+ColBERT+KG) |
| Memobase | 75.78 | 46.88 | **85.05** | cloud |
| memOS | 72.99 ± 0.14 | — | — | cloud |
| Full-Context | 72.90 | — | — | upper bound |
| **synapt (audited)** | **72.4** | **70.92** | 59.19 | Ministral 8B cloud enrich |
| **synapt local-first** | **72.4** | 67.02 | 61.06 | **local 3B on M2 Air** |
| Mem0 | 64.73 ± 0.17 | 51.15 | 55.51 | cloud GPT-4 |
| Zep | 42.29 ± 0.18 | — | — | cloud |

What matters for the pitch:
- synapt is competitive with the best published systems
- the local-first path remains strong on commodity hardware
- the system is explicit about benchmark methodology, retrieval tradeoffs, and judge-model drift
- the 72.4 LOCOMO score is the audited, reproducible number to cite

Sources:
- [LOCOMO](https://snap-research.github.io/locomo/)
- [Mem0 paper](https://arxiv.org/abs/2504.19413)
- [Engram paper](https://arxiv.org/abs/2511.12960)
- [Memobase benchmark](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md)

### CodeMemo

CodeMemo evaluates coding-memory tasks across factual recall, debugging context, architecture, temporal ordering, conventions, and cross-session continuity.

| System | Overall |
|--------|---------|
| **synapt v0.6.2** | **90.51** |
| Mem0 | 76.00 |

Source:
- [CodeMemo benchmark](evaluation/codememo/README.md)

## Security and compliance

synapt is built for teams that care where memory lives and how it is inspected.

- **Local-first by default**: transcripts, channels, journals, and indexes live on disk under `.synapt/`
- **No mandatory cloud memory backend**: core recall works locally
- **Inspectable storage**: JSONL transcripts plus SQLite/FTS5 state
- **Portable backup path**: export/import via `.synapt-archive`
- **Optional remote behavior is explicit**: sync and plugin integrations are opt-in

For disclosure and reporting policy, see [SECURITY.md](SECURITY.md).

## Example workflows

Search a prior fix:

```bash
synapt recall search "why did we disable snippets in retrieval-only mode"
```

Find a file’s prior context:

```bash
synapt recall files "src/synapt/recall/channel.py"
```

Run Codex on a timed review loop:

```bash
./scripts/codex-loop.sh \
  --interval 60 \
  --prompt "check #dev, review fresh PRs, or pick up the next unowned task. Post what you're doing in #dev." \
  -- --full-auto
```

## Why teams adopt it

Without memory, every new assistant session starts as a stranger.

With synapt, teams can:
- recover prior decisions instead of re-deriving them
- hand off work without losing context
- coordinate multiple agents without duplicating effort
- keep operational memory local, inspectable, and portable

That is the difference between an assistant demo and an operational system.
