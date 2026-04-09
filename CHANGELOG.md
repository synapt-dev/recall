# Changelog

All notable changes to synapt are documented here.

## [0.11.0] — 2026-04-09

### Added
- **Agent-attributed recall** — `TranscriptChunk` now carries an optional `agent_id` field, auto-populated from `SYNAPT_AGENT_ID` env var. Scoped search via `lookup(agent_id="opus")` returns only that agent's transcripts plus legacy chunks and shared knowledge nodes. Wildcard `agent_id="*"` searches all. (#618)
- **ActionRegistry for plugin-aware channel dispatch** — new `synapt.recall.actions` module replaces the monolithic if/elif dispatcher in `recall_channel()`. OSS registers 13 base actions; premium plugin can register additional actions or override existing ones at import time. Three-tier status model (available/locked/unknown) for action discovery. (#621, #622)
- **Structured channel message types** — messages can carry a `msg_type` field (status, claim, pr, code, message) for filtering on read. (#444)

### Improved
- **Channel dispatch** — `recall_channel()` now routes through the shared ActionRegistry instead of a 150-line hard-coded switch. Net -128 lines in server.py.
- **SQLite schema migration** — `agent_id TEXT` column added transparently to existing databases via `_migrate_chunks_table()`.

## [0.6.1] — 2026-03-13

### Added
- **Secret scrubbing** — API keys, tokens, passwords, JWTs, and connection strings are scrubbed from transcripts at index time with deterministic `[REDACTED:hash]` placeholders. New `synapt recall rescrub` CLI command retroactively cleans existing archives (#65)
- **X/Twitter MCP plugin** — read timelines, search, post, reply, and thread via `synapt.plugins` entry point with prompt injection safeguards (#62)
- **MCP server setup guide** — step-by-step configuration docs for Claude Code, Cursor, and Windsurf (#66)
- **Advanced search/config reference** — documentation for all search parameters, intent types, and configuration options (#66)
- **Windows & cross-platform support** — platform-aware paths, optional MLX/ONNX, graceful fallbacks (#45)

### Improved
- **Content-aware adaptive filtering** — conversations classified as code/personal/mixed; personal content gets relaxed consolidation but `max_knowledge=0` in retrieval (#54)
- **Proactive recall** — `recall_quick` tool for fast, speculative memory checks; improved MCP server instructions (#54)
- **Blog & website** — index page, mobile responsive, cross-links, SEO, OG tags, token efficiency section (#55, #56, #57)

### Benchmarks
- **LOCOMO J-Score: 73.38%** (v0.5.1, Ministral 3B local) — unchanged from v0.6.0; now documented with Full-Context comparison (#56)
- **CodeMemo: 94% J-Score** — new benchmark for code-project memory (50 questions, 1 project)

## [0.6.0] — 2026-03-12

### Added
- **Aggregation-aware entity search** — reduced tier discounts and wider search limits for aggregation queries, plus entity-only knowledge FTS to surface scattered facts about a person across sessions
- **Category-intent alignment** — knowledge nodes whose category matches the query intent get a 1.5x boost (e.g., decision nodes for decision queries)
- **Inferential multi-hop patterns** — aggregation classifier now handles "would X enjoy", "based on the conversation", "who is [Name]" style queries
- **Decision intent** — new intent category for surfacing past decisions, with dedicated patterns and journal decision boost
- **MCP server --dev mode** — auto-reload on source changes via `synapt server --dev` (requires `watchfiles`)
- **Plugin backend registry** — extensible model routing via `synapt.backends` entry points
- **CLI subcommand discovery** — plugins can register CLI commands via `synapt.commands` entry points
- **Tool result enrichment** — enrichment summaries now include tool output content (config values, URLs, command outputs)
- **Entity-anchored FTS** — supplementary FTS search using extracted entities for better multi-hop retrieval
- **Source session IDs** — knowledge nodes display their source session for provenance tracking

### Improved
- **Embedding-based inline dedup** — cosine similarity (≥0.80) fallback after Jaccard for knowledge node deduplication
- **Generic knowledge filter** — tool-tautology patterns and specificity signals remove low-value knowledge nodes
- **Cluster summary hallucination detection** — novel entity check prevents fabricated summaries
- **Word-aware truncation** — prevents mid-word corruption in knowledge node content
- **Garbled knowledge rejection** — filters corrupt nodes with section prefixes or malformed content
- **3B model robustness** — improved handling of smaller model outputs (JSON repair, truncated dict repair, text fallback parser)
- **Intent classification expanded** — broader factual, decision, and aggregation pattern coverage with tightened patterns to reduce false positives
- **Aggregation knowledge_boost** — increased from 1.5 to 2.5 after A/B testing showed higher boost improves retrieval
- **Default decoder model** — switched from Llama-3.2-3B to Ministral-3-3B-Instruct-2512-4bit

### Fixed
- **Intent threading** — intent parameter now threaded through both global and progressive lookup paths
- **Chunk ID collision** — fixed for short session IDs
- **Consolidation** — fixed producing 0 knowledge nodes for non-code data
- **Timestamp truncation** — preserve full timestamp for free-text date formats
- **Knowledge interleaving** — interleave by relevance instead of always prepending

### Benchmarks
- **LOCOMO J-Score: 73.38%** (Ministral 3B local enrichment) — beats Full-Context upper bound (72.90%), Mem0+Graph (68.44%), Mem0 (66.88%), Zep (65.99%)
- Open-domain 80.14% — best of all systems tested
- Multi-hop 70.21% — best of all systems tested

## [0.5.0] — 2026-03-10

### Added
- **Hybrid RRF search** — reciprocal rank fusion combining FTS5, BM25, and semantic embeddings
- **Intent classification** — routes queries to adjust embedding weight, recency decay, and knowledge boost
- **Knowledge graph** — LLM-powered enrichment and consolidation pipeline
- **Cross-encoder reranking** — ms-marco-MiniLM-L-6-v2 for result re-ranking
- **ONNX Runtime inference** — 6.6x faster T5 enrichment on CPU
- **Configurable model selection** — global and project-level config with env var overrides
- **Result deduplication** — Jaccard-based near-duplicate filtering
- **Confidence-weighted knowledge boost** — higher-confidence nodes rank higher
- **Query result cache** — repeated lookups return cached results
- **Temporal date extraction** — parse date ranges from search queries
- **Source expansion** — knowledge nodes include source_turns for chunk-level provenance
- **Proactive MCP instructions** — server instructions tell Claude to search before answering
- **GitHub Pages site** — landing page at synapt.dev
- **CLA bot** — contributor license agreement enforcement

### Fixed
- **Consolidation model routing** — fixed T5 being used instead of decoder-only
- **Markdown response parser** — handle markdown-wrapped JSON from LLMs
- **Non-project path filtering** — filter irrelevant paths from journal file lists
- **Enrichment prompt** — generalized for non-code sessions

## [0.3.0] — 2026-03-08

Initial public release with core recall functionality:
- Session transcript indexing and search
- BM25 + semantic embedding retrieval
- Journal entries and cross-session reminders
- MCP server with 13 tools
- Timeline and session listing
- Plugin architecture
