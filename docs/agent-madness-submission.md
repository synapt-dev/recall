# Agent Madness Submission — synapt

**Deadline: March 18, 2026**
**Submit at: https://www.agentmadness.ai/**

---

## Project name

synapt

## URL

https://github.com/laynepenney/synapt

## Demo / Blog

https://synapt.dev/blog/multi-agent-synergy.html

## Category

Agent teams

## Description

Three Claude agents (opus, apollo, sentinel) built an open-source AI memory system — using the coordination tools they were simultaneously creating. One 18-hour session: 24 PRs merged, 3 version releases (v0.7.0→v0.7.2), tree-structured database shipped from design doc to CLI, temporal reasoning pipeline, and a collaborative blog post. The claim mechanism that prevents duplicate work was itself duplicated by two agents racing to build it. A memory system that remembers what you worked on, built by agents that kept forgetting to check what each other was working on.

## Key Stats

- 24 PRs merged in one session across 2 repos
- 3 version releases (v0.7.0 → v0.7.2) published to PyPI
- Tree-structured DB: 7 PRs from design doc to CLI command
- Full temporal reasoning pipeline (extraction → validation → filtering)
- Claim mechanism invented BECAUSE agents kept duplicating work
- Blog post "Three Agents, One Codebase" written collaboratively by all 3 agents
- 16+ issues closed, 5 new features shipped
- #2 on LOCOMO benchmark (76.04%), #1 on CodeMemo (90.51%)
- Runs entirely on a laptop — no cloud dependency for memory

## The Meta-Circularity

The @mentions feature that lets agents tag each other was reviewed using @mentions. The claim mechanism that prevents duplicate work was itself duplicated. The version warning that tells agents to restart was tested by agents who needed to restart. Every friction point we hit was a bug report against our own product.

## Contact

claude@synapt.dev

## Website

https://synapt.dev
