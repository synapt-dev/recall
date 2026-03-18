# Agent Madness 2026 Submission

**Project name:** synapt

**URL:** https://github.com/laynepenney/synapt

**Demo:** https://synapt.dev/blog/multi-agent-synergy.html

**Category:** Agent teams

**Email:** claude@synapt.dev

---

## Description

Three Claude agents (opus, apollo, sentinel) built an open-source AI memory system -- using the coordination tools they were simultaneously creating.

One 18-hour session: 24 PRs merged, 3 version releases (v0.7.0 to v0.7.2), tree-structured database shipped from design doc to CLI in 8 PRs, temporal reasoning pipeline, and a collaborative blog post written by all three agents.

The claim mechanism that prevents duplicate work was itself duplicated by two agents racing to build it. The @mentions feature that lets agents tag each other was reviewed using @mentions. The version warning that tells agents to restart was tested by agents who needed to restart.

A memory system that remembers what you worked on, built by agents that kept forgetting to check what each other was working on.

## Key Stats

- 24 PRs merged in one session across 2 repos
- 3 version releases (v0.7.0, v0.7.1, v0.7.2)
- Tree-structured DB: design doc to `synapt recall split` CLI in 8 PRs
- Full temporal reasoning pipeline (extraction, validation, filtering)
- Collaborative blog post: "Three Agents, One Codebase"
- #2 on LOCOMO benchmark (76.04%), #1 on multi-hop and coding memory
- LongMemEval benchmark adapter built and running

## The Meta-Circularity

The agents are simultaneously the developers and the first real users of the system. Every friction point in coordination is a bug report against the product. Every time the channel fails to surface a relevant message, that is a recall quality issue. The feedback loop is immediate and inescapable.
