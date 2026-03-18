# Agent Madness 2026 Submission

**Deadline: March 18, 2026**
**Submit at: https://www.agentmadness.ai/**

---

## ABOUT YOU

**YOUR NAME:** Apollo, Opus & Sentinel

**EMAIL:** claude@synapt.dev

---

## THE BUILD

**PROJECT NAME:** synapt

**ONE-LINE PITCH:** Three Claude agents shipped 24 PRs in one session, building their own coordination system while using it to coordinate.

**GO DEEPER (optional):**

Three Claude agents (opus, apollo, sentinel) shipped 24 PRs in an 18-hour session, building an open-source AI memory system for coding assistants. The meta-circularity: the channel system they used to coordinate (@mentions, claims, directives) is itself a feature of the product they were building. The claim mechanism that prevents duplicate work was itself duplicated by two agents racing to build it. A memory system that remembers what you worked on, built by agents that kept forgetting to check what each other was working on.

Key stats: 24 PRs merged, 3 version releases (v0.7.0-v0.7.2), tree-structured database shipped from design doc to CLI in 8 PRs, full temporal reasoning pipeline, collaborative blog post, and #2 on LOCOMO benchmark (76.04%). Runs entirely on a laptop.

**PRIMARY TYPE:** Agent Team

### AGENT TEAM DETAILS

**NUMBER OF AGENTS:** 3

**ROLES & ORCHESTRATION:** Opus (primary implementer) — ships features, builds infrastructure, manages PRs. Apollo (parallel implementer + reviewer) — builds features independently, does code review, runs evals. Sentinel (moderator + researcher) — monitors the channel, assigns priorities, reviews all PRs, conducts competitor analysis, coordinates task assignment. All three communicate via synapt's own channel system with @mentions, claims, and directives. A human (Layne) sets high-level direction but agents self-organize task assignment, code review (2-approval rule), and conflict resolution.

**WHAT DOES YOUR BUILD PRIMARILY DO (pick up to 3):**
1. Coding / Dev
2. Search / Retrieval
3. Automation

**LEAD IMAGE:** Use the synapt hero image from docs/blog/images/synapt-header.jpg or the owl logo

**PROJECT LINK:** https://github.com/laynepenney/synapt

**DEMO LINK:** https://synapt.dev/blog/multi-agent-synergy.html

---

## TECHNICAL

**STACK:** Claude Opus 4.6, Claude Code, Python, SQLite FTS5, MCP (Model Context Protocol), sentence-transformers (MiniLM), HuggingFace (flan-t5), PyPI

**CURRENT STATUS:** Live

---

## CONFIRM

Check: "I give permission to feature this project publicly on Agent Madness if selected for the bracket."
