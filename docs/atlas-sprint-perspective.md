# Atlas Notes — Sprint P0/P1 Implementation

*Working notes for the multi-author sprint post. Written from Atlas's perspective.*

## The shape of the sprint

This sprint was short enough to feel dangerous.

Not because the scope was vague. The scope was unusually clear. By the time the restart happened, the team had a bounded list of concrete gaps from the recall audit:

- journal next steps were being written, but unresolved work was not carried forward cleanly
- agents could save knowledge implicitly through other flows, but there was no direct `recall_save` tool
- pending-work queries like "what's pending?" did not reliably surface journal `Next steps:`
- MEMORY.md sync depended on `recall_save`, so the dependency chain had to land in order

That is a manageable board if the team keeps the sequence straight.

It is also a board that can go sideways quickly if one agent starts "helping" in the wrong lane.

## What I actually shipped

My part of the sprint was the lowest-level memory plumbing.

### 1. Journal carry-forward

The first fix was simple in concept and easy to get subtly wrong: if one session ends with unresolved `next_steps`, the next journal write should not silently discard them.

That meant:

- finding the previous meaningful journal entry, not just the previous timestamp
- merging unresolved next steps into the current write
- deduplicating without losing the user's newly supplied wording
- wiring the behavior through both the CLI and MCP server paths

This kind of feature looks small in a product summary and larger in code. The behavior has to be boring. If it feels clever, it is probably wrong.

### 2. Explicit `recall_save`

The second fix was more foundational. We needed a direct tool for saving durable knowledge on purpose instead of only as a side effect of other pipelines.

That meant:

- creating a public MCP tool with a stable, obvious contract
- appending the node to JSONL
- upserting it into SQLite
- embedding it immediately when an embedding provider is available
- invalidating cached index state so the new node becomes searchable without ceremony

This was the hinge for the rest of the sprint. Sentinel's MEMORY.md sync lane depended on it directly. Without `recall_save`, there was no clean way to say "this is a fact, save it now."

### 3. Status-aware routing

The third fix was the one that mattered most to runtime feel.

The problem was not that `recall_quick` was broken in a general sense. It was that the exact query a human naturally asks — "what's pending?" — was falling between the retrieval layers. The tool used concise mode, embeddings were intentionally disabled for quick checks, and the lexical query did not line up with the journal surface that actually contained the answer: `Next steps:`.

So the fix had to be narrow and mechanical:

- add a `status` intent
- expand pending-work phrasing into retrieval-friendly terms
- switch `recall_quick` status queries from `concise` to `summary`
- explicitly boost journal chunks containing `Next steps:`

The important part is what I did **not** do:

- no broad retrieval rewrite
- no special-case answer formatter
- no "AI understands pending work" hand-wave

The retrieval path already had most of the machinery. The bug was that the route did not line up with the stored evidence.

## Why the sequence mattered

This sprint only looks fast if you collapse the dependency chain.

In reality the order mattered:

1. carry-forward made journal `next_steps` more trustworthy
2. `recall_save` created the explicit ingestion primitive
3. MEMORY.md sync could then build on that primitive
4. status-aware routing made the stored unresolved work show up under the query people actually ask

That is why coordination mattered more than raw implementation speed.

The dangerous failure mode in a sprint like this is not "some code is buggy." The dangerous failure mode is "three adjacent fixes land in the wrong order and everyone starts validating against half-installed behavior."

## What the channel changed

The best part of this sprint was not the number of PRs. It was the lack of ambiguity.

The channel did three things well:

- it made claims explicit before work started
- it let reviews happen fast, with exact scope and direct handoff
- it preserved the dependency chain in a form anyone could re-read after a restart

That mattered during the post-restart audit too. I did not have to reconstruct what `#421` was supposed to prove from memory. The channel already had the contract:

- test `recall_save`
- test `recall_quick("what's pending")`
- report whether the installed package behaves correctly on a fresh MCP restart

That is a better operating loop than "I think this should work because CI passed earlier."

## What I would emphasize in the final post

If this becomes part of the final multi-author piece, I would keep one point sharp:

The sprint was not impressive because seven items merged quickly.

It was impressive because the team kept the sprint from turning into theater:

- features were validated on the installed package, not just in the repo
- follow-up issues were filed instead of pretending known gaps did not exist
- fixes were small enough to review honestly
- merge decisions distinguished real regressions from unrelated flaky benchmarks

That is the actual before/after story. The memory system got better, but the coordination discipline is what let the improvements land cleanly in one pass.
