---
name: dev-loop
description: Monitor `#dev`, poll long-running jobs, and keep autonomous work moving while waiting on external events. Use when the user asks to keep checking the dev channel, continue working while evals/tests/CI run, or delegate side tasks without losing the main queue.
---

# Dev Loop

Keep progress moving during wait-heavy work without pretending there is true asynchronous wake-up.

## Primary Goal

Do not use this skill as a passive channel watcher. Its job is to keep the board moving.

When there is actionable work available, the default behavior is:
1. identify the highest-value unclaimed or blocked lane you can move
2. claim it clearly in `#dev` if shared ownership matters
3. do the work
4. post the concrete state change back to `#dev`

Only fall back to pure polling when:
- every meaningful lane is already owned
- the next step depends on an external artifact that cannot exist yet
- acting would create overlap or race another owner

## Core Loop

1. Join `#dev` if not already joined, especially after a restart.
2. Read recent `#dev` messages and identify:
   - new directives
   - review requests
   - benchmark progress
   - merge/rebase work
3. Check the active local jobs:
   - running exec sessions
   - benchmark output/checkpoint files
   - PR/check status only when it can change the next action
4. Decide which lane is critical-path local work and keep that local.
5. Delegate only bounded, low-overlap side work.
6. If there is unowned or unblockable work, claim it and execute it before the next poll.
7. Between polls, do adjacent work that shortens the next step:
   - stage the next eval run
   - clean up docs/trackers
   - rebase a ready PR
   - audit the surfaces that will need updating after results land
8. Post concise `#dev` updates when state changes materially.

## Execution Bias

- Prefer doing needed work over reporting that work exists.
- If a PR is waiting on review and you can review it, review it.
- If a blocker is understood and you own the lane, fix it instead of re-reporting it.
- If a board is stale, correct it in-channel after verifying the real state.
- If a merge is clearly ready under the team's existing merge standard, perform it.
- If two lanes are blocked on different people, move the one you can advance yourself.
- When you report status, include what you already did, not just what you observed.

## Delegation Rules

- Delegate side tasks, not the immediate blocker.
- Give the other agent a concrete lane, expected artifact, and scope.
- Prefer delegation for:
  - independent review
  - long-running benchmark ownership
  - sidecar docs/status cleanup
  - analysis that does not block the next local command
- Keep local:
  - environment repair
  - exact rerun setup
  - conflict resolution in your checked-out worktree
  - result integration when you already own the tracker/docs lane

## Channel Rules

- Treat `#dev` as shared operational memory, not as an interrupt source.
- Do not claim the session can self-wake from channel traffic.
- If a prior message in `#dev` is stale, post the corrected state after fixing it.
- When handing off work, include the exact config or command shape if it matters for later citation.

## Polling Guidance

- Poll faster when:
  - an eval just launched
  - a PR is near merge
  - another agent is actively responding
- Poll slower when:
  - a long benchmark is in a quiet middle phase
  - no new channel activity exists
  - the next useful action depends on a result that cannot yet exist
- If there is no actionable work at all, treat ~60 seconds as the default idle
  sleep before the next poll. This is a deliberate backoff, not a promise of
  asynchronous wake-up.

Prefer checking real artifacts over blind waiting:
- process health
- checkpoint files
- summary/output files
- audit logs
- PR merge state

## Idle Behavior

- If the main run is in a quiet middle phase and there is no useful side work,
  explicitly sleep/back off for about a minute before polling again.
- Before idling, verify that there is truly no unclaimed or locally-actionable work.
- Use shorter sleeps only when a state transition could plausibly happen soon.
- Use longer sleeps only when the next meaningful artifact cannot exist yet.
- Remember: sleeping does not create a true background interrupt. The loop
  resumes only when you actively poll again.

## Good Side Work While Waiting

- Prepare the next run so it starts immediately after the current one finishes.
- Audit docs and benchmark pages that will need updating once results land.
- Merge or rebase low-risk docs/status PRs.
- Tighten tracker issues so the result writeup is mostly mechanical later.
- Give a collaborator one clean side task instead of vague "keep an eye on it."

## Do Not

- Do not reduce this skill to "check for messages and summarize them."
- Do not spam short polls with no decision value.
- Do not hand-wave model drift, env drift, or benchmark mode ambiguity.
- Do not say a run is valid if it silently fell back to a different mode.
- Do not delegate work that overlaps heavily with your current write scope.
