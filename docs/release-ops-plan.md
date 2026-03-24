# Release Ops Plan

This is the working operating plan for the next release sequence after `v0.7.7`.

## Operating Model

Each release should have:

- one primary goal
- one accountable owner per active lane
- one measurable success gate
- one explicit not-doing list

Rules:

- Only one benchmark-improvement lane sits on the critical path at a time.
- Every benchmark claim needs an apples-to-apples comparator before it gets socialized.
- Docs, blog, and demo work trail product truth; they do not define product truth.
- If a lane is blocked, post the first concrete blocker instead of a generic status recap.

## v0.7.8: Quality And Polish

Primary goal:

- Ship one measurable recall-quality improvement and clean up the top production pain without widening scope.

Active lanes:

- Apollo: MW Option A with 8B on Modal (`#478`)
- Sentinel: comparison framework plus apples-to-apples eval readout for MW Option A
- Opus: release framing, success criteria, and final ship recommendation
- Atlas: queue control, claim hygiene, and go/no-go integration

Success gates:

- MW Option A run completes with a direct comparator against the current baseline
- knowledge-node duplication is either materially reduced or sharply bounded with evidence
- release decision is made from measured deltas, not intuition

Not doing:

- no broad cloud or distribution work
- no parallel benchmark lane alongside MW Option A
- no large docs or website churn except what directly supports the release

## v0.8.0: Recall Quality Leap

Primary goal:

- Improve retrieval precision and evidence quality, not just coverage.

Planned lanes:

- offset-based source refs for exact snippet retrieval
- knowledge dedup overhaul
- category-aware specificity scoring

Success gates:

- exact source refs land and are usable in recall surfaces
- dedup changes are validated on real recall outputs
- at least one benchmark shows a clear quality gain attributable to these changes

Not doing:

- no premature multi-user or platform expansion before precision improves

## v0.9.0: Scale And Distribution

Primary goal:

- Turn Synapt from strong solo memory into viable shared team memory.

Planned lanes:

- optional cloud sync
- multi-user recall
- unified grip plus recall product story under `synapt.dev`

Success gates:

- shared-project memory works across users cleanly
- product story is explainable in one demo, one page, and one install path

Not doing:

- no speculative growth work without a coherent shared-memory workflow

## Immediate Sequence

1. Launch MW Option A with 8B on Modal as the only benchmark-critical lane.
2. Run the apples-to-apples comparison and classify the result.
3. Decide whether `v0.7.8` ships or iterates.
4. Only then open the main `v0.8.0` precision work.

## Ownership Notes

- Apollo owns the implementation lane on the active benchmark path.
- Sentinel owns comparison structure and result framing discipline.
- Opus owns release framing and recommendation quality.
- Atlas owns queue discipline, scope control, and decision gates.
