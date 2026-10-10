#!/bin/sh
# witness-where-is-generation-cost.sh [<python>]
# Run from a GRIPSPACE ROOT. Asks recall_code, the function the MCP tool calls, "where do we compute generation cost for a
# run record" and passes only if the answer names a file inside the runner member repo. Exit 0 = the root answered with
# runner's cost surface; 1 = it did not (a refusal, or an answer from somewhere else); 2 = the instrument itself is broken.
# The classifier is checked first against two strings it must reject (a refusal, an answer naming only another member), so a
# witness that can only say "pass" fails here instead of reading as green.
PY=${1:-python3}
"$PY" -I - <<'PYEOF'
import os, re, sys

QUESTION = "where do we compute generation cost for a run record"


def verdict(answer: str) -> str:
    """'ok' only when the answer is not a refusal AND the GenerationCostSurface hit line names a runner/ path."""
    if "is not itself a git repository" in answer or "Pass repo_root" in answer:
        return "refused"
    if re.search(r"GenerationCostSurface\s+runner/[\w./-]+", answer):
        return "ok"
    return "other"


# the instrument must be able to disagree: a refusal and a non-runner answer both fail
assert verdict("Repo root /x is not itself a git repository and contains member repos. Pass repo_root pointing at one") == "refused"
assert verdict("function cost_of  eval/src/eval/cost.py:10-20  match: name") == "other"
assert verdict("class RunRecord  runner/synapt/runner/records.py:33-74") == "other"  # a runner path alone is not the cost surface
assert verdict("class GenerationCostSurface  other-member/x/cost.py:1-9") == "other"  # right name, wrong member
assert verdict("class GenerationCostSurface  runner/synapt/runner/modal.py:41-66  [prefix]") == "ok"
if verdict("") != "other":
    print("BROKEN INSTRUMENT: an empty answer did not classify as other")
    sys.exit(2)

root = os.getcwd()
from synapt.recall import server  # noqa: E402
print("cwd:", root)
print("server module:", server.__file__)
answer = server.recall_code(QUESTION)
v = verdict(answer)
print("--- answer (first 1500 chars) ---")
print(answer[:1500])
print("--- verdict:", v)
sys.exit(0 if v == "ok" else 1)
PYEOF
