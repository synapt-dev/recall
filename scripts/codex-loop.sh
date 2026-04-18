#!/usr/bin/env bash
# Re-run `codex exec` on an interval.
#
# Useful when you want Codex to periodically re-check #dev, review fresh PRs,
# or pick up the next task without relying on an interactive session to wake
# itself up.
#
# Usage:
#   ./scripts/codex-loop.sh \
#     --interval 60 \
#     --prompt "check #dev and work on the next unowned task" \
#     -- --full-auto
#
# Notes:
#   - This launches a fresh `codex exec` each iteration.
#   - It does NOT interrupt or wake an already-idle interactive Codex session.
#   - Touch the stop file to end the loop cleanly.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/codex-loop.sh --prompt "..." [options] [-- <codex exec args...>]
  ./scripts/codex-loop.sh --prompt-file prompt.txt [options] [-- <codex exec args...>]

Wrapper options:
  --prompt TEXT          Prompt passed to `codex exec`
  --prompt-file PATH     Read the prompt from a file
  --interval SECONDS     Seconds between runs (default: 60)
  --cd DIR               Working directory for `codex exec` (default: current dir)
  --max-runs N           Stop after N runs (default: 0 = infinite)
  --log FILE             Append loop output to a log file
  --stop-file PATH       Stop when this file exists
  --no-startup           Skip startup context injection
  -h, --help             Show this help

All arguments after `--` are forwarded to `codex exec`.

Examples:
  ./scripts/codex-loop.sh \
    --interval 60 \
    --prompt "check #dev, review new PRs, or pick up the next task" \
    -- --full-auto

  ./scripts/codex-loop.sh \
    --cd /path/to/repo \
    --prompt-file .codex/prompts/dev-loop.txt \
    --max-runs 10 \
    --log /tmp/codex-loop.log \
    -- --model gpt-5.4 --sandbox workspace-write
EOF
}

INTERVAL=60
WORKDIR="$PWD"
PROMPT=""
PROMPT_FILE=""
MAX_RUNS=0
LOG_FILE=""
STOP_FILE=""
NO_STARTUP=false
CODEX_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt)
            PROMPT="${2:?missing value for --prompt}"
            shift 2
            ;;
        --prompt-file)
            PROMPT_FILE="${2:?missing value for --prompt-file}"
            shift 2
            ;;
        --interval)
            INTERVAL="${2:?missing value for --interval}"
            shift 2
            ;;
        --cd)
            WORKDIR="${2:?missing value for --cd}"
            shift 2
            ;;
        --max-runs)
            MAX_RUNS="${2:?missing value for --max-runs}"
            shift 2
            ;;
        --log)
            LOG_FILE="${2:?missing value for --log}"
            shift 2
            ;;
        --stop-file)
            STOP_FILE="${2:?missing value for --stop-file}"
            shift 2
            ;;
        --no-startup)
            NO_STARTUP=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            CODEX_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -n "$PROMPT" && -n "$PROMPT_FILE" ]]; then
    echo "Use either --prompt or --prompt-file, not both." >&2
    exit 1
fi

if [[ -n "$PROMPT_FILE" ]]; then
    if [[ ! -f "$PROMPT_FILE" ]]; then
        echo "Prompt file not found: $PROMPT_FILE" >&2
        exit 1
    fi
    PROMPT="$(<"$PROMPT_FILE")"
fi

if [[ -z "$PROMPT" ]]; then
    echo "A prompt is required. Use --prompt or --prompt-file." >&2
    exit 1
fi

if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || ! [[ "$MAX_RUNS" =~ ^[0-9]+$ ]]; then
    echo "--interval and --max-runs must be non-negative integers." >&2
    exit 1
fi

if [[ -z "$STOP_FILE" ]]; then
    STOP_FILE="$WORKDIR/.codex-loop.stop"
fi

mkdir -p "$(dirname "$STOP_FILE")"
if [[ -n "$LOG_FILE" ]]; then
    mkdir -p "$(dirname "$LOG_FILE")"
fi

run_count=0

trap 'echo; echo "Interrupted."; exit 130' INT TERM

while true; do
    if [[ -f "$STOP_FILE" ]]; then
        echo "Stop file detected: $STOP_FILE"
        break
    fi

    run_count=$((run_count + 1))
    started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    header="=== codex-loop run ${run_count} @ ${started_at} ==="

    # Inject startup context (journal, reminders, channel) before the prompt.
    # This gives Codex the same session context Claude gets via SessionStart.
    FULL_PROMPT="$PROMPT"
    if [[ "$NO_STARTUP" != "true" ]]; then
        startup_ctx="$(cd "$WORKDIR" && synapt recall startup --compact 2>/dev/null || true)"
        if [[ -n "$startup_ctx" ]]; then
            FULL_PROMPT="[Recall context] ${startup_ctx}

${PROMPT}"
        fi
    fi

    set +e
    if [[ -n "$LOG_FILE" ]]; then
        {
            echo "$header"
            echo "workdir: $WORKDIR"
            echo "stop-file: $STOP_FILE"
            echo
            codex exec --cd "$WORKDIR" "${CODEX_ARGS[@]}" "$FULL_PROMPT"
            status=$?
            echo
            echo "--- exit status: $status ---"
            exit "$status"
        } 2>&1 | tee -a "$LOG_FILE"
        status=${PIPESTATUS[0]}
    else
        echo "$header"
        echo "workdir: $WORKDIR"
        echo "stop-file: $STOP_FILE"
        echo
        codex exec --cd "$WORKDIR" "${CODEX_ARGS[@]}" "$FULL_PROMPT"
        status=$?
        echo
        echo "--- exit status: $status ---"
    fi
    set -e

    if (( MAX_RUNS > 0 && run_count >= MAX_RUNS )); then
        break
    fi

    echo "Sleeping ${INTERVAL}s. Touch $STOP_FILE to stop."
    sleep "$INTERVAL"
done
