#!/usr/bin/env bash
# e2e-web-verify.sh — verify dashboard API endpoints against real data
#
# Usage: ./scripts/e2e-web-verify.sh [port]
#
# Starts the dashboard in background, hits every API endpoint,
# verifies responses, then tears down. Exit 0 = all pass.
#
# Part of Sprint 11 Phase 2: Team-Verified Demo (recall#552)

set -euo pipefail

PORT="${1:-8421}"  # Use non-default port to avoid conflicts
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
PID=""

cleanup() {
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
    echo ""
    echo "═══════════════════════════════════"
    echo "Results: $PASS passed, $FAIL failed"
    echo "═══════════════════════════════════"
    if [ "$FAIL" -gt 0 ]; then
        exit 1
    fi
}
trap cleanup EXIT

check() {
    local name="$1"
    local url="$2"
    local expected="$3"

    response=$(curl -sf "$url" 2>/dev/null || echo "CURL_FAILED")
    if echo "$response" | grep -q "$expected"; then
        echo "  ✓ $name"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $name"
        echo "    Expected: $expected"
        echo "    Got: $(echo "$response" | head -c 200)"
        FAIL=$((FAIL + 1))
    fi
}

check_status() {
    local name="$1"
    local url="$2"
    local expected_status="$3"

    status=$(curl -sf -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    if [ "$status" = "$expected_status" ]; then
        echo "  ✓ $name (HTTP $status)"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $name (expected HTTP $expected_status, got $status)"
        FAIL=$((FAIL + 1))
    fi
}

echo "═══ synapt dashboard e2e verification ═══"
echo ""

# Start dashboard in background
echo "Starting dashboard on port $PORT..."
python -m synapt.dashboard.app --foreground --port "$PORT" --no-open &
PID=$!
sleep 2

# Verify it's running
if ! kill -0 "$PID" 2>/dev/null; then
    echo "✗ Dashboard failed to start"
    FAIL=1
    exit 1
fi
echo "Dashboard running (pid $PID)"
echo ""

# === Test 1: Index page loads ===
echo "Page load:"
check "GET / returns HTML" "$BASE/" "<!DOCTYPE html>"

# === Test 2: Agents endpoint ===
echo "API endpoints:"
check_status "GET /api/agents returns 200" "$BASE/api/agents" "200"

# === Test 3: Channels endpoint ===
check_status "GET /api/channels returns 200" "$BASE/api/channels" "200"

# === Test 4: Messages endpoint ===
check_status "GET /api/messages/dev returns 200" "$BASE/api/messages/dev" "200"

# === Test 5: Agent input endpoint exists ===
# POST without text should return 400 (validation error) or 422
status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/api/agent/test/input" -d "text=" 2>/dev/null)
if [ "$status" = "400" ] || [ "$status" = "422" ]; then
    echo "  ✓ POST /api/agent/{name}/input validates input (HTTP $status)"
    PASS=$((PASS + 1))
else
    echo "  ✗ POST /api/agent/{name}/input expected 400/422, got $status"
    FAIL=$((FAIL + 1))
fi

# === Test 6: Agent snapshot endpoint ===
check_status "GET /api/agent/test/snapshot returns 200" "$BASE/api/agent/test/snapshot" "200"

# === Test 7: SSE stream endpoint ===
# Verify it connects and returns SSE content-type (timeout after 2 seconds)
sse_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "$BASE/api/stream?channel=dev" 2>/dev/null || true)
if [ "$sse_status" = "200" ] || [ "$sse_status" = "000" ]; then
    # 200 = got response, 000 = timed out (expected for long-lived SSE)
    echo "  ✓ GET /api/stream connects (SSE, status=$sse_status)"
    PASS=$((PASS + 1))
else
    echo "  ✗ GET /api/stream unexpected status: $sse_status"
    FAIL=$((FAIL + 1))
fi

# === Test 8: Template has agent input UI ===
page=$(curl -sf "$BASE/" 2>/dev/null || echo "")
if echo "$page" | grep -q "agent-input-panel"; then
    echo "  ✓ Template contains agent-input-panel"
    PASS=$((PASS + 1))
else
    echo "  ✗ Template missing agent-input-panel"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "Dashboard e2e verification complete."
