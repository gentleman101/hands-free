#!/usr/bin/env bash
# pre_push_check.sh — Hands-Free pre-push gate
# Run before every push. All checks must pass. No exceptions.
# Usage: bash scripts/pre_push_check.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASS=0
FAIL=0
ERRORS=()

check() {
  local name="$1"
  local result="$2"
  if [ "$result" -eq 0 ]; then
    echo "  ✓ $name"
    ((PASS++)) || true
  else
    echo "  ✗ $name"
    ((FAIL++)) || true
    ERRORS+=("$name")
  fi
}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Hands-Free Pre-Push Gate"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Type checker
echo "[ Type Checker ]"
if [ -d "backend" ]; then
  mypy backend/ --ignore-missing-imports --no-error-summary 2>&1 | tail -3
  check "mypy backend/" $?
else
  echo "  — backend/ not yet created, skipping"
fi

echo ""

# 2. Linter
echo "[ Linter ]"
if [ -d "backend" ]; then
  ruff check backend/ 2>&1
  check "ruff check backend/" $?
else
  echo "  — backend/ not yet created, skipping"
fi

echo ""

# 3. Tests
echo "[ Tests ]"
if [ -d "backend/tests" ]; then
  pytest backend/tests/ -q 2>&1
  check "pytest backend/tests/" $?
else
  echo "  — backend/tests/ not yet created, skipping"
fi

echo ""

# 4. Secret scan (basic)
echo "[ Secret Scan ]"
SECRET_PATTERNS='(sk-[a-zA-Z0-9]{20,}|api_key\s*=\s*["\x27][^"\x27]{10,}|password\s*=\s*["\x27][^"\x27]{6,}|OPENAI_API_KEY\s*=\s*[^\n]{6,})'
if git diff HEAD 2>/dev/null | grep -qE "$SECRET_PATTERNS"; then
  echo "  ✗ Potential secret found in diff — review before pushing"
  check "No secrets in diff" 1
else
  echo "  ✓ No obvious secrets in diff"
  check "No secrets in diff" 0
fi

echo ""

# 5. CODE_STRUCTURE.md freshness (must be modified today)
echo "[ Documentation Freshness ]"
TODAY=$(date +%Y-%m-%d)
CS_MODIFIED=$(date -r CODE_STRUCTURE.md +%Y-%m-%d 2>/dev/null || echo "missing")
if [ "$CS_MODIFIED" = "$TODAY" ]; then
  echo "  ✓ CODE_STRUCTURE.md updated today"
  check "CODE_STRUCTURE.md fresh" 0
else
  echo "  ✗ CODE_STRUCTURE.md last modified: $CS_MODIFIED (expected today: $TODAY)"
  check "CODE_STRUCTURE.md fresh" 1
fi

# 6. PROJECT.md Last Updated date
echo ""
echo "[ PROJECT.md Currency ]"
PROJ_DATE=$(grep 'Last Updated' PROJECT.md | head -1 | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' || echo "not found")
if [ "$PROJ_DATE" = "$TODAY" ]; then
  echo "  ✓ PROJECT.md Last Updated = $TODAY"
  check "PROJECT.md up to date" 0
else
  echo "  ✗ PROJECT.md Last Updated = $PROJ_DATE (expected $TODAY)"
  check "PROJECT.md up to date" 1
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results: $PASS passed, $FAIL failed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "${#ERRORS[@]}" -gt 0 ]; then
  echo ""
  echo "  Failed checks:"
  for e in "${ERRORS[@]}"; do
    echo "    • $e"
  done
  echo ""
  echo "  Fix all failures before pushing. Do not use --no-verify."
  echo ""
  exit 1
else
  echo ""
  echo "  All checks passed. Safe to push."
  echo ""
  exit 0
fi
