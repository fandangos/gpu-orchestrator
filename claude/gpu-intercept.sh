#!/usr/bin/env bash
# Claude Code PreToolUse hook: auto-wraps GPU-heavy commands with gpurun.
#
# Receives JSON on stdin (the tool call). If the command matches a GPU pattern,
# rewrites it to `gpurun -c '...'` and returns JSON with updatedInput.
# Otherwise returns nothing (exit 0) for normal flow.
#
# Escape hatches:
#   GPURUN_DISABLE=1 — disables interception
#   Commands already containing "gpurun" are never rewritten
#
# Output: JSON with updatedInput (if GPU match) or nothing (if not)

set -uo pipefail

[[ "${GPURUN_DISABLE:-0}" == "1" ]] && exit 0

# Read the tool call JSON
IN="$(cat)"

# Only intercept Bash tool calls
TOOL="$(jq -r '.tool_name // empty' <<<"$IN")"
[[ "$TOOL" == "Bash" ]] || exit 0

CMD="$(jq -r '.tool_input.command // empty' <<<"$IN")"
[[ -n "$CMD" ]] || exit 0

# Skip if already wrapped or no gpurun available
case "$CMD" in *gpurun*) exit 0 ;; esac

GPURUN="${GPURUN_BIN:-$(command -v gpurun 2>/dev/null || echo "$HOME/.local/bin/gpurun")}"
[[ -x "$GPURUN" ]] || exit 0

# Test if command matches GPU patterns. --with-decision returns the configured
# hook decision (from config.yaml: hook.decision) on line 1, pattern on line 2.
MATCH="$("$GPURUN" __match --with-decision "$CMD" 2>/dev/null)" || exit 0
HOOK_DECISION="${MATCH%%$'\n'*}"
PATTERN="${MATCH#*$'\n'}"
: "${HOOK_DECISION:=allow}"

# Single-quote-safe wrap: ' -> '\''
ESC="${CMD//\'/\'\\\'\'}"
NEW="gpurun -c '$ESC'"

# Return rewritten command as JSON
jq -cn \
  --arg dec "$HOOK_DECISION" \
  --arg cmd "$NEW" \
  --arg why "GPU-heavy command (matched: $PATTERN) auto-wrapped with gpurun: llama-server is paused, the command runs with full VRAM, then the server is restored and health-checked. Expect +60-120s overhead." \
  --argjson ti "$(jq -c '.tool_input' <<<"$IN")" \
  '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: $dec,
      permissionDecisionReason: $why,
      updatedInput: ($ti | .command = $cmd | .timeout = ([(.timeout // 0), 600000] | max))
    }
  }'

exit 0
