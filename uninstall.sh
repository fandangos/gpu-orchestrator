#!/usr/bin/env bash
# gpu-orchestrator uninstaller.
#
# Removes the cron guard, Claude hook registrations, and runtime copies.
# By default the llama-server PROCESS IS LEFT RUNNING.
#
# Usage:
#   uninstall.sh [--stop-server] [--purge]
#
#   --stop-server   also stop llama-server
#   --purge         also delete config + state/logs

set -euo pipefail

BIN="$HOME/.local/bin"
SHARE="$HOME/.local/share/gpu-orchestrator"
CONF_DIR="$HOME/.config/gpu-orchestrator"
STATE_DIR="$HOME/.local/state/gpu-orchestrator"

STOP=0
PURGE=0
for a in "$@"; do
  case "$a" in
    --stop-server) STOP=1 ;;
    --purge)       PURGE=1 ;;
  esac
done

if ((STOP)) && [[ -x "$BIN/gpurun" ]]; then
  echo ">> stopping llama-server"
  "$BIN/gpurun" off 2>/dev/null || true
fi

echo ">> removing cron guard"
M1="# >>> gpu-orchestrator (managed) >>>"
M2="# <<< gpu-orchestrator <<<"
(crontab -l 2>/dev/null | sed "/^${M1}\$/,/^${M2}\$/d") | crontab - 2>/dev/null || true

echo ">> removing Claude hook registrations"
for d in "$HOME/.claude"; do
  f="$d/settings.json"
  [[ -f "$f" ]] || continue
  cp -p "$f" "$f.bak-uninstall-$(date +%Y%m%dT%H%M%S)"
  tmp="$(mktemp)"
  jq '
    if .hooks.PreToolUse then
      .hooks.PreToolUse |= map(select(
        (((.hooks // []) | map(.command // "") | join(" ")) | contains("gpu-intercept")) | not
      ))
    else . end
  ' "$f" >"$tmp" && mv "$tmp" "$f"
  echo "   unhooked: $f"
done

echo ">> removing runtime copies"
rm -f "$BIN/gpurun"
rm -rf "$SHARE"

if ((PURGE)); then
  echo ">> purging config + state/logs"
  rm -rf "$CONF_DIR" "$STATE_DIR"
fi

echo ""
echo "uninstalled. llama-server is now unmanaged (manage it manually)."
