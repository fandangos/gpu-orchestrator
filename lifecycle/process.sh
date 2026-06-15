#!/usr/bin/env bash
# lifecycle/process.sh — plain-process lifecycle driver (backward compat).
#
# Sourced by the old shell-based gpurun. The new Python gpurun uses its
# built-in ProcessDriver instead. This file is kept for rollback compatibility.
#
# Contract: drv_start, drv_stop, drv_is_running, drv_info, drv_diag

set -uo pipefail

PIDFILE="${PIDFILE:-$HOME/.local/state/gpu-orchestrator/llama.pid}"
SERVER_LOG="${SERVER_LOG:-$HOME/.local/state/gpu-orchestrator/llama-server.log}"

drv_is_running() { pgrep -f "$LLAMA_PROC_PATTERN" >/dev/null 2>&1; }

drv_info() { pgrep -f "$LLAMA_PROC_PATTERN" 2>/dev/null | paste -sd, - ; }

drv_diag() {
  echo "--- tail $SERVER_LOG ---"
  tail -n 25 "$SERVER_LOG" 2>/dev/null || echo "(no server log)"
}

drv_start() {
  if [[ ! -x "$LLAMA_SCRIPT" ]]; then
    echo "[driver] LLAMA_SCRIPT missing or not executable: $LLAMA_SCRIPT" >&2
    return 1
  fi
  if [[ -f "$SERVER_LOG" ]] && (($(stat -c%s "$SERVER_LOG" 2>/dev/null || echo 0) > 20 * 1024 * 1024)); then
    mv -f "$SERVER_LOG" "$SERVER_LOG.1"
  fi
  setsid bash -c 'echo "$$" > "$1"; exec "$2"' _ "$PIDFILE" "$LLAMA_SCRIPT" \
    >>"$SERVER_LOG" 2>&1 </dev/null 9>&- &
  local i
  for i in {1..10}; do
    drv_is_running && return 0
    sleep 0.5
  done
  echo "[driver] server process did not appear within 5s" >&2
  return 1
}

drv_stop() {
  local pid deadline
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  fi
  pkill -TERM -f "$LLAMA_PROC_PATTERN" 2>/dev/null
  deadline=$((SECONDS + STOP_TIMEOUT))
  while ((SECONDS < deadline)); do
    if ! drv_is_running; then rm -f "$PIDFILE"; return 0; fi
    sleep 0.5
  done
  [[ -n "$pid" ]] && kill -KILL -- "-$pid" 2>/dev/null
  pkill -KILL -f "$LLAMA_PROC_PATTERN" 2>/dev/null
  deadline=$((SECONDS + 5))
  while ((SECONDS < deadline)); do
    if ! drv_is_running; then rm -f "$PIDFILE"; return 0; fi
    sleep 0.5
  done
  return 1
}
