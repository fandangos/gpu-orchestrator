#!/usr/bin/env bash
# lifecycle/systemd.sh — OPTIONAL systemd --user lifecycle driver.
#
# NOT active by default and NOT wired into the current Python gpurun, which
# uses its built-in ProcessDriver. This file is kept as a reference template
# for running llama-server under systemd --user. To use it manually:
#   1. source it and set LLAMA_SCRIPT / LLAMA_PROC_PATTERN / STOP_TIMEOUT
#   2. run: drv_setup   (writes + enables the unit)
#   3. run: loginctl enable-linger
#
# Same contract as process.sh. Kept for backward compat only.

set -uo pipefail

LLAMA_UNIT="${LLAMA_UNIT:-llama-server}"

drv_is_running() { systemctl --user --quiet is-active "$LLAMA_UNIT" 2>/dev/null; }

drv_info() { systemctl --user show -p MainPID --value "$LLAMA_UNIT" 2>/dev/null; }

drv_diag() {
  echo "--- systemctl --user status $LLAMA_UNIT ---"
  systemctl --user status "$LLAMA_UNIT" --no-pager 2>&1 | head -20
  echo "--- journalctl tail ---"
  journalctl --user -u "$LLAMA_UNIT" -n 25 --no-pager 2>/dev/null || true
}

drv_start() { systemctl --user start "$LLAMA_UNIT" 2>&1; }

drv_stop() {
  systemctl --user stop "$LLAMA_UNIT" 2>&1
  ! pgrep -f "$LLAMA_PROC_PATTERN" >/dev/null 2>&1
}

drv_setup() {
  local unit_dir="$HOME/.config/systemd/user"
  mkdir -p "$unit_dir"
  cat >"$unit_dir/$LLAMA_UNIT.service" <<EOF
[Unit]
Description=llama.cpp server (gpu-orchestrator)
After=network-online.target

[Service]
ExecStart=$LLAMA_SCRIPT
Restart=on-failure
RestartSec=5
TimeoutStopSec=$STOP_TIMEOUT

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable "$LLAMA_UNIT" >/dev/null 2>&1
  echo "[driver] wrote $unit_dir/$LLAMA_UNIT.service (enabled). Remember: loginctl enable-linger"
}
