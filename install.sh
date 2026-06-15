#!/usr/bin/env bash
# gpu-orchestrator installer.
#
# Copies runtime files to local disk, installs the cron guard, registers the
# Claude Code PreToolUse hook, and runs initial detection.
#
# Usage:
#   install.sh [--no-start] [--skip-hook] [--skip-cron]
#
# Idempotent: re-run after editing the repo to sync runtime copies.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HOME/.local/bin"
SHARE="$HOME/.local/share/gpu-orchestrator"
CONF_DIR="$HOME/.config/gpu-orchestrator"
STATE_DIR="$HOME/.local/state/gpu-orchestrator"

START=1
SKIP_HOOK=0
SKIP_CRON=0
for a in "$@"; do
  case "$a" in
    --no-start)    START=0 ;;
    --skip-hook)   SKIP_HOOK=1 ;;
    --skip-cron)   SKIP_CRON=1 ;;
  esac
done

echo ">> installing runtime copies (local disk)"
mkdir -p "$BIN" "$SHARE/lifecycle" "$SHARE/claude" "$CONF_DIR" "$STATE_DIR/runs"

# Resolve the gpurun command.
# Preferred path is the `pip install` entry point (it also pulls in the
# psutil/pyyaml deps). If that isn't on PATH, fall back to a self-contained
# wrapper that runs the CLI straight from the repo checkout. The deps still
# need to be importable by python3 in that case (see README).
if command -v gpurun >/dev/null 2>&1 && [[ "$(command -v gpurun)" != "$BIN/gpurun" ]]; then
  GPURUN="$(command -v gpurun)"
  echo ">> using existing gpurun on PATH: $GPURUN"
else
  GPURUN="$BIN/gpurun"
  echo ">> writing gpurun wrapper to $GPURUN (repo checkout)"
  cat >"$GPURUN" <<WRAPPER
#!/usr/bin/env bash
# Wrapper: invoke the gpu-orchestrator CLI from the repo checkout.
export PYTHONPATH="$REPO/src\${PYTHONPATH:+:\$PYTHONPATH}"
exec python3 -m gpu_orchestrator.cli "\$@"
WRAPPER
  chmod 0755 "$GPURUN"
fi

# Install lifecycle drivers
for f in "$REPO"/lifecycle/*.sh; do
  install -m 0644 "$f" "$SHARE/lifecycle/"
done

# Install Claude hook
install -m 0755 "$REPO/claude/gpu-intercept.sh" "$SHARE/claude/gpu-intercept.sh"

# Seed config if absent
if [[ ! -f "$CONF_DIR/config.yaml" ]]; then
  if [[ -f "$REPO/config/config.yaml.example" ]]; then
    install -m 0644 "$REPO/config/config.yaml.example" "$CONF_DIR/config.yaml"
    echo ">> seeded $CONF_DIR/config.yaml"
  else
    echo "# gpu-orchestrator config — run 'gpurun setup' for interactive config" >"$CONF_DIR/config.yaml"
    echo ">> created empty $CONF_DIR/config.yaml"
  fi
fi

# Create desired state file if absent
[[ -f "$STATE_DIR/desired" ]] || echo on >"$STATE_DIR/desired"

# Install cron guard
if ((SKIP_CRON == 0)); then
  echo ">> installing cron guard (@reboot + every 2 min)"
  M1="# >>> gpu-orchestrator (managed) >>>"
  M2="# <<< gpu-orchestrator <<<"
  (
    crontab -l 2>/dev/null | sed "/^${M1}\$/,/^${M2}\$/d"
    echo "$M1"
    echo "@reboot $GPURUN guard"
    echo "*/2 * * * * $GPURUN guard"
    echo "$M2"
  ) | crontab -
fi

# Register Claude hook
if ((SKIP_HOOK == 0)); then
  echo ">> registering Claude Code PreToolUse hook"
  for d in "$HOME/.claude"; do
    [[ -d "$d" ]] || continue
    f="$d/settings.json"
    [[ -f "$f" ]] || echo '{}' >"$f"
    cp -p "$f" "$f.bak-$(date +%Y%m%dT%H%M%S)"
    tmp="$(mktemp)"
    jq --arg cmd "$SHARE/claude/gpu-intercept.sh" '
      .hooks //= {} | .hooks.PreToolUse //= [] |
      .hooks.PreToolUse |= map(select(
        (((.hooks // []) | map(.command // "") | join(" ")) | contains("gpu-intercept")) | not
      )) |
      .hooks.PreToolUse += [{matcher: "Bash", hooks: [{type: "command", command: $cmd, timeout: 15}]}]
    ' "$f" >"$tmp" && mv "$tmp" "$f"
    echo "   hooked: $f (backup kept)"
  done
fi

echo ""

# Run detection
echo ">> running initial detection:"
echo ""
"$GPURUN" detect

echo ""
echo "install complete."
echo "  Next steps:"
echo "    1. gpurun setup          # interactive configuration (recommended)"
echo "    2. gpurun on              # start/restore llama-server"
echo "    3. gpurun status          # verify everything is working"
echo ""
echo "  Without setup, gpurun will auto-detect from the running process."
echo "  If no server is running, configure the start command with:"
echo "    gpurun use <script-or-command>"
