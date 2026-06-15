"""Process lifecycle management for llama-server.

Two drivers are available:
  - ProcessDriver: uses psutil + subprocess (default, cross-platform)
  - ShellDriver: delegates to lifecycle/process.sh (backward compat)

The ProcessDriver is the default and recommended choice.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import psutil

from .health import health_ok, models_ok, wait_for_health


class ProcessDriver:
    """Manage llama-server as a background process."""

    def __init__(
        self,
        start_cmd: str,
        proc_pattern: str | None = None,
        health_url: str = "http://127.0.0.1:8080",
        port: int = 8080,
        state_dir: Path | None = None,
        stop_timeout: int = 20,
    ):
        self.start_cmd = start_cmd
        self.proc_pattern = proc_pattern
        self.health_url = health_url
        self.port = port
        self.state_dir = state_dir or Path.home() / ".local" / "state" / "gpu-orchestrator"
        self.stop_timeout = stop_timeout

        self.pid_file = self.state_dir / "llama.pid"
        self.server_log = self.state_dir / "llama-server.log"

    # -- state queries ----------------------------------------------------
    def is_running(self) -> bool:
        """Check if a llama-server process is currently running."""
        if self.proc_pattern:
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    if proc.info["pid"] == os.getpid():
                        continue
                    if self._pattern_matches(cmdline):
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        # Fallback: check if anything is listening on the port
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == self.port and conn.pid is not None:
                try:
                    proc = psutil.Process(conn.pid)
                    cmdline = " ".join(proc.cmdline())
                    if "llama-server" in cmdline.lower():
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        return False

    def get_pid(self) -> int | None:
        """Get the PID of the running server, or None."""
        if self.proc_pattern:
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    if proc.info["pid"] == os.getpid():
                        continue
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    if self._pattern_matches(cmdline):
                        return proc.info["pid"]
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        return None

    def get_diag(self) -> str:
        """Return diagnostic information."""
        lines = [f"--- llama-server status ---"]
        pid = self.get_pid()
        lines.append(f"  PID: {pid}")
        lines.append(f"  Running: {self.is_running()}")
        lines.append(f"  Health: {self.health_url}/health")
        lines.append(f"  Models: {self.health_url}/v1/models")

        if self.server_log.exists():
            lines.append(f"  Log: {self.server_log} ({self.server_log.stat().st_size} bytes)")
            try:
                with open(self.server_log) as f:
                    tail = f.readlines()[-10:]
                    if tail:
                        lines.append("  Last log lines:")
                        for line in tail:
                            lines.append(f"    {line.rstrip()}")
            except OSError:
                pass
        return "\n".join(lines)

    # -- lifecycle --------------------------------------------------------
    def start(self) -> bool:
        """Start llama-server in a new session. Returns True on success."""
        # Validate start command
        # Parse the command to check the binary exists
        parts = self.start_cmd.split()
        if not parts:
            print(f"[driver] empty start command", flush=True)
            return False

        binary = parts[0]
        if not os.path.isfile(binary):
            # Try as a script path
            try:
                with open(binary) as f:
                    first_line = f.readline()
                if first_line.startswith("#!"):
                    shebang = first_line[2:].strip().split()
                    if shebang:
                        binary = shebang[0]
            except (OSError, IndexError):
                pass

        if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            # Could be a shell script — just try to run it
            pass

        # Log rotation (20 MB)
        if self.server_log.exists():
            if self.server_log.stat().st_size > 20 * 1024 * 1024:
                self.server_log.rename(self.server_log.with_suffix(".log.1"))

        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Start in a new session so it survives gpurun death.
        # We explicitly do NOT inherit fd 9 (GPU lock).
        try:
            with open(self.server_log, "a") as log_fh:
                proc = subprocess.Popen(
                    self.start_cmd,
                    shell=True,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )

            # Wait for process to appear (poll for up to 5 seconds)
            for _ in range(10):
                if self.is_running():
                    # Write PID file
                    pid = self.get_pid()
                    if pid:
                        self.pid_file.write_text(str(pid))
                    return True
                time.sleep(0.5)

            print(f"[driver] server process did not appear within 5s", flush=True)
            return False

        except OSError as e:
            print(f"[driver] failed to start: {e}", flush=True)
            return False

    def stop(self) -> bool:
        """Stop llama-server: TERM -> KILL escalation, then verify dead."""
        # Phase 1: try graceful stop via PID file
        pid = None
        try:
            pid = int(self.pid_file.read_text().strip())
        except (ValueError, OSError):
            pass

        if pid and self._pid_alive(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

        # Phase 2: pkill by pattern
        if self.proc_pattern:
            self._pkill(signal.SIGTERM)

        # Wait for graceful shutdown
        deadline = time.monotonic() + self.stop_timeout
        while time.monotonic() < deadline:
            if not self.is_running():
                try:
                    self.pid_file.unlink(missing_ok=True)
                except OSError:
                    pass
                return True
            time.sleep(0.5)

        # Phase 3: escalate to KILL
        if pid and self._pid_alive(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

        self._pkill(signal.SIGKILL)

        # Verify dead
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not self.is_running():
                try:
                    self.pid_file.unlink(missing_ok=True)
                except OSError:
                    pass
                return True
            time.sleep(0.5)

        return False

    def restore(self, health_timeout: int = 180) -> tuple[bool, int]:
        """Start server and wait for health. Returns (success, wait_ms).

        Exit codes:
          0  = healthy
          97 = failed to spawn
          98 = spawn OK but not healthy in time
        """
        if not self.start():
            return False, 97

        is_running = lambda: self.is_running()
        healthy = wait_for_health(
            self.health_url,
            timeout=health_timeout,
            is_running=is_running,
        )

        if healthy:
            return True, 0

        # Server died during load
        if not self.is_running():
            return False, 97

        return False, 98

    # -- internals --------------------------------------------------------
    def _pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _pattern_matches(self, cmdline: str) -> bool:
        """Check if cmdline matches the proc_pattern."""
        if not self.proc_pattern:
            return "llama-server" in cmdline.lower()
        try:
            return bool(re.search(self.proc_pattern, cmdline))
        except re.error:
            # Fallback to substring match if pattern is invalid
            return self.proc_pattern.replace("\\", "") in cmdline

    def _pkill(self, sig: int) -> None:
        """Send signal to all processes matching proc_pattern."""
        if not self.proc_pattern:
            return
        for proc in psutil.process_iter(["pid", "ppid", "cmdline"]):
            try:
                if proc.info["pid"] == os.getpid():
                    continue
                cmdline = " ".join(proc.info.get("cmdline") or [])
                if self._pattern_matches(cmdline):
                    try:
                        os.kill(proc.info["pid"], sig)
                    except (ProcessLookupError, OSError):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass


