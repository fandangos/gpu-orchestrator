"""Structured event logging for gpu-orchestrator.

Two log outputs:
  - events.jsonl  — machine-readable structured events
  - gpurun.log    — human-readable log (same data, formatted)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


class EventLogger:
    """Append structured events to JSONL + human-readable log."""

    def __init__(self, state_dir: Path, run_id: str | None = None):
        self.state_dir = state_dir
        self.run_id = run_id or ""
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.events_file = self.state_dir / "events.jsonl"
        self.log_file = self.state_dir / "gpurun.log"

    def evt(self, event: str, **kwargs) -> dict:
        """Record an event and return the event dict.

        Args:
            event: Event name (e.g. 'run_requested', 'llama_stopped').
            **kwargs: Key=value pairs to include in the event.

        Returns:
            The event dict that was written.
        """
        ts = datetime.now(timezone.utc).strftime("%FT%T.%f")[:-3] + "Z"
        record: dict = {
            "ts": ts,
            "run": self.run_id,
            "event": event,
        }
        record.update(kwargs)

        # Write JSONL
        with open(self.events_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        # Write human-readable
        fields = " ".join(f"{k}={v}" for k, v in kwargs.items())
        prefix = f"[{event}]"
        if self.run_id:
            prefix += f" run={self.run_id}"
        line = f"{datetime.now().strftime('%F %T')} {prefix} {fields}".strip()
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

        return record

    def log(self, message: str) -> None:
        """Append a free-form log line."""
        line = f"{datetime.now().strftime('%F %T')} {message}"
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def tail(self, n: int = 40) -> list[str]:
        """Return last N lines of the human-readable log."""
        if not self.log_file.exists():
            return []
        with open(self.log_file) as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-n:]]

    def follow(self, n: int = 10) -> None:
        """Tail the log file in real-time (like `tail -f`)."""
        import sys
        import time

        with open(self.log_file) as f:
            # Seek to after the last N lines
            lines = f.readlines()
            for line in lines[-n:]:
                sys.stdout.write(line)
                sys.stdout.flush()

            while True:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)
