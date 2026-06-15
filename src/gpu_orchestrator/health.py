"""Health checking for llama.cpp OpenAI-compatible API."""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error


def health_ok(url: str, timeout: int = 3) -> bool:
    """Check /health endpoint."""
    try:
        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def models_ok(url: str, timeout: int = 3) -> bool:
    """Check /v1/models endpoint (must return 200 with data)."""
    try:
        req = urllib.request.Request(f"{url}/v1/models")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            data = resp.read()
            # Must have a non-empty data array
            import json

            parsed = json.loads(data)
            data_arr = parsed.get("data", [])
            return len(data_arr) > 0
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_for_health(
    url: str,
    timeout: int = 180,
    interval: int = 2,
    is_running=None,
) -> bool:
    """Wait until both /health and /v1/models respond.

    Args:
        url: Base URL of the API.
        timeout: Maximum seconds to wait.
        interval: Seconds between checks.
        is_running: Optional callable that returns True while process is alive.
            If provided and returns False, aborts early (server died during load).

    Returns:
        True if healthy, False on timeout or process death.
    """
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if health_ok(url) and models_ok(url):
            return True
        if is_running is not None and not is_running():
            return False
        time.sleep(interval)
    return False
