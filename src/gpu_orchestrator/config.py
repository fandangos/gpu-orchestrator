"""Configuration management for gpu-orchestrator.

Loads from ~/.config/gpu-orchestrator/config.yaml.  All values have sane
defaults so the config file can be empty (or absent) for the common case.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import psutil

# ----------------------------------------------------------------------- defaults
DEFAULTS: dict[str, Any] = {
    "llama": {
        "url": "http://127.0.0.1:8080",
        "port": 8080,
        "proc_pattern": None,  # auto-detected
        "start_cmd": None,  # discovered from running process cmdline
    },
    "timeouts": {
        "health": 180,
        "stop": 20,
        "lock": 180,
        "grace": 15,
        "command": 900,  # max seconds for the wrapped command (15 min)
    },
    "hook": {
        "decision": "allow",  # allow | ask | defer
    },
    "patterns": {
        "extra": [],
    },
    "paths": {
        # Directories searched (in order) when `gpurun use NAME` is given a
        # bare name instead of a path. `~` is expanded.
        "model_dirs": [
            "~/.config/gpu-orchestrator/models",
            "~/models",
            "~/scripts",
            "~/.local/bin",
        ],
    },
}

# Auto-detection search paths for llama-server binary (in priority order).
LLAMA_SEARCH_PATHS = [
    # llama.cpp from source
    Path.home() / "ai/llama.cpp/build/bin/llama-server",
    Path.home() / "llama.cpp/build/bin/llama-server",
    Path.home() / "ai/llama.cpp/build/release/bin/llama-server",
    # Homebrew (macOS)
    Path("/opt/homebrew/bin/llama-server"),
    Path("/usr/local/bin/llama-server"),
    Path("/usr/bin/llama-server"),
    # Nix
    Path.home() / ".nix-profile/bin/llama-server",
    # Snap
    Path("/snap/bin/llama-server"),
    # Docker (unlikely but possible)
    Path.home() / ".local/share/containers/storage/overlay",
]

# GPU-command patterns (ERE, case-insensitive matching).
GPU_PATTERNS = [
    "comfyui",
    "sd-webui|stable-diffusion|webui\\.(sh|py)|automatic1111|a1111|fooocus|invokeai",
    "sd_scripts",
    "torchrun|accelerate|deepspeed",
    "(^|[/\\s])(inpaint|img2img|txt2img)[a-zA-Z0-9_]*\\.py($|[/\\s])",
    "\\.venvs/[a-zA-Z0-9._-]*(inpaint|diffus|comfy|flux|sdxl)",
    "flux1|flux\\.1|flux[_.]",
]


# ------------------------------------------------------------------ loading
def _config_dir() -> Path:
    return Path.home() / ".config" / "gpu-orchestrator"


def _config_path() -> Path:
    return _config_dir() / "config.yaml"


def _load_raw() -> dict[str, Any]:
    """Load config YAML, merging with defaults."""
    import yaml

    cfg: dict[str, Any] = {}
    path = _config_path()
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                cfg = data
            # Empty/comment-only file → yaml.safe_load returns None → use {}
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (base is not mutated).

    None values in override are skipped — this handles YAML keys like
    ``patterns:``  (no value) which yaml.safe_load turns into None.
    """
    result = _clone(base)
    for k, v in override.items():
        if v is None:
            continue  # skip empty YAML keys, preserve base default
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = _clone(v) if isinstance(v, (dict, list)) else v
    return result


def _clone(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone(v) for v in obj]
    return obj


class Config:
    """Immutable-ish config with auto-detection helpers."""

    def __init__(self, raw: dict[str, Any] | None = None):
        merged = _deep_merge(DEFAULTS, raw or {})
        self._raw = merged

    # -- accessors --------------------------------------------------------
    @property
    def llama_url(self) -> str:
        return self._raw["llama"]["url"]

    @property
    def llama_port(self) -> int:
        return self._raw["llama"]["port"]

    @property
    def llama_proc_pattern(self) -> str | None:
        return self._raw["llama"]["proc_pattern"]

    @property
    def llama_start_cmd(self) -> str | None:
        return self._raw["llama"]["start_cmd"]

    @property
    def health_timeout(self) -> int:
        return self._raw["timeouts"]["health"]

    @property
    def stop_timeout(self) -> int:
        return self._raw["timeouts"]["stop"]

    @property
    def lock_timeout(self) -> int:
        return self._raw["timeouts"]["lock"]

    @property
    def grace_timeout(self) -> int:
        return self._raw["timeouts"]["grace"]

    @property
    def command_timeout(self) -> int:
        return self._raw["timeouts"]["command"]

    @property
    def hook_decision(self) -> str:
        return self._raw["hook"]["decision"]

    @property
    def extra_patterns(self) -> list[str]:
        return self._raw["patterns"]["extra"]

    @property
    def all_patterns(self) -> list[str]:
        return GPU_PATTERNS + self.extra_patterns

    @property
    def model_dirs(self) -> list[Path]:
        """Directories searched for `gpurun use NAME` (bare-name lookup)."""
        dirs = self._raw.get("paths", {}).get("model_dirs") or []
        return [Path(d).expanduser() for d in dirs]

    @property
    def state_dir(self) -> Path:
        return Path.home() / ".local" / "state" / "gpu-orchestrator"

    @property
    def share_dir(self) -> Path:
        return Path.home() / ".local" / "share" / "gpu-orchestrator"

    # -- persistence ------------------------------------------------------
    def save(self, data: dict[str, Any] | None = None) -> None:
        """Persist (partial) config to YAML.  Merges with existing."""
        import yaml

        path = _config_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        existing = _load_raw() if path.exists() else {}
        if data:
            existing = _deep_merge(existing, data)

        with open(path, "w") as f:
            yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

    # -- auto-detection ---------------------------------------------------
    def with_auto_detected(self) -> Config:
        """Return a new Config with auto-detected values filled in."""
        import copy

        new = copy.deepcopy(self._raw)

        # Detect proc_pattern and start_cmd from running process
        info = _find_running_llama(self.llama_url, self.llama_port)
        if info:
            if new["llama"]["proc_pattern"] is None:
                new["llama"]["proc_pattern"] = info["proc_pattern"]
            if new["llama"]["start_cmd"] is None:
                new["llama"]["start_cmd"] = info["start_cmd"]
        else:
            # Try to find a binary even if not running
            binary = _find_llama_binary()
            if binary:
                if new["llama"]["proc_pattern"] is None:
                    new["llama"]["proc_pattern"] = re.escape(str(binary))

        return Config(new)


def load() -> Config:
    """Load config from disk with auto-detection applied."""
    raw = _load_raw()
    cfg = Config(raw)
    return cfg.with_auto_detected()


def load_raw() -> dict[str, Any]:
    """Load raw config dict (no auto-detection)."""
    return _load_raw()


# ------------------------------------------------------------------ detection


def _find_llama_binary() -> Path | None:
    """Find llama-server binary by searching known paths, then $PATH."""
    # Check known paths first
    for p in LLAMA_SEARCH_PATHS:
        if p.exists() and p.is_file() and os.access(str(p), os.X_OK):
            return p

    # Check $PATH
    result = shutil.which("llama-server")
    if result:
        return Path(result)

    return None


def _find_running_llama(url: str, port: int) -> dict[str, str] | None:
    """Find a running llama-server by probing the HTTP endpoint.

    Returns dict with 'proc_pattern' and 'start_cmd' or None.
    """
    import urllib.request

    # First, verify the endpoint responds
    try:
        req = urllib.request.Request(f"{url}/v1/models")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                # Endpoint is live — find the process listening on this port
                return _process_from_port(port, url)
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return None


def _process_from_port(port: int, url: str) -> dict[str, str] | None:
    """Find the process listening on `port` and return its info."""
    try:
        # Use psutil to find processes listening on the port
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.pid is not None:
                try:
                    proc = psutil.Process(conn.pid)
                    cmdline = proc.cmdline()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

                # Check if it's a llama-server process
                cmdline_str = " ".join(cmdline)
                if "llama-server" in cmdline_str or "llama.cpp" in cmdline_str:
                    # Build a safe regex pattern from the cmdline
                    # Use the binary path (argv[0]) anchored to avoid false matches
                    if cmdline:
                        binary = cmdline[0]
                        proc_pattern = "^[^ ]*" + re.escape(binary) + ".*"
                    else:
                        proc_pattern = "llama-server"

                    return {
                        "proc_pattern": proc_pattern,
                        "start_cmd": cmdline_str,
                    }
    except (psutil.AccessDenied, OSError):
        pass

    # Fallback: search all processes for llama-server
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"]
            if not cmdline:
                continue
            cmdline_str = " ".join(cmdline)
            if "llama-server" in cmdline_str:
                binary = cmdline[0] if cmdline else "llama-server"
                proc_pattern = "^[^ ]*" + re.escape(binary) + ".*"
                return {
                    "proc_pattern": proc_pattern,
                    "start_cmd": cmdline_str,
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            continue

    return None
