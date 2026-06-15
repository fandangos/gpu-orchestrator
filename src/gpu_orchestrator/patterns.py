"""GPU-command pattern matching.

All patterns are ERE (Extended Regular Expressions), matched case-insensitively.
"""

from __future__ import annotations

import re
import shlex

from .config import GPU_PATTERNS


def matches_gpu(command: str, extra_patterns: list[str] | None = None) -> str | None:
    """Check if a command matches any GPU pattern.

    Returns the matched pattern string, or None if no match.
    """
    flags = re.IGNORECASE
    patterns = list(GPU_PATTERNS)
    if extra_patterns:
        patterns.extend(extra_patterns)

    for pattern in patterns:
        if not pattern or not pattern.strip():
            continue
        try:
            if re.search(pattern, command, flags):
                return pattern
        except re.error:
            continue

    return None


def shell_quote(command: str) -> str:
    """Safely quote a command for shell interpolation (single-quote style).

    Handles embedded single quotes by closing, escaping, and reopening.
    """
    return shlex.quote(command)
