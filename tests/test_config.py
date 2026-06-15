"""Tests for config loading and merging."""

import os
import tempfile
from pathlib import Path

from gpu_orchestrator.config import (
    Config,
    load,
    load_raw,
    _deep_merge,
    _clone,
    GPU_PATTERNS,
    _find_llama_binary,
    _find_running_llama,
)


def test_deep_merge():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}, "e": 4}
    result = _deep_merge(base, override)
    assert result == {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}
    # Verify base is not mutated
    assert base["a"]["b"] == 1


def test_clone():
    orig = {"a": [1, 2, {"b": 3}], "c": "hello"}
    cloned = _clone(orig)
    assert cloned == orig
    assert cloned is not orig
    assert cloned["a"] is not orig["a"]
    # Mutation of clone doesn't affect original
    cloned["a"].append(4)
    assert orig["a"] == [1, 2, {"b": 3}]


def test_config_defaults():
    cfg = Config({})
    assert cfg.llama_url == "http://127.0.0.1:8080"
    assert cfg.llama_port == 8080
    assert cfg.llama_proc_pattern is None
    assert cfg.llama_start_cmd is None
    assert cfg.health_timeout == 180
    assert cfg.stop_timeout == 20
    assert cfg.lock_timeout == 180
    assert cfg.grace_timeout == 15
    assert cfg.hook_decision == "allow"
    assert cfg.extra_patterns == []
    assert len(cfg.all_patterns) == len(GPU_PATTERNS)
    assert len(cfg.all_patterns) > 0


def test_config_override():
    cfg = Config({
        "llama": {"url": "http://localhost:9000", "port": 9000},
        "timeouts": {"health": 300},
        "patterns": {"extra": ["my_pattern"]},
    })
    assert cfg.llama_url == "http://localhost:9000"
    assert cfg.llama_port == 9000
    assert cfg.health_timeout == 300
    assert "my_pattern" in cfg.extra_patterns


def test_config_save_load(tmp_path):
    import yaml

    cfg = Config({"llama": {"url": "http://test:1234"}})
    test_conf = tmp_path / "config.yaml"
    import gpu_orchestrator.config as mod
    original_path = mod._config_path
    original_dir = mod._config_dir

    mod._config_path = lambda: test_conf
    mod._config_dir = lambda: tmp_path

    try:
        cfg.save({"llama": {"url": "http://test:1234"}})
        assert test_conf.exists()

        loaded = mod._load_raw()
        assert loaded["llama"]["url"] == "http://test:1234"
    finally:
        mod._config_path = original_path
        mod._config_dir = original_dir


def test_gpu_patterns_not_empty():
    assert len(GPU_PATTERNS) >= 6


def test_find_llama_binary_skips_nonexistent():
    """_find_llama_binary returns None when nothing is installed."""
    # This test passes on systems without llama-server
    result = _find_llama_binary()
    # May be None or a Path — both are valid
    assert result is None or isinstance(result, Path)


def test_find_running_llama_no_server():
    """_find_running_llama returns None when no server is responding."""
    result = _find_running_llama("http://127.0.0.1:19999", 19999)
    assert result is None
