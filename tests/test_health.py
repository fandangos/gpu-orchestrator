"""Tests for health checking."""

from gpu_orchestrator.health import health_ok, models_ok, wait_for_health


def test_health_ok_unreachable():
    """health_ok returns False for unreachable URL."""
    assert health_ok("http://127.0.0.1:19999/health") is False


def test_models_ok_unreachable():
    """models_ok returns False for unreachable URL."""
    assert models_ok("http://127.0.0.1:19999/v1/models") is False


def test_wait_for_health_timeout():
    """wait_for_health returns False when server never comes up."""
    result = wait_for_health("http://127.0.0.1:19999", timeout=1, interval=1)
    assert result is False
