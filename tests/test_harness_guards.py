"""Unit tests for harness guards. No model needed."""
import pytest
from nanolab.harness.guards import EmptyTurnGuard, MaxTurnsGuard, HarnessConfig


def test_empty_turn_guard_passes_on_valid_call():
    g = EmptyTurnGuard(limit=2)
    g.check({"name": "read_contract", "arguments": {}})  # should not raise


def test_empty_turn_guard_raises_after_limit():
    g = EmptyTurnGuard(limit=2)
    g.check(None)
    with pytest.raises(StopIteration):
        g.check(None)


def test_empty_turn_guard_resets_on_valid():
    g = EmptyTurnGuard(limit=2)
    g.check(None)
    g.check({"name": "read_contract", "arguments": {}})  # resets count
    g.check(None)  # only 1 empty again — should not raise


def test_empty_turn_guard_manual_reset():
    g = EmptyTurnGuard(limit=2)
    g.check(None)
    g.reset()
    g.check(None)  # should not raise after reset


def test_max_turns_guard_passes_under_limit():
    g = MaxTurnsGuard(max_turns=3)
    g.check()
    g.check()
    g.check()  # exactly at limit — does not raise


def test_max_turns_guard_raises_over_limit():
    g = MaxTurnsGuard(max_turns=2)
    g.check()
    g.check()
    with pytest.raises(StopIteration):
        g.check()


def test_max_turns_guard_turns_used():
    g = MaxTurnsGuard(max_turns=5)
    g.check()
    g.check()
    assert g.turns_used == 2


def test_max_turns_guard_reset():
    g = MaxTurnsGuard(max_turns=1)
    g.check()
    g.reset()
    g.check()  # should not raise after reset


def test_harness_config_defaults():
    cfg = HarnessConfig()
    assert cfg.max_turns == 10
    assert cfg.empty_turn_limit == 2
    assert cfg.force_constrained is True


def test_harness_config_custom():
    cfg = HarnessConfig(max_turns=5, empty_turn_limit=1, valid_tool_names={"read_contract"})
    assert cfg.max_turns == 5
    assert "read_contract" in cfg.valid_tool_names
