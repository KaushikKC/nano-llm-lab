"""Runtime harness guards for tool-calling agents.

Guards prevent common failure modes that training alone can't fix:
  - EmptyTurnGuard:  catches turns where the model emits nothing useful
  - MaxTurnsGuard:   stops infinite tool-calling loops
  - ToolCallHarness: orchestrates guards + constrained decoding in a single call

These sit between the model and the environment (tool executor). They are
defensive: they catch failures gracefully rather than letting them propagate
as exceptions into the calling application.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HarnessConfig:
    """Runtime configuration for the tool-calling harness."""
    max_turns: int = 10           # hard cap on consecutive tool calls
    empty_turn_limit: int = 2     # max consecutive empty/unparseable turns
    valid_tool_names: set[str] = field(default_factory=set)
    force_constrained: bool = True  # always use constrained decoding for name


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class EmptyTurnGuard:
    """Detects turns where the model produced no parseable tool call.

    Consecutive empty turns often indicate the model is confused or looping.
    After `limit` consecutive empty turns, raises StopIteration to break the
    harness loop.
    """

    def __init__(self, limit: int = 2) -> None:
        self.limit = limit
        self._consecutive = 0

    def reset(self) -> None:
        self._consecutive = 0

    def check(self, tool_call: dict | None) -> None:
        """Call after each turn. Raises StopIteration on threshold breach."""
        if tool_call is None or not tool_call.get("name"):
            self._consecutive += 1
            if self._consecutive >= self.limit:
                raise StopIteration(
                    f"EmptyTurnGuard: {self._consecutive} consecutive empty turns — stopping"
                )
        else:
            self._consecutive = 0


class MaxTurnsGuard:
    """Hard cap on total tool calls in a single session.

    Prevents infinite loops where a model keeps calling tools without
    converging on a final answer or no_tool_needed.
    """

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self._turns = 0

    def reset(self) -> None:
        self._turns = 0

    def check(self) -> None:
        """Call before each turn. Raises StopIteration when limit is reached."""
        self._turns += 1
        if self._turns > self.max_turns:
            raise StopIteration(
                f"MaxTurnsGuard: reached {self.max_turns} tool calls — stopping"
            )

    @property
    def turns_used(self) -> int:
        return self._turns


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class ToolCallHarness:
    """Thin orchestration layer around a ConstrainedGenerator.

    Applies guards and executes a tool-call loop until the model uses
    no_tool_needed, a guard fires, or max_turns is hit.

    Usage:
        harness = ToolCallHarness(generator, cfg, tool_executor)
        result  = harness.run(user_prompt)
    """

    def __init__(
        self,
        generator,          # ConstrainedGenerator instance
        config: HarnessConfig,
        tool_executor=None, # callable(name, arguments) -> str | None
    ) -> None:
        self.generator     = generator
        self.config        = config
        self.tool_executor = tool_executor
        self._max_guard    = MaxTurnsGuard(config.max_turns)
        self._empty_guard  = EmptyTurnGuard(config.empty_turn_limit)

    def run(self, prompt: str) -> dict[str, Any]:
        """Run the tool-call loop for one user prompt.

        Returns:
            {
                "turns":       list of {tool_call, think, tool_result} dicts
                "stopped_by":  "no_tool_needed" | "max_turns" | "empty_turns" | "error"
                "turns_used":  int
            }
        """
        self._max_guard.reset()
        self._empty_guard.reset()

        turns: list[dict] = []
        current_prompt    = prompt
        stopped_by        = "no_tool_needed"

        try:
            while True:
                self._max_guard.check()

                result = self.generator.generate(current_prompt)
                tc     = result.get("tool_call", {})

                self._empty_guard.check(tc if tc.get("name") else None)

                tool_result = None
                if self.tool_executor and tc.get("name") not in ("no_tool_needed", ""):
                    try:
                        tool_result = self.tool_executor(tc["name"], tc.get("arguments", {}))
                    except Exception as exc:
                        tool_result = f"ERROR: {exc}"

                turns.append({
                    "think":       result.get("think", ""),
                    "tool_call":   tc,
                    "tool_result": tool_result,
                })

                if tc.get("name") == "no_tool_needed":
                    stopped_by = "no_tool_needed"
                    break

                # Append tool result to context for next turn
                if tool_result is not None:
                    current_prompt = (
                        f"{current_prompt}\n"
                        f"<tool_result>{tool_result}</tool_result>\n"
                    )

        except StopIteration as exc:
            stopped_by = "max_turns" if "MaxTurns" in str(exc) else "empty_turns"

        return {
            "turns":      turns,
            "stopped_by": stopped_by,
            "turns_used": self._max_guard.turns_used,
        }
