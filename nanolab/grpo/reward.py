"""Ternary reward for tool-use GRPO.

+1  — model picked the correct tool (name matches expected_tool)
 0  — model abstained with no_tool_needed (honest uncertainty)
-1  — model hallucinated a tool name not in the whitelist

This reward structure punishes confident fabrication more than honest
uncertainty, which is the alignment goal: a model that says "I don't know
which tool to use" is less harmful than one that invents a fake tool.
"""
from __future__ import annotations

import json
import re

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def score_tool_call(
    generated: str,
    expected_tool: str,
    valid_tool_names: set[str],
) -> float:
    """Return the ternary reward for a single generated response.

    Args:
        generated:        Raw text output from the model.
        expected_tool:    The correct tool name for this prompt.
        valid_tool_names: The complete whitelist of known tool names.

    Returns:
        +1.0 if the model emitted the correct tool name.
         0.0 if the model abstained (no_tool_needed).
        -1.0 if the model emitted a hallucinated / wrong tool name.
    """
    m = TOOL_CALL_RE.search(generated)
    if not m:
        # No structured output at all — treat as abstain
        return 0.0

    try:
        tc = json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        # Malformed JSON — can't determine intent, penalise lightly
        return -1.0

    name = tc.get("name", "")

    if name == expected_tool:
        return 1.0                          # correct tool
    if name == "no_tool_needed":
        return 0.0                          # honest abstention
    if name not in valid_tool_names:
        return -1.0                         # hallucinated tool name
    return -1.0                             # wrong (but valid) tool


def ternary_reward(
    batch_generated: list[str],
    batch_expected: list[str],
    valid_tool_names: set[str],
) -> list[float]:
    """Vectorised ternary reward over a batch."""
    return [
        score_tool_call(gen, exp, valid_tool_names)
        for gen, exp in zip(batch_generated, batch_expected)
    ]
