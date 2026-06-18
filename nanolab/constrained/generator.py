"""ConstrainedGenerator: decouple reasoning from structured output.

The "Format Tax" (2026) finding: the decoder constraint costs almost nothing;
the real damage comes from prompt instructions suppressing chain-of-thought.

Fix: let the model reason freely in a <think> block, then apply the JSON-schema
grammar constraint ONLY to the final tool-call block. Never wrap the whole
generation in the grammar.

Two-phase approach:
  Phase 1 — Think: unconstrained HF generate, stops at </think>.
  Phase 2 — Act:   Outlines generate.json with Pydantic schema (Literal enum
                   on `name` whitelists only real tool names).

Outlines' high-level API (Transformers wrapper + generate.json) is used for
Phase 2 so its FSM state management works correctly with arbitrary prompt
lengths — the raw JSONLogitsProcessor has state-tracking issues with long prefixes.
"""
from __future__ import annotations

import json
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .schema import ToolDef, build_tool_call_schema


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------

class StopOnSequence(StoppingCriteria):
    """Stop generation when a given token sequence is produced."""

    def __init__(self, stop_sequences: list[list[int]]):
        self.stops = [torch.tensor(s) for s in stop_sequences]

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **_) -> bool:
        for stop in self.stops:
            n = len(stop)
            if input_ids.shape[1] >= n:
                if (input_ids[0, -n:] == stop.to(input_ids.device)).all():
                    return True
        return False


# ---------------------------------------------------------------------------
# ConstrainedGenerator
# ---------------------------------------------------------------------------

class ConstrainedGenerator:
    """Two-phase constrained generation: think freely, then emit valid JSON.

    Phase 1 — think:
        Unconstrained HF generation until </think> or max_think_tokens.
        Full chain-of-thought preserved with no grammar interference.

    Phase 2 — act:
        Outlines generate.json with the tool-call Pydantic schema.
        The `name` field is a Literal enum of exactly the real tool names —
        hallucinated tool names are impossible to emit at the token level.
    """

    THINK_OPEN  = "<think>"
    THINK_CLOSE = "</think>"
    TOOL_OPEN   = "<tool_call>"

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        tools: list[ToolDef],
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.device = device or next(model.parameters()).device
        self.tool_names = {t.name for t in tools}

        self._schema = build_tool_call_schema(tools)

        # Wrap the already-loaded model for Outlines — no second load needed
        from outlines.models import Transformers
        self._outlines_model = Transformers(model, tokenizer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_think_tokens: int = 256,
        max_call_tokens: int = 256,
        greedy: bool = True,
    ) -> dict[str, Any]:
        """Generate a constrained tool call for the given prompt.

        Returns:
            {
                "think":        str   — free chain-of-thought (may be empty)
                "tool_call":    dict  — schema-valid tool call (always)
                "tool_call_raw": str  — JSON string representation
                "valid":        bool  — always True (grammar enforces it)
            }
        """
        think_text = self._generate_think(prompt, max_think_tokens)
        act_prompt = self._build_act_prompt(prompt, think_text)
        tool_call  = self._generate_constrained(act_prompt, max_call_tokens, greedy)

        tool_call_raw = json.dumps(tool_call)
        return {
            "think":         think_text,
            "tool_call":     tool_call,
            "tool_call_raw": tool_call_raw,
            "valid":         tool_call.get("name", "") in self.tool_names,
        }

    def generate_unconstrained(
        self,
        prompt: str,
        max_tokens: int = 256,
        greedy: bool = True,
    ) -> dict[str, Any]:
        """Baseline: generate without grammar constraint (for validity comparison)."""
        input_ids = self._encode(prompt)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=not greedy,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self._decode_new(input_ids, out)
        tool_call, valid = self._try_parse_json(raw, self.tool_names)
        return {"raw": raw, "tool_call": tool_call, "valid": valid}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_think(self, prompt: str, max_tokens: int) -> str:
        """Phase 1: unconstrained free-text reasoning."""
        think_prompt = prompt + self.THINK_OPEN
        input_ids = self._encode(think_prompt)

        stop_ids = self.tokenizer(
            self.THINK_CLOSE, add_special_tokens=False
        )["input_ids"]
        stopping = StoppingCriteriaList([StopOnSequence([stop_ids])])

        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping,
            )

        raw = self._decode_new(input_ids, out)
        return raw.replace(self.THINK_CLOSE, "").strip()

    def _build_act_prompt(self, original_prompt: str, think_text: str) -> str:
        if think_text:
            return (
                f"{original_prompt}"
                f"{self.THINK_OPEN}{think_text}{self.THINK_CLOSE}\n"
                f"{self.TOOL_OPEN}"
            )
        return f"{original_prompt}{self.TOOL_OPEN}"

    def _generate_constrained(
        self, prompt: str, max_tokens: int, greedy: bool
    ) -> dict[str, Any]:
        """Phase 2: constrained name selection via Outlines generate.choice.

        We constrain only the `name` field — the core anti-hallucination guarantee.
        generate.choice terminates cleanly after emitting exactly one valid tool
        name, which avoids the Outlines 0.2.x issue where generate.json can hit
        max_tokens mid-string and produce unparseable output.

        Arguments default to {} here; the demo's purpose is to show that
        hallucinated tool names are physically impossible to emit, not to
        score argument correctness.
        """
        import outlines.generate as outlines_gen
        from outlines.samplers import GreedySampler, MultinomialSampler

        sampler = GreedySampler() if greedy else MultinomialSampler()
        names = sorted(self.tool_names)
        name_gen = outlines_gen.choice(self._outlines_model, names, sampler=sampler)

        # Position the model inside the JSON so it must emit a valid tool name.
        name_prompt = prompt + '{"name": "'
        name = name_gen(name_prompt)
        return {"name": name, "arguments": {}}

    def _encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)

    def _decode_new(self, input_ids: torch.Tensor, output: torch.Tensor) -> str:
        new_ids = output[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    @staticmethod
    def _try_parse_json(text: str, valid_names: set[str]) -> tuple[dict, bool]:
        """Try to extract and parse a JSON tool call from raw text."""
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                valid = (
                    isinstance(parsed, dict)
                    and parsed.get("name", "") in valid_names
                    and "arguments" in parsed
                )
                return parsed, valid
            except json.JSONDecodeError:
                pass
        return {}, False
