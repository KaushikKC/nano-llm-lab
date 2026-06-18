"""Tool schema builder for constrained decoding.

Converts a list of ToolDef objects into a Pydantic model whose `name` field
is a Literal enum of exactly the real tool names. Outlines turns this into a
grammar that makes hallucinated tool names (e.g. ListMcpResources, read_directory)
physically impossible to emit — those tokens are masked at the logit level.

Usage:
    tools = [ToolDef(name="read_file", description="...", parameters={...}), ...]
    schema = build_tool_call_schema(tools)   # returns a Pydantic BaseModel class
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, Field, create_model


@dataclass
class ToolDef:
    """Lightweight tool definition — name, description, parameter schema."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ToolDef":
        return cls(
            name=d["name"],
            description=d["description"],
            parameters=d.get("parameters", {}),
        )

    @classmethod
    def load_json(cls, path: str) -> list["ToolDef"]:
        with open(path) as f:
            raw = json.load(f)
        return [cls.from_dict(t) for t in raw]


def build_tool_call_schema(tools: list[ToolDef]) -> type[BaseModel]:
    """Build a Pydantic model for a tool call constrained to the given tools.

    The `name` field becomes a Literal enum of exactly the real tool names,
    so the grammar enforces the whitelist. The `arguments` field is a free
    JSON object (string-keyed, any values) — the per-tool argument type
    checking happens at execution time, not at decoding time.

    Returns a Pydantic BaseModel subclass usable with outlines.generate.json
    or outlines.processors.JSONLogitsProcessor.
    """
    if not tools:
        raise ValueError("tools list must not be empty")

    names = tuple(t.name for t in tools)
    name_literal = Literal[names]  # type: ignore[valid-type]

    descriptions = "\n".join(f"  {t.name}: {t.description}" for t in tools)

    # Bound each argument value to 200 chars so the grammar FSM can't
    # generate unbounded strings that exceed max_tokens before closing.
    ArgValue = Annotated[str, Field(max_length=200)]

    ToolCallModel = create_model(
        "ToolCall",
        name=(name_literal, ...),
        arguments=(dict[str, ArgValue], {}),
        __doc__=f"A single tool call. Available tools:\n{descriptions}",
    )
    return ToolCallModel


def tool_schema_json(tools: list[ToolDef]) -> str:
    """Return the JSON schema string for the tool-call model (useful for logging)."""
    model = build_tool_call_schema(tools)
    return json.dumps(model.model_json_schema(), indent=2)
