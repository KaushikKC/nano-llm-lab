"""Unit tests for constrained decoding schema module.

No model downloads required — tests only the schema building logic.
"""
import json
import pytest
from typing import get_args

from nanolab.constrained.schema import ToolDef, build_tool_call_schema, tool_schema_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TOOLS = [
    ToolDef(name="read_file",    description="Read a file"),
    ToolDef(name="run_analysis", description="Run analysis"),
    ToolDef(name="no_tool",      description="No tool needed"),
]


# ---------------------------------------------------------------------------
# ToolDef tests
# ---------------------------------------------------------------------------

def test_tooldef_from_dict():
    d = {"name": "read_file", "description": "reads a file", "parameters": {"path": {"type": "string"}}}
    t = ToolDef.from_dict(d)
    assert t.name == "read_file"
    assert t.description == "reads a file"
    assert "path" in t.parameters


def test_tooldef_from_dict_no_params():
    d = {"name": "no_tool", "description": "no tool needed"}
    t = ToolDef.from_dict(d)
    assert t.parameters == {}


def test_tooldef_load_json(tmp_path):
    data = [
        {"name": "tool_a", "description": "desc a"},
        {"name": "tool_b", "description": "desc b"},
    ]
    f = tmp_path / "tools.json"
    f.write_text(json.dumps(data))
    tools = ToolDef.load_json(str(f))
    assert len(tools) == 2
    assert tools[0].name == "tool_a"
    assert tools[1].name == "tool_b"


# ---------------------------------------------------------------------------
# Schema builder tests
# ---------------------------------------------------------------------------

def test_schema_has_name_and_arguments():
    schema = build_tool_call_schema(SAMPLE_TOOLS)
    fields = schema.model_fields
    assert "name" in fields
    assert "arguments" in fields


def test_schema_name_is_exact_enum():
    schema = build_tool_call_schema(SAMPLE_TOOLS)
    annotation = schema.model_fields["name"].annotation
    allowed = set(get_args(annotation))
    assert allowed == {"read_file", "run_analysis", "no_tool"}


def test_schema_rejects_unknown_name():
    schema = build_tool_call_schema(SAMPLE_TOOLS)
    with pytest.raises(Exception):
        schema(name="hallucinated_tool", arguments={})


def test_schema_accepts_valid_name():
    schema = build_tool_call_schema(SAMPLE_TOOLS)
    instance = schema(name="read_file", arguments={"path": "foo.sol"})
    assert instance.name == "read_file"


def test_schema_different_tools_different_enums():
    tools_a = [ToolDef("alpha", "desc"), ToolDef("beta", "desc")]
    tools_b = [ToolDef("gamma", "desc"), ToolDef("delta", "desc")]
    schema_a = build_tool_call_schema(tools_a)
    schema_b = build_tool_call_schema(tools_b)
    # schema_b should reject tool names from schema_a
    with pytest.raises(Exception):
        schema_b(name="alpha", arguments={})
    with pytest.raises(Exception):
        schema_a(name="gamma", arguments={})


def test_schema_empty_arguments():
    schema = build_tool_call_schema(SAMPLE_TOOLS)
    instance = schema(name="no_tool", arguments={})
    assert instance.arguments == {}


def test_empty_tools_raises():
    with pytest.raises(ValueError, match="empty"):
        build_tool_call_schema([])


def test_tool_schema_json_is_valid_json():
    json_str = tool_schema_json(SAMPLE_TOOLS)
    parsed = json.loads(json_str)
    assert "properties" in parsed or "$defs" in parsed or "title" in parsed


def test_tool_schema_json_contains_tool_names():
    json_str = tool_schema_json(SAMPLE_TOOLS)
    for t in SAMPLE_TOOLS:
        assert t.name in json_str
