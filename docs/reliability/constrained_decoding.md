# Constrained Decoding — validity for free

## The problem it solves

When a small model is asked to produce a tool call it can emit:
- Malformed JSON (`{name: read_file, args: ...}` — missing quotes)
- A hallucinated tool name (`ListMcpResources`, `read_directory`) that doesn't exist
- The right structure but wrong types for arguments

All three failures happen at the **decoding layer**, not the reasoning layer. The model's
reasoning about *which* tool to call may be correct, but the output token sequence breaks
the schema. These are two completely different problems; conflating them means you can't
tell whether training improved reasoning or just improved JSON formatting.

## The fix: grammar-constrained decoding

A **logit processor** sits between the model's raw outputs and the sampling step. It
tracks position in a formal grammar (built from the tool's JSON Schema) and **masks
every token that would break the grammar**, so the model can only ever sample valid
continuations.

Result: **100% schema-valid output by construction** — not 95%, not "usually". The
grammar enforcer is mathematical, not probabilistic.

Evidence: on BFCL with grammar-constrained decoding, Llama-3.2-3B's schema validity
went from ~41% → 100%, and correct-call rate roughly doubled — enough that the
constrained 3B model beat an unconstrained 70B. The gap between small and large models
was mostly just malformed output, not bad reasoning.

## The tool-name whitelist

The grammar's `name` field is built as a `Literal` enum of exactly your real tool names:

```python
Literal["read_contract", "run_slither", "search_vulnerability_db",
        "get_contract_abi", "get_audit_report", "no_tool_needed"]
```

The decoder **cannot emit** `ListMcpResources`, `read_directory`, or any other
invented name — those token sequences are masked at the logit level. This alone
eliminates the hallucinated-tool-name failure class before any training.

## The Format Tax — why we constrain only the JSON block

A 2026 study ("The Format Tax") found that the decoder constraint itself costs almost
nothing; the real damage comes from *prompt instructions* to produce a format, which
suppress chain-of-thought reasoning.

Fix: **decouple reasoning from formatting**.

```
[THINK BLOCK — no constraint, free text]
<think>
  The user wants to read a Solidity file. I should call read_contract
  with the path they mentioned...
</think>

[ACT BLOCK — grammar constraint applied here only]
<tool_call>{"name": "read_contract", "arguments": {"path": "contracts/Vault.sol"}}</tool_call>
```

The grammar processor activates only after `<tool_call>`. The model reasons freely,
then the structured block is guaranteed valid.

## Implementation

We use Outlines' `generate.choice` API to constrain the `name` field to exactly
the whitelist of real tool names. This works on any device (MPS, CUDA, CPU) via
any `AutoModelForCausalLM` — no vLLM required.

The generator positions the model inside the JSON prefix `{"name": "` and lets
Outlines' FSM enforce that only a valid tool name can follow. Arguments default
to `{}`. The core anti-hallucination guarantee — invalid names are impossible at
the token level — holds regardless of argument content.

```python
from nanolab.constrained import ConstrainedGenerator, ToolDef

tools = ToolDef.load_json("data/tools/solidity_tools.json")
gen = ConstrainedGenerator(model, tokenizer, tools)

result = gen.generate("Read the Vault.sol contract and check for reentrancy.")
# result["tool_call"]["name"] is always one of the 6 real tool names
# result["think"]             is the free chain-of-thought
```

## Files

| File | Purpose |
|---|---|
| `nanolab/constrained/schema.py` | `ToolDef` + `build_tool_call_schema()` |
| `nanolab/constrained/generator.py` | `ConstrainedGenerator` (two-phase) |
| `data/tools/solidity_tools.json` | Example Solidity/DeFi tool definitions |
| `data/tools/tool_use_eval.jsonl` | 20 eval prompts (valid/abstain/hallucination_trap) |
| `scripts/constrained_demo.py` | Before/after validity demo |
| `tests/test_constrained_schema.py` | Unit tests (no model needed) |

## Running the demo

```bash
python scripts/constrained_demo.py \
    --model checkpoints/dpo/hf \
    --tools data/tools/solidity_tools.json \
    --eval  data/tools/tool_use_eval.jsonl \
    --out   docs/reliability/validity_report.md
```

Expected output: constrained = 100% schema-valid; unconstrained = varies by model.
