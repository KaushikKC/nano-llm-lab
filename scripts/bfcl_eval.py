"""BFCL-style offline eval harness — Stage 5 Section 5.

Runs the tool_use_eval.jsonl holdout set through the full production stack:
  constrained decoding (Section 1) + harness guards (Section 4)

Reports BFCL-compatible metrics:
  name_acc     — correct tool name (matches expected_tool)
  schema_valid — output is a parseable <tool_call> JSON block
  abstain_acc  — correct abstention when expected_tool == no_tool_needed
  trap_resist  — correctly refuses hallucination_trap prompts

This is an offline subset eval, not the full BFCL v4 benchmark.
Use it to track regression across training rungs.

Usage:
    python scripts/bfcl_eval.py \
        --model checkpoints/tool_dpo/hf \
        --tools data/tools/solidity_tools.json \
        --eval  data/tools/tool_use_eval.jsonl \
        --out   docs/reliability/bfcl_eval_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanolab.constrained import ConstrainedGenerator, ToolDef
from nanolab.harness import HarnessConfig, ToolCallHarness


SYSTEM = (
    "You are a Solidity smart-contract security assistant with access to these tools: "
    "read_contract, run_slither, search_vulnerability_db, get_contract_abi, "
    "get_audit_report, no_tool_needed. "
    "Respond with <think>reasoning</think>\n"
    "<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>"
)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_prompt(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_eval(args) -> None:
    device = resolve_device()
    print(f"Device: {device}")

    tools = ToolDef.load_json(args.tools)
    tool_names = {t.name for t in tools}
    rows = [json.loads(l) for l in Path(args.eval).read_text().splitlines() if l.strip()]

    print(f"Loading model: {args.model}")
    attn = "eager" if str(device) == "mps" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=attn
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.\n")

    gen = ConstrainedGenerator(model, tokenizer, tools, device)
    cfg = HarnessConfig(max_turns=1, valid_tool_names=tool_names, force_constrained=True)
    harness = ToolCallHarness(gen, cfg, tool_executor=None)

    results = []
    counts = {"name_acc": 0, "schema_valid": 0, "abstain_acc": 0, "trap_resist": 0}
    abstain_total = trap_total = 0

    for i, row in enumerate(rows, 1):
        prompt  = build_prompt(tokenizer, row["prompt"])
        outcome = harness.run(prompt)
        tc      = outcome["turns"][0]["tool_call"] if outcome["turns"] else {}
        name    = tc.get("name", "")
        exp     = row["expected_tool"]
        cat     = row["category"]

        name_acc    = name == exp
        schema_valid = bool(name and "arguments" in tc and isinstance(tc["arguments"], dict))

        counts["name_acc"]    += int(name_acc)
        counts["schema_valid"] += int(schema_valid)

        if exp == "no_tool_needed":
            abstain_total += 1
            counts["abstain_acc"] += int(name_acc)

        if cat == "hallucination_trap":
            trap_total += 1
            counts["trap_resist"] += int(name == "no_tool_needed")

        mark = "✓" if name_acc else "✗"
        print(f"  {mark} [{row['id']}] {cat:20s}  expected={exp:28s}  got={name}")

        results.append({
            "id": row["id"], "category": cat, "expected": exp,
            "got": name, "name_acc": name_acc, "schema_valid": schema_valid,
        })

    n = len(rows)
    print(f"\n{'='*60}")
    print(f"BFCL-STYLE OFFLINE EVAL  ({n} prompts)")
    print(f"{'='*60}")
    print(f"  name_acc     : {counts['name_acc']}/{n} ({100*counts['name_acc']//n}%)")
    print(f"  schema_valid : {counts['schema_valid']}/{n} ({100*counts['schema_valid']//n}%)")
    if abstain_total:
        print(f"  abstain_acc  : {counts['abstain_acc']}/{abstain_total} ({100*counts['abstain_acc']//abstain_total}%)")
    if trap_total:
        print(f"  trap_resist  : {counts['trap_resist']}/{trap_total} ({100*counts['trap_resist']//trap_total}%)")

    if args.out:
        _write_report(args.out, results, counts, n, abstain_total, trap_total)
        print(f"\nReport written: {args.out}")


def _write_report(path, results, counts, n, abstain_total, trap_total):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BFCL-style offline eval report",
        "",
        "Full production stack: constrained decoding (Section 1) + harness guards (Section 4).",
        "",
        "## Summary",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| name_acc (overall) | {counts['name_acc']}/{n} ({100*counts['name_acc']//n}%) |",
        f"| schema_valid | {counts['schema_valid']}/{n} ({100*counts['schema_valid']//n}%) |",
    ]
    if abstain_total:
        lines.append(f"| abstain_acc | {counts['abstain_acc']}/{abstain_total} ({100*counts['abstain_acc']//abstain_total}%) |")
    if trap_total:
        lines.append(f"| trap_resist | {counts['trap_resist']}/{trap_total} ({100*counts['trap_resist']//trap_total}%) |")

    lines += [
        "",
        "## Per-example",
        "",
        "| # | Category | Expected | Got | ✓? |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        m = "✓" if r["name_acc"] else "✗"
        lines.append(f"| {r['id']} | {r['category']} | `{r['expected']}` | `{r['got']}` | {m} |")

    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/tool_dpo/hf")
    p.add_argument("--tools", default="data/tools/solidity_tools.json")
    p.add_argument("--eval",  default="data/tools/tool_use_eval.jsonl")
    p.add_argument("--out",   default="docs/reliability/bfcl_eval_report.md")
    run_eval(p.parse_args())
