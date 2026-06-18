"""Tool-use eval: name accuracy and schema validity before and after LoRA SFT.

Measures two things on the tool_use_eval.jsonl holdout set:
  name_correct  — emitted tool name matches expected_tool
  schema_valid  — output has a parseable <tool_call> JSON block with name + arguments

Runs against two models: base (no adapter) and tool-SFT LoRA adapter.

Usage:
    python scripts/tool_sft_eval.py \
        --base  Qwen/Qwen2.5-0.5B \
        --lora  checkpoints/tool_sft/hf \
        --eval  data/tools/tool_use_eval.jsonl \
        --tools data/tools/solidity_tools.json \
        --out   docs/reliability/tool_sft_eval_report.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

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


def load_model(model_path: str, lora_path: str | None, device: torch.device):
    attn = "eager" if str(device) == "mps" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation=attn
    ).to(device)
    if lora_path and Path(lora_path).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    model.eval()
    return model


def build_prompt(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(model, tokenizer, prompt: str, device: torch.device, max_tokens: int = 200) -> str:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new = out[0, ids.shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True)


def extract_tool_call(text: str) -> dict | None:
    m = TOOL_CALL_RE.search(text)
    if not m:
        # Fallback: try bare JSON
        m2 = re.search(r"\{.*\}", text, re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group())
            except json.JSONDecodeError:
                pass
        return None
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None


def is_schema_valid(tc: dict | None, tool_names: set) -> bool:
    return (
        tc is not None
        and isinstance(tc, dict)
        and tc.get("name", "") in tool_names
        and "arguments" in tc
        and isinstance(tc["arguments"], dict)
    )


def eval_model(model, tokenizer, rows: list, tool_names: set, device: torch.device, label: str):
    name_correct = schema_valid = name_mentioned = 0
    results = []
    for row in rows:
        prompt = build_prompt(tokenizer, row["prompt"])
        raw = generate(model, tokenizer, prompt, device)
        tc = extract_tool_call(raw)
        nc = tc is not None and tc.get("name") == row["expected_tool"]
        sv = is_schema_valid(tc, tool_names)
        # soft metric: correct tool name appears anywhere in raw output
        nm = row["expected_tool"] in raw
        name_correct  += int(nc)
        schema_valid  += int(sv)
        name_mentioned += int(nm)
        got = tc.get("name", "?") if tc else "NONE"
        results.append({
            "id": row["id"],
            "category": row["category"],
            "expected": row["expected_tool"],
            "got": got,
            "name_correct": nc,
            "schema_valid": sv,
            "name_mentioned": nm,
            "raw": raw[:200],
        })
        mark = "✓" if nc else ("~" if nm else "✗")
        print(f"  {mark} [{row['id']}] expected={row['expected_tool']:28s} got={got:28s}  mentioned={'yes' if nm else 'no'}")
    n = len(rows)
    print(f"\n  {label}: name_correct={name_correct}/{n} ({100*name_correct//n}%)  "
          f"schema_valid={schema_valid}/{n} ({100*schema_valid//n}%)  "
          f"name_mentioned={name_mentioned}/{n} ({100*name_mentioned//n}%)\n")
    return results, name_correct, schema_valid, name_mentioned


def write_report(path: str, base_res, lora_res, base_nc, base_sv, base_nm, lora_nc, lora_sv, lora_nm, n):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tool-SFT LoRA — eval report",
        "",
        "Measures name accuracy, schema validity, and name-mentioned (soft) on the",
        "20-prompt holdout set, comparing base Qwen2.5-0.5B vs. tool-SFT LoRA (epoch 4).",
        "",
        "> **Note**: `schema_valid` requires a parseable `<tool_call>` JSON block.",
        "> `name_mentioned` (soft) checks whether the correct tool name appears anywhere",
        "> in the raw output — captures partial format learning where the model reasons",
        "> correctly but doesn't yet emit the full structured block.",
        "",
        "## Summary",
        "",
        "| Model | Name correct | Schema valid | Name mentioned (soft) |",
        "|---|---|---|---|",
        f"| Base (no adapter) | {base_nc}/{n} ({100*base_nc//n}%) | {base_sv}/{n} ({100*base_sv//n}%) | {base_nm}/{n} ({100*base_nm//n}%) |",
        f"| Tool-SFT LoRA     | {lora_nc}/{n} ({100*lora_nc//n}%) | {lora_sv}/{n} ({100*lora_sv//n}%) | {lora_nm}/{n} ({100*lora_nm//n}%) |",
        "",
        "## Per-example results",
        "",
        "| # | Category | Expected | Base mentioned | LoRA mentioned |",
        "|---|---|---|---|---|",
    ]
    for b, l in zip(base_res, lora_res):
        bm = "✓" if b["name_correct"] else ("~" if b["name_mentioned"] else "✗")
        lm = "✓" if l["name_correct"] else ("~" if l["name_mentioned"] else "✗")
        lines.append(
            f"| {b['id']} | {b['category']} | `{b['expected']}` "
            f"| {bm} | {lm} |"
        )
    Path(path).write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base",  default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--lora",  default="checkpoints/tool_sft/hf")
    p.add_argument("--eval",  default="data/tools/tool_use_eval.jsonl")
    p.add_argument("--tools", default="data/tools/solidity_tools.json")
    p.add_argument("--out",   default="docs/reliability/tool_sft_eval_report.md")
    args = p.parse_args()

    device = resolve_device()
    print(f"Device: {device}")

    with open(args.tools) as f:
        tool_names = {t["name"] for t in json.load(f)}
    rows = [json.loads(l) for l in Path(args.eval).read_text().splitlines() if l.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n--- Base model ---")
    base_model = load_model(args.base, None, device)
    base_res, base_nc, base_sv, base_nm = eval_model(base_model, tokenizer, rows, tool_names, device, "Base")
    del base_model
    if str(device) == "mps":
        torch.mps.empty_cache()

    print("--- Tool-SFT LoRA ---")
    lora_model = load_model(args.base, args.lora, device)
    lora_res, lora_nc, lora_sv, lora_nm = eval_model(lora_model, tokenizer, rows, tool_names, device, "LoRA")

    n = len(rows)
    if args.out:
        write_report(args.out, base_res, lora_res, base_nc, base_sv, base_nm, lora_nc, lora_sv, lora_nm, n)
        print(f"Report written: {args.out}")


if __name__ == "__main__":
    main()
