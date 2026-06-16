"""Merge a LoRA adapter into the base model weights (or unmerge to verify).

After training with lora_train.py, the adapter sits on top of the frozen base.
Merging folds A and B into each W: W' = W + (lora_alpha/r) * B @ A
The merged model has no inference overhead and can be used exactly like a full FT model.

Usage:
    # Merge adapter into base, save as standalone HF model
    python scripts/merge_adapter.py \\
        --adapter checkpoints/lora/hf \\
        --out     checkpoints/lora/merged

    # Verify: unmerge and compare base output to confirm W is restored
    python scripts/merge_adapter.py \\
        --adapter checkpoints/lora/hf \\
        --out     checkpoints/lora/merged \\
        --verify
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(adapter_path: str, out_path: str, verify: bool = False) -> None:
    print(f"Loading base model from adapter config …")
    # PEFT stores the base model name inside adapter_config.json
    from peft import PeftConfig
    peft_cfg = PeftConfig.from_pretrained(adapter_path)
    base_name = peft_cfg.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.bfloat16)

    print(f"Attaching adapter from {adapter_path} …")
    model = PeftModel.from_pretrained(base, adapter_path)

    # --- optional unmerge verification ---
    if verify:
        print("Verifying unmerge: capturing pre-merge base output …")
        probe_ids = torch.ones(1, 16, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            # disable adapters to get clean base output
            with model.disable_adapter():
                base_logits = model(probe_ids).logits.clone()

        model.merge_adapter()
        with torch.no_grad():
            # re-enable adapter path but now weights are merged
            merged_logits = model(probe_ids).logits.clone()

        model.unmerge_adapter()
        with torch.no_grad():
            with model.disable_adapter():
                restored_logits = model(probe_ids).logits.clone()

        max_diff_merge = (merged_logits - base_logits).abs().max().item()
        max_diff_restore = (restored_logits - base_logits).abs().max().item()
        print(f"  max|merged − base|   = {max_diff_merge:.6f}  (adapter did change output)")
        print(f"  max|restored − base| = {max_diff_restore:.6f}  (should be ~0 if unmerge is clean)")
        # Re-merge for the final save
        model.merge_adapter()
    else:
        print("Merging adapter weights into base …")
        model.merge_adapter()

    # Extract the plain base model (no PEFT wrapper) and save
    merged = model.base_model.model
    print(f"Saving merged model to {out_path} …")
    merged.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="path to saved PEFT adapter (HF format)")
    parser.add_argument("--out",     required=True, help="output path for merged model")
    parser.add_argument("--verify",  action="store_true",
                        help="verify unmerge restores original base weights")
    args = parser.parse_args()
    merge(args.adapter, args.out, args.verify)
