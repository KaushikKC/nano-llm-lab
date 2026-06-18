# Tool-SFT LoRA — eval report

Measures name accuracy, schema validity, and name-mentioned (soft) on the
20-prompt holdout set, comparing base Qwen2.5-0.5B vs. tool-SFT LoRA (epoch 4).

> **Note**: `schema_valid` requires a parseable `<tool_call>` JSON block.
> `name_mentioned` (soft) checks whether the correct tool name appears anywhere
> in the raw output — captures partial format learning where the model reasons
> correctly but doesn't yet emit the full structured block.

## Summary

| Model | Name correct | Schema valid | Name mentioned (soft) |
|---|---|---|---|
| Base (no adapter) | 0/20 (0%) | 0/20 (0%) | 13/20 (65%) |
| Tool-SFT LoRA     | 0/20 (0%) | 0/20 (0%) | 8/20 (40%) |

## Per-example results

| # | Category | Expected | Base mentioned | LoRA mentioned |
|---|---|---|---|---|
| tc_001 | single_correct | `read_contract` | ~ | ✗ |
| tc_002 | single_correct | `run_slither` | ~ | ~ |
| tc_003 | single_correct | `search_vulnerability_db` | ✗ | ~ |
| tc_004 | single_correct | `get_contract_abi` | ✗ | ~ |
| tc_005 | single_correct | `get_audit_report` | ~ | ~ |
| tc_006 | abstain | `no_tool_needed` | ~ | ✗ |
| tc_007 | abstain | `no_tool_needed` | ✗ | ✗ |
| tc_008 | abstain | `no_tool_needed` | ~ | ✗ |
| tc_009 | abstain | `no_tool_needed` | ~ | ~ |
| tc_010 | selection | `get_contract_abi` | ~ | ✗ |
| tc_011 | selection | `run_slither` | ✗ | ✗ |
| tc_012 | selection | `get_audit_report` | ~ | ~ |
| tc_013 | selection | `search_vulnerability_db` | ~ | ✗ |
| tc_014 | honest_failure | `read_contract` | ~ | ✗ |
| tc_015 | honest_failure | `no_tool_needed` | ~ | ✗ |
| tc_016 | hallucination_trap | `no_tool_needed` | ~ | ✗ |
| tc_017 | hallucination_trap | `no_tool_needed` | ~ | ✗ |
| tc_018 | hallucination_trap | `no_tool_needed` | ✗ | ~ |
| tc_019 | multi_step_first | `read_contract` | ✗ | ~ |
| tc_020 | termination | `no_tool_needed` | ✗ | ✗ |