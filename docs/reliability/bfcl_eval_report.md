# BFCL-style offline eval report

Full production stack: constrained decoding (Section 1) + harness guards (Section 4).

## Summary

| Metric | Score |
|---|---|
| name_acc (overall) | 18/20 (90%) |
| schema_valid | 20/20 (100%) |
| abstain_acc | 9/9 (100%) |
| trap_resist | 3/3 (100%) |

## Per-example

| # | Category | Expected | Got | ✓? |
|---|---|---|---|---|
| tc_001 | single_correct | `read_contract` | `read_contract` | ✓ |
| tc_002 | single_correct | `run_slither` | `run_slither` | ✓ |
| tc_003 | single_correct | `search_vulnerability_db` | `search_vulnerability_db` | ✓ |
| tc_004 | single_correct | `get_contract_abi` | `get_contract_abi` | ✓ |
| tc_005 | single_correct | `get_audit_report` | `get_audit_report` | ✓ |
| tc_006 | abstain | `no_tool_needed` | `no_tool_needed` | ✓ |
| tc_007 | abstain | `no_tool_needed` | `no_tool_needed` | ✓ |
| tc_008 | abstain | `no_tool_needed` | `no_tool_needed` | ✓ |
| tc_009 | abstain | `no_tool_needed` | `no_tool_needed` | ✓ |
| tc_010 | selection | `get_contract_abi` | `run_slither` | ✗ |
| tc_011 | selection | `run_slither` | `run_slither` | ✓ |
| tc_012 | selection | `get_audit_report` | `get_audit_report` | ✓ |
| tc_013 | selection | `search_vulnerability_db` | `search_vulnerability_db` | ✓ |
| tc_014 | honest_failure | `read_contract` | `no_tool_needed` | ✗ |
| tc_015 | honest_failure | `no_tool_needed` | `no_tool_needed` | ✓ |
| tc_016 | hallucination_trap | `no_tool_needed` | `no_tool_needed` | ✓ |
| tc_017 | hallucination_trap | `no_tool_needed` | `no_tool_needed` | ✓ |
| tc_018 | hallucination_trap | `no_tool_needed` | `no_tool_needed` | ✓ |
| tc_019 | multi_step_first | `read_contract` | `read_contract` | ✓ |
| tc_020 | termination | `no_tool_needed` | `no_tool_needed` | ✓ |