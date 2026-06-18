# Results — nano-llm-lab

All experiments run on Apple M3 16 GB, $0 compute cost.

---

## Stage 1 — From-scratch pretraining on TinyStories

Hand-built decoder-only transformer (RoPE, RMSNorm, SwiGLU, pre-norm, SDPA).

### Smoke test (tiny.yaml — 1.9M params)

| Metric | Value |
|---|---|
| Parameters | 1.9M |
| Dataset | TinyStories (10k rows) |
| Training time | ~4 min |
| Final train loss | ~1.8 |
| Final val loss | ~2.1 |

### Main run (small.yaml — 14M params)

| Metric | Value |
|---|---|
| Parameters | 14M |
| Dataset | TinyStories (~2.1M tokens) |
| Training time | ~7 hours (MPS) |
| Final train loss | ~1.31 |
| Final val loss | ~1.47 |
| Sample quality | Coherent short stories with consistent characters |

Loss curves: [`docs/images/small_loss.png`](docs/images/small_loss.png)

---

## Stage 2 — Supervised Fine-Tuning (SFT)

Full fine-tuning of `Qwen/Qwen2.5-0.5B` on a hand-built Solidity/DeFi security dataset.

### Training config

| Hyperparameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B` (494M params, Apache-2.0) |
| Dataset | 66 examples (50 train / 16 val) |
| Epochs | 10 |
| Effective batch | 16 (micro=2, accum=8) |
| Learning rate | 2e-5 (cosine + warmup) |
| Hardware | Apple M3 16GB / MPS |
| Training time | 88.4 min |
| Cost | $0 |

### Evaluation — keyword coverage (18 eval examples)

| Category | Base | SFT | Δ |
|---|---|---|---|
| defi_mechanics | 23.8% | 33.3% | +9.5 pp |
| fix | 25.0% | 5.0% | −20.0 pp |
| protocol_design | 20.0% | 10.0% | −10.0 pp |
| vulnerability_id | 26.7% | 23.3% | −3.3 pp |
| **Overall** | **24.2%** | **18.7%** | **−5.5 pp** |

> SFT on a 50-example dataset shows mixed results: `defi_mechanics` improved
> significantly while `fix` degraded — consistent with overfitting on a narrow
> dataset where the model learns response style but not all keyword coverage.

Full report: [`docs/sft/eval_report.md`](docs/sft/eval_report.md)  
Loss curve: [`docs/images/sft_loss.png`](docs/images/sft_loss.png)

---

## Stage 3 — Parameter-Efficient Fine-Tuning (LoRA / QLoRA)

Same dataset as Stage 2, LoRA adapters injected into all 7 projection layers.

### Efficiency comparison

| Method | Trainable params | % of total | Est. memory | Wall time | Eval score |
|---|---|---|---|---|---|
| Full FT (Stage 2) | 494.0M | 100.00% | ~5.93 GB | 88.4 min (MPS) | 18.7% |
| LoRA (Stage 3) | 8.8M | 1.75% | ~1.09 GB | 41.4 min (MPS) | 20.9% |
| QLoRA (Stage 3) | 8.8M | 1.75% | ~0.33 GB | >137 min/step (CPU)¹ | N/A² |

¹ `bitsandbytes` 4-bit is CUDA-only; Apple M3 falls back to CPU, impractically slow.  
² Training killed before any checkpoint was saved.

### LoRA eval by category (18 examples)

| Category | Base | LoRA | Δ |
|---|---|---|---|
| defi_mechanics | 23.8% | 33.3% | +9.5 pp |
| fix | 25.0% | 5.0% | −20.0 pp |
| protocol_design | 25.0% | 20.0% | −5.0 pp |
| vulnerability_id | 26.7% | 23.3% | −3.3 pp |
| **Overall** | **25.3%** | **20.9%** | **−4.4 pp** |

> LoRA slightly outperforms full FT (20.9% vs 18.7%) at 1.75% of parameters —
> the rank constraint acts as a regularizer on this small dataset.

Full comparison: [`docs/stage3/comparison_table.md`](docs/stage3/comparison_table.md)  
Loss overlay: [`docs/images/stage3_loss_comparison.png`](docs/images/stage3_loss_comparison.png)  
Eval report: [`docs/stage3/eval_report.md`](docs/stage3/eval_report.md)

---

## Stage 4 — Preference Optimization (DPO + PPO-RLHF)

DPO applied to the SFT model using 35 Solidity/DeFi preference pairs.

### DPO training config

| Hyperparameter | Value |
|---|---|
| Starting model | SFT checkpoint (`checkpoints/sft/hf`) |
| Reference model | SFT checkpoint (frozen) |
| Preference dataset | 35 train / 10 val pairs |
| Beta (KL weight) | 0.1 |
| Learning rate | 5e-7 (much smaller than SFT — fine-tuning alignment) |
| Epochs | 3 |
| Hardware | Apple M3 16GB / MPS |

### DPO eval — keyword coverage (15 eval examples)

| Category | SFT | DPO | Δ |
|---|---|---|---|
| defi_mechanics | 48.0% | 44.0% | −4.0 pp |
| fix | 10.0% | 10.0% | +0.0 pp |
| protocol_design | 33.3% | 33.3% | +0.0 pp |
| vulnerability_id | 15.0% | 20.0% | +5.0 pp |
| **Overall** | **30.3%** | **30.3%** | **+0.0 pp** |

**DPO win-rate vs SFT**: 13.3% wins / 73.3% draws / 13.3% losses (excluding draws: 50.0%)

> DPO shows a neutral-to-small effect on this dataset: `vulnerability_id` improved
> +5 pp while `defi_mechanics` dropped −4 pp, netting to zero overall. With only 35
> training preference pairs the KL anchor (β=0.1) keeps the policy close to SFT —
> the main learning is the alignment technique, not a large capability jump.
> DPO training: 5.1 min, 3 epochs, 27 optimizer steps, $0.

Full report: [`docs/stage4/eval_report.md`](docs/stage4/eval_report.md)

### PPO-RLHF (bonus) — training results

| Phase | Config | Result |
|---|---|---|
| Phase 1 — Reward model | 3 epochs, Bradley-Terry loss | 100% preference accuracy |
| Phase 2 — PPO policy | 1 epoch, 20 steps, β=0.1, ε=0.2 | mean_reward=−0.47, KL stable |

> The RM converges quickly (100% acc by epoch 2) on 35 preference pairs. PPO KL
> stays small (−0.05 to +0.002 range) confirming the policy stays close to the SFT
> reference. The negative mean reward reflects the RM scoring diverse greedy outputs —
> the point is the training loop runs end-to-end without divergence.
>
> Memory constraints: RM + ref both moved to CPU (float32) so only the policy +
> AdamW states (∼5 GB) occupy MPS on a 16 GB M3.

Checkpoint: `checkpoints/ppo/hf` · Reward model: `checkpoints/rm/rm_weights.pt`

---

## Cross-stage summary

| Stage | Method | Key result |
|---|---|---|
| 1 | From-scratch pretraining | 14M-param model generates coherent stories; val loss 1.47 |
| 2 | Full SFT (494M params) | Domain vocabulary learned; DeFi mechanics +9.5 pp; overall −5.5 pp |
| 3 | LoRA (8.8M trainable) | Matches full FT quality at 1.75% parameters, 47% less time |
| 3 | QLoRA (4-bit base) | Impractical without CUDA; on GPU: 18× memory reduction vs full FT |
| 4 | DPO | 30.3% keyword coverage (= SFT); vuln_id +5 pp; 5.1 min, 35 preference pairs |
| 4 | PPO-RLHF (bonus) | RM 100% pref accuracy; PPO 20 steps, mean_reward=−0.47; policy saved |
