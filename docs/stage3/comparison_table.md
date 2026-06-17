# Stage 3: full FT vs LoRA vs QLoRA — comparison

All three methods use `Qwen/Qwen2.5-0.5B` trained for 10 epochs on the same
66-example Solidity/DeFi SFT dataset (50 train / 16 val / 18 eval), effective
batch size 16, Apple M3 16 GB.

## Comparison table

| Method | Trainable params | % of model | Est. GPU memory | Wall-clock time | Eval score (keyword %) |
|---|---|---|---|---|---|
| Full FT (Stage 2) | 494.0 M | 100.00% | ~5.93 GB | 88.4 min (MPS) | 18.7% |
| LoRA (Stage 3) | 8.8 M | 1.75% | ~1.09 GB | 41.4 min (MPS) | 20.9% |
| QLoRA (Stage 3) | 8.8 M | 1.75% | ~0.33 GB | ~4.7 h (CPU)¹ | TBD² |

¹ bitsandbytes 4-bit kernels are CUDA-only; Apple M3 MPS falls back to CPU.
  Measured ~7 min/step × 40 steps = ~280 min ≈ 4.7 h. On a GPU the time
  would be comparable to LoRA (typically +10–20% overhead for dequantization).

² QLoRA eval to be added once training completes.

## Memory breakdown

| Component | Full FT | LoRA | QLoRA |
|---|---|---|---|
| Base weights | 494 M × 2 B = **988 MB** (bf16) | 494 M × 2 B = **988 MB** (frozen, bf16) | 494 M × 0.5 B = **247 MB** (NF4 4-bit) |
| Adapter weights | — | 8.8 M × 2 B = **18 MB** | 8.8 M × 2 B = **18 MB** |
| Gradients | 494 M × 2 B = **988 MB** | 8.8 M × 2 B = **18 MB** | 8.8 M × 2 B = **18 MB** |
| AdamW m + v | 494 M × 8 B = **3,952 MB** | 8.8 M × 8 B = **70 MB** | 8.8 M × 8 B = **70 MB** |
| **Total** | **~5.93 GB** | **~1.09 GB** | **~0.35 GB** |

## Key takeaways

- **LoRA** reduces trainable parameters by **56×** (from 494 M to 8.8 M) and peak
  memory by **5.4×**, while matching or slightly exceeding full FT eval score
  on this dataset (20.9% vs 18.7%).
- **QLoRA** compresses the base further to 4-bit (NF4), dropping memory to
  **~0.33 GB** — a **18×** reduction vs full FT. The trade-off on this hardware
  is speed: CUDA-only bitsandbytes kernels force CPU fallback on MPS, making
  it ~6.8× slower than LoRA on MPS.
- On a GPU, QLoRA would run at roughly LoRA speed (dequant overhead ~10–20%)
  while cutting VRAM by ~3×. That makes it the preferred method for fine-tuning
  large models (7B+) on consumer GPUs.
- All three methods use the same LoRA hyperparameters (r=16, α=32, all 7
  projection layers). The only difference between LoRA and QLoRA is the base
  model precision.
