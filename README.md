# nano-llm-lab

A language model, built and aligned end to end. **Stage 1** (this repo, so far): a
decoder-only transformer implemented from scratch in pure PyTorch — every component
(attention, RoPE, RMSNorm, SwiGLU, the training loop, the sampler) is hand-written, no
`nn.Transformer`, no `trl`/`peft` — and trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

Later stages (planned) take the same model through supervised fine-tuning,
parameter-efficient fine-tuning (LoRA/QLoRA), and preference optimization (DPO).

## Table of contents

- [Architecture](#architecture)
- [Setup](#setup)
- [Data](#data)
- [Training](#training)
- [Results](#results)
- [What I learned](#what-i-learned)

## Architecture

_TBD — diagram and module breakdown added once the model is implemented._

## Setup

_TBD_

## Data

_TBD_

## Training

_TBD_

## Results

_TBD — loss curves, samples, throughput, and cost will be added after training._

## What I learned

_TBD_
