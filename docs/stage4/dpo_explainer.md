# DPO — why it works without a separate reward model

## The problem DPO solves

Classic RLHF needs three separate training phases: supervised fine-tuning → reward model
training → PPO policy optimization. The reward model is a full neural network you have
to train on human preference labels, and PPO is notoriously unstable (clipping,
advantage normalization, KL penalties — lots of moving parts).

DPO (Rafailov et al. 2023) collapses phases 2 and 3 into a single supervised loss.
No reward model, no PPO, no rollouts.

## The key insight

In standard RLHF, the optimal policy under a KL-constrained reward objective has a
closed-form solution:

```
π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)
```

This means you can rearrange: given any policy π and reference π_ref, the implicit
reward they define is:

```
r(x,y) = β · log( π(y|x) / π_ref(y|x) ) + β · log Z(x)
```

The partition function `Z(x)` cancels when you compute the *difference* in reward
between two responses for the same prompt:

```
r(x, y_w) - r(x, y_l) = β · [ log π(y_w|x)/π_ref(y_w|x)
                               - log π(y_l|x)/π_ref(y_l|x) ]
```

A Bradley-Terry preference model says the probability of preferring y_w over y_l is
`σ(r_w - r_l)`. Substituting the expression above and taking the negative log-likelihood
gives the DPO loss directly:

```
L_DPO = -E[ log σ( β · (log π_θ(y_w|x)/π_ref(y_w|x)
                       - log π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

## What this means concretely

- **The log-ratio IS the reward.** You never train a separate reward model.
  The policy's own probability of generating a response, divided by the reference's
  probability, implicitly encodes how much that response was preferred.

- **β controls how far from reference you allow.** Small β (we used 0.1) keeps
  the policy close to SFT — it can only move as far as the preference data justifies.
  Large β loosens the KL constraint and risks reward hacking.

- **The reference model is frozen SFT.** It provides the baseline distribution.
  Without it the loss has no anchor — the policy could collapse by making
  log π(y_w) → ∞ for any y_w regardless of content.

- **Training signal**: loss decreases when the policy increases its log-ratio on
  chosen responses and decreases it on rejected ones. The `reward_margin` metric
  we log is `β · (log_ratio_w - log_ratio_l)` — positive means the policy correctly
  ranks chosen over rejected.

## Why not PPO?

PPO requires sampling from the policy during training (rollouts), scoring those samples
with the reward model, and then computing policy gradient updates — all online. DPO
is purely supervised: forward pass on the fixed (chosen, rejected) pairs, compute
log-probs, backprop. No rollouts, no value network, no clipping tuning.

The tradeoff: DPO can only train on offline preference pairs you already have. PPO can
explore and improve beyond what's in the dataset. For most practical alignment tasks
on small models, DPO's simplicity wins.

## Our results

With 35 preference pairs (3 epochs, β=0.1, lr=5e-7):

| Category | SFT → DPO |
|---|---|
| vulnerability_id | 15% → 20% (+5 pp) |
| defi_mechanics | 48% → 44% (−4 pp) |
| fix / protocol_design | unchanged |
| **Overall** | **30.3% → 30.3%** |

The flat overall score is expected: 35 preference pairs is a tiny signal relative to
a 494M-parameter model. The β=0.1 KL anchor keeps the policy very close to SFT, which
is the right behaviour — we're demonstrating the technique, not overfitting on 35 examples.
