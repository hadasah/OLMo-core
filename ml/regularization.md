# Regularization Techniques

This document describes the regularization knobs available for training in this repo,
how they work, where they live in the code, and how to sweep over them.

All four techniques below are **training-only** (they are no-ops during eval) and are
**disabled by default**, so existing configs are unaffected unless you set them explicitly.

| Technique | Scope | CLI arg (`single_train_launch.py`) | Config path |
|---|---|---|---|
| Dropout | Any transformer block | `--dropout` *(see note)* | `model.block.dropout` |
| Weight decay | Optimizer | `--weight_decay` | `train_module.optim.weight_decay` |
| Expert Output Masking (EOM) | MoE router | `--moe_eom_prob` | `model.block.feed_forward_moe.routers_list.0.eom_prob` |
| Final Output Masking (FOM) | MoE layer | `--moe_fom_prob` | `model.block.feed_forward_moe.fom_prob` |

---

## Dropout

**What it does.** Standard dropout applied to the sub-layer outputs (attention and
feed-forward) before they are added back to the residual stream. For MoE blocks, dropout
is applied to the attention and MoE outputs.

**Mechanism.** During training, each activation element is independently zeroed with
probability `p`, and surviving elements are rescaled by `1 / (1 - p)` (standard
inverted dropout, via `nn.Dropout`).

**Where it lives.**
- `TransformerBlockConfig.dropout` (`src/olmo_core/nn/transformer/config.py`) — the config field.
- Applied inside each `ResidualStream` (`src/olmo_core/nn/residual_stream.py`) for dense
  blocks, and as a standalone `nn.Dropout` in `MoETransformerBlock`
  (`src/olmo_core/nn/transformer/block.py`).

**Default.** `None` (equivalent to `0.0`, i.e. no dropout).

**How to set it.** Dropout lives on the model architecture config, so it is applied when the
model is built. In `train_sweep.py` set it as a nested `model` key in the grid:

```python
"main_grid": {
    "model": {
        "block": {
            "dropout": [0.1],
        },
    },
},
```

To sweep multiple values: `"dropout": [0.0, 0.1, 0.2]`.

---

## Weight Decay

**What it does.** Decoupled (AdamW-style) weight decay: shrinks parameters toward zero
each optimizer step, independent of the gradient. Discourages large weights and acts as
an L2-style regularizer.

**Mechanism.** Each step multiplies parameters by `1 - lr * weight_decay` (see
`src/olmo_core/optim/adamw.py`).

**Where it lives.** `AdamWConfig.weight_decay` (`src/olmo_core/optim/adamw.py`), reached
via the train module's optimizer config.

**Defaults.**
- `AdamWConfig` default: `1e-2`.
- `single_train_launch.py` `build_config` default: `0.0`.

**How to set it.** It lives on the optimizer, so the grid path is
`train_module.optim.weight_decay`:

```python
"main_grid": {
    "train_module": {
        "optim": {
            "lr": [4e-4],
            "weight_decay": [0.1, 0.2],
        },
    },
},
```

Related knob: `max_grad_norm` (gradient clipping) lives at `train_module.max_grad_norm`.
Its `single_train_launch.py` default is `1.0`; set it to `null` (YAML `None`) to disable
clipping.

---

## Expert Output Masking (EOM)

**What it does.** Masks the output of *individual experts* for a random fraction of
(token, expert) pairs. In a top-k setup, this can independently skip the 1st, 2nd, ...,
k-th expert for a given token, effectively introducing dynamic capacity by letting the
network skip high-compute expert paths for some inputs.

**Mechanism.** During training, each entry of the top-k combination weight tensor
`expert_weights` (shape `(batch, seq_len, top_k)`) is independently zeroed with
probability `eom_prob`. Zeroing a combination weight removes that expert's contribution
to the token's output.

- Masking is **independent per (token, top-k slot)** — with top-2, either or both experts
  can be dropped for a token.
- **No rescaling** is applied to surviving weights (unlike dropout). This matches the
  "skip the compute path" / dynamic-capacity intent.
- **Routing stability:** EOM only touches `expert_weights`. The auxiliary
  load-balancing loss and router z-loss are computed from the full routing
  `scores` / `logits` / `expert_indices`, which are untouched — so masking does not
  interfere with the router's ability to keep a balanced token distribution.
- Intended to be applied **on top of** dropout, not as a replacement.

**Where it lives.**
- `MoERouterConfig.eom_prob` (`src/olmo_core/nn/moe/router.py`) — config field.
- Applied in `MoERouter.forward` right after top-k selection / normalization.

**Default.** `None` (disabled).

**How to set it.** Exposed as a top-level CLI arg, so in `train_sweep.py` it is a simple
`main_grid` key:

```python
"main_grid": {
    "moe_eom_prob": [0.1, 0.2],
},
```

---

## Final Output Masking (FOM)

**What it does.** A simpler, more generic alternative to EOM. Instead of masking
individual experts, it masks the *combined* output of the MoE layer for a random fraction
of tokens. Because it targets the final stage of the layer, the same idea generalizes to
dense models (not just MoE).

**Mechanism.** During training, the combined MoE output (shape `(batch, seq_len, d_model)`)
is masked per token: each token's entire output vector is independently zeroed with
probability `fom_prob` (a mask of shape `(batch, seq_len, 1)` broadcast over `d_model`).

- Masking is **per token** (the whole `d_model` vector is dropped together).
- **No rescaling** is applied.
- Applied at the single output choke point of the MoE layer, so it covers all variants
  (default and dropless, expert-parallel and not, including the shared-MLP contribution).
- Intended to be applied **on top of** dropout.

**Where it lives.**
- `MoEConfig.fom_prob` (`src/olmo_core/nn/moe/moe.py`) — config field.
- Applied in `MoEBase.forward` after the expert outputs are combined.

**Default.** `None` (disabled).

**How to set it.** Exposed as a top-level CLI arg, so in `train_sweep.py`:

```python
"main_grid": {
    "moe_fom_prob": [0.1],
},
```

---

## EOM vs. FOM at a glance

| | EOM | FOM |
|---|---|---|
| Granularity | per (token, expert) | per token (whole layer output) |
| Level | individual expert combine weight | combined MoE output |
| Applies to dense models? | No (MoE-specific) | Conceptually yes (currently wired in MoE only) |
| Rescaling | None | None |
| Affects load-balancing loss? | No | No |

Both can be combined with each other and with dropout. All values are probabilities in
`[0, 1]`; `None` (the default) disables the technique.
