# Regularization Techniques

This document describes the regularization knobs available for training in this repo,
how they work, where they live in the code, and how to sweep over them.

Most techniques below are **training-only** (they are no-ops during eval) and are
**disabled by default**, so existing configs are unaffected unless you set them explicitly.
(Weight decay is the exception — it is always active whenever its value is non-zero.)

| Technique | Scope | CLI arg (`single_train_launch.py`) | Config path |
|---|---|---|---|
| Dropout | Any transformer block | `--dropout` *(see note)* | `model.block.dropout` |
| Weight decay | Optimizer | `--weight_decay` | `train_module.optim.weight_decay` |
| Gradient clipping | Train module | `--max_grad_norm` *(see note)* | `train_module.max_grad_norm` |
| Router jitter | MoE router | `--moe_jitter_eps` | `model.block.feed_forward_moe.routers_list.0.jitter_eps` |
| Expert weight normalization | MoE router | `--moe_normalize_expert_weights` | `model.block.feed_forward_moe.routers_list.0.normalize_expert_weights` |
| Expert Output Masking (EOM) | MoE router | `--moe_eom_prob` | `model.block.feed_forward_moe.routers_list.0.eom_prob` |
| Final Output Masking (FOM) | MoE layer | `--moe_fom_prob` | `model.block.feed_forward_moe.fom_prob` |

See also [Related MoE loss knobs](#related-moe-loss-knobs) at the bottom for the
load-balancing loss, router z-loss, and aux-loss-free bias, which shape routing behavior
but are not strictly regularizers.

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

---

## Gradient Clipping

**What it does.** Caps the global norm of the gradients before each optimizer step. Not a
classic regularizer, but it stabilizes training and limits the size of any single update.

**Mechanism.** If the total gradient norm exceeds `max_grad_norm`, all gradients are scaled
down so the norm equals `max_grad_norm`.

**Where it lives.** `TransformerTrainModuleConfig.max_grad_norm`
(`src/olmo_core/train/train_module/transformer/config.py`).

**Defaults.**
- Config default: `None` (no clipping).
- `single_train_launch.py` `build_config` default: `1.0`.

**How to set it.** Grid path is `train_module.max_grad_norm`:

```python
"main_grid": {
    "train_module": {
        "max_grad_norm": [1.0],
    },
},
```

To **disable** clipping in a sweep, set it to `null` (which YAML parses as Python `None`);
`None` written directly becomes the string `"None"` and will not work:

```python
"train_module": {
    "max_grad_norm": ["null"],
},
```

---

## Router Jitter (jitter_eps)

**What it does.** Adds multiplicative noise to the router's input during training, so the
expert routing decision is stochastically perturbed. This discourages the router from
locking too hard onto a single expert and can improve exploration / robustness of routing.

**Mechanism.** Before computing router logits, the input `x` is multiplied element-wise by
uniform noise in `[1 - jitter_eps, 1 + jitter_eps]` (see `MoERouter.jitter` in
`src/olmo_core/nn/moe/router.py`). Active only during training.

**Where it lives.** `MoERouterConfig.jitter_eps` (`src/olmo_core/nn/moe/router.py`), applied
at the very start of `MoERouter.forward`.

**Default.** `None` (no jitter).

**How to set it.** Exposed as a top-level CLI arg, so in `train_sweep.py`:

```python
"main_grid": {
    "moe_jitter_eps": [0.01, 0.1],
},
```

---

## Expert Weight Normalization (normalize_expert_weights)

**What it does.** Normalizes the top-k expert combination weights per token so they have a
fixed norm. For example, `1.0` (L1) makes the selected experts' weights sum to 1;
`2.0` (L2) gives them unit L2 norm. This controls the overall scale of the combined expert
output and can stabilize training.

**Mechanism.** After top-k selection, `expert_weights` is divided by its L-p norm along the
top-k dimension, where `p = normalize_expert_weights` (see `MoERouter.forward` in
`src/olmo_core/nn/moe/router.py`). Applied in both training and eval (it is part of the
routing math, not a stochastic regularizer).

**Where it lives.** `MoERouterConfig.normalize_expert_weights`
(`src/olmo_core/nn/moe/router.py`).

**Default.** `None` (no normalization).

**How to set it.** Exposed as a top-level CLI arg:

```python
"main_grid": {
    "moe_normalize_expert_weights": [1.0, 2.0],
},
```

**Note on ordering with EOM.** Normalization is applied *before* EOM masking, so masking a
combine weight to zero after normalization means the surviving weights will not re-sum to 1.
This is intentional (EOM does no rescaling — see below).

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

---

## Related MoE loss knobs

These are not regularizers in the dropout/decay sense, but they shape routing and are worth
knowing about when tuning MoE runs. All are exposed as top-level CLI args in
`single_train_launch.py`.

| Knob | CLI arg | Config path | Default (launch script) | What it does |
|---|---|---|---|---|
| Load-balancing loss weight | `--moe_lb_loss_weight` | `...feed_forward_moe.lb_loss_weight` | `0.01` | Weight on the auxiliary load-balancing loss that keeps tokens spread evenly across experts. Set `0` to disable. |
| Router z-loss weight | `--moe_z_loss_weight` | `...feed_forward_moe.z_loss_weight` | `0.001` | Penalizes large router logits (encourages numerically stable, less peaky routing). |
| Aux-loss-free bias gamma | `--moe_bias_gamma` | `...routers_list.0.bias_gamma` | `None` | Enables DeepSeek-v3-style bias-based load balancing (per-expert bias nudged each step) instead of / in addition to the aux loss. A reasonable value is ~`0.0001`. |

Note: EOM and FOM are deliberately designed **not** to interfere with the load-balancing
loss (it is computed from the full routing scores/indices), so they can be combined freely
with `lb_loss_weight` tuning.

