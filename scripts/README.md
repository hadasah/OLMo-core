# Slicing and Dicing Code

This repository contains code for Slicing and Dicing: Configuring Optimal Mixture-of-Experts Models.

Training and evaluation code is built on top of a hard fork from (OLMo-core)[].

---

## Setup

```bash
ENV_NAME=sandd_olmo_core
SANDD_PATH=$YOUR_DIR/slicing_and_dicing
OLMO_CORE_PATH=$SANDD_PATH/OLMo-core

module load cuda/12.6.3
mamba create -n $ENV_NAME python=3.11
mamba activate $ENV_NAME

git clone https://github.com/hadasah/slicing_and_dicing.git
cd slicing_and_dicing/OLMo-core
pip install -e .[all]

# Optional but recommended for fast MoE kernels.
git clone --recursive https://github.com/sneha-rk/grouped_gemm
cd grouped_gemm
GROUPED_GEMM_CUTLASS=1 pip install .
```

## Data

We use the pretraining data mix from OLMoE. Follow the (instructions)[https://github.com/allenai/OLMoE/blob/main/README.md#pretraining] in their repo to download and preprocess the data.

## Running a single job from command line

Our pretraining jobs may be launched with `scripts/single_train_launch.py`:

```bash
torchrun --nproc-per-node=1 scripts/single_train_launch.py my_run_name \
    --model_name=olmo2_ml_80M \
    --moe_num_experts_list=32 \
    --moe_hidden_multipliers_list=0.25 \
    --moe_router_top_ks_list=4 \
    --moe_generalist_hidden_multiplier=0 \
    --moe_type=dropless \
    --train_datamix_name= \
    --train_module.optim.lr=4e-4
```

The positional `run_name` argument becomes the W&B run name and the subdirectory under `save_root`.

---

## Running a single eval-only job from command line

```bash
torchrun --nproc-per-node=1 scripts/single_eval_launch.py my_run_name \
    --model_name=olmo2_ml_80M \
    --moe_num_experts_list=32 \
    --moe_hidden_multipliers_list=0.25 \
    --moe_router_top_ks_list=4 \
    --moe_generalist_hidden_multiplier=0 \
    --moe_type=dropless \
    --train_datamix_name= \
    --train_module.optim.lr=4e-4
```

This run name should be identical to what you specified for `scripts/single_train_launch.py`. 
The script will look for the checkpoint under `{save_dir}/{run_name}`.

---

## MoE configuration

A variety of MoE architecture configuration choices may be specified through command line with `single_train_launch.py`:

| Flag                                  | Type           | Default     | Meaning                                                                       |
|---------------------------------------|----------------|-------------|-------------------------------------------------------------------------------|
| `--moe_num_experts_list`              | csv ints       | `32,64`     | Experts per MoE layer. `1` means dense (no MoE). Multi-value = heterogeneous MoE (different expert counts per MoE group within one layer). |
| `--moe_hidden_multipliers_list`       | csv floats     | `1024,2048` | Expert hidden-size multipliers (multiplied by `d_model`). Length must match `--moe_num_experts_list`. |
| `--moe_router_top_ks_list`            | csv ints       | `4,8`       | Top-k per router. Length must match `--moe_num_experts_list`.                 |
| `--moe_generalist_hidden_multiplier`  | float          | `1`         | Hybrid MoE: dense FFN width multiplier (0 = pure MoE, no dense path).         |
| `--moe_type`                          | `default` / `dropless` | `default` | "dropless" uses uncapped routing (no token drops); "default" uses capacity factor.    |
| `--moe_bias_gamma`                    | float or None  | None        | DeepSeek-v3-style router bias adjustment (γ).                                 |
| `--moe_z_loss_weight`                 | float          | `0.001`     | Router z-loss (entropy regularization) weight.                                |
| `--moe_lb_loss_weight`                | float          | `0.01`      | Load-balancing loss weight. Pass `0` to disable.                              |

Additional flags can be easily added for a variety of choices, such as learning rate or weight decay. Those are exposed in the config constructor calls.

### Examples

**Dense baseline** (no MoE):

```bash
--moe_num_experts_list=1
```

**Homogeneous MoE** 32 experts, top-4, dropless:

```bash
--moe_num_experts_list=32 --moe_hidden_multipliers_list=0.25 \
--moe_router_top_ks_list=4 --moe_generalist_hidden_multiplier=0 \
--moe_type=dropless
```

**MoE w/ Generalist** (32 experts + a dense FFN with multiplier 0.5):

```bash
--moe_num_experts_list=32 --moe_hidden_multipliers_list=0.5 \
--moe_router_top_ks_list=2 --moe_generalist_hidden_multiplier=0.5
```

**Heterogeneous MoE** (two groups per layer, 8 small experts + 32 small experts):

```bash
--moe_num_experts_list=8,32 --moe_hidden_multipliers_list=0.25,0.125 \
--moe_router_top_ks_list=2,4 --moe_generalist_hidden_multiplier=0
```

### Available model sizes

`olmo2_ml_10M`, `olmo2_ml_20M`, `olmo2_ml_50M`, `olmo2_ml_80M`, `olmo2_ml_100M`, `olmo2_ml_110M`, `olmo2_ml_200M`, `olmo2_ml_300M`, `olmo2_ml_500M` (see `MODEL_CONFIG_LOOKUP` in `single_train_launch.py`). Each has Chinchilla-ish defaults for `lr` and `max_duration.value` in `MODEL_HP_DEFAULTS` (`constants.py`).

---

## Data and repetition

| Flag                       | Type    | Default          | Meaning                                                                |
|----------------------------|---------|------------------|------------------------------------------------------------------------|
| `--train_datamix_name`     | str     | `OLMoE_mix_0824` | Training mix (e.g. `OLMoE_mix_0824`).         |
| `--valid_datamix_name`     | str     | `v3_small_ppl_validation` | Validation mix.                                              |
| `--unique_data_fraction`   | float   | `1.0`            | Fraction of source paths used. `0.25` + same `max_duration` = 4× data repetition. |
| `--num_repetitions`        | int     | `1`              | Expected repetition factor — only used for naming/tagging.             |
| `--sequence_length`        | int     | `2048`           | Token sequence length.                                                 |
| `--global_batch_size`      | int     | `512`            | Sequences per global step.                                             |
| `--per_gpu_batch_size`     | int     | `16`             | Sequences per GPU.                                                     |

---

## Optimization, scheduling, misc

Any field of the built `ExperimentConfig` can be overridden from the CLI via dotted-path flags. Common ones:

```bash
--train_module.optim.lr=4e-4
--train_module.optim.weight_decay=0.1
--train_module.scheduler.warmup_steps=2000
--trainer.max_duration.value=1600000000   # tokens
--trainer.save_interval=200
```

These get captured by `parse_known_args()` and merged into the config via `ExperimentConfig.merge(overrides)`.


