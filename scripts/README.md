# Slicing and Dicing — Training & Eval Scripts

This directory contains the launch scripts for the *Slicing and Dicing: Configuring Optimal Mixture-of-Experts Models* paper. The training/eval stack is a hard fork of [OLMo-core](https://github.com/allenai/OLMo-core); see the top-level `README.md` for the underlying library docs.

- `single_train_launch.py` — single-node pretraining entry point.
- `single_eval_launch.py` — eval-only entry point that loads a checkpoint and runs the same LM + downstream evaluators that fire during training.
- `data/` — shell helpers for downloading and preprocessing the OLMoE data mixes. **Provided as-is**: they contain cluster-specific paths and will need to be edited to your environment before they run.

---

## Setup

```bash
ENV_NAME=sandd_olmo_core
SANDD_PATH=$YOUR_DIR/slicing_and_dicing
OLMO_CORE_PATH=$SANDD_PATH/OLMo-core-ml

# On a SLURM-style cluster you may also need: `module load cuda/12.6.3`
mamba create -n $ENV_NAME python=3.11
mamba activate $ENV_NAME

git clone https://github.com/hadasah/slicing_and_dicing.git $SANDD_PATH
cd $OLMO_CORE_PATH
pip install -e .[all]

# Optional but recommended for fast MoE kernels.
git clone --recursive https://github.com/sneha-rk/grouped_gemm
cd grouped_gemm
GROUPED_GEMM_CUTLASS=1 pip install .
```

### Environment variables / accounts

| Service | Variable / setting | Notes |
|---|---|---|
| Weights & Biases | `WANDB_API_KEY`; `--wandb_entity` / `--wandb_project` (or edit `USER_PROJECT_SPECS` in the launch script) | Enabled by default in `single_train_launch.py`. |
| Comet | `COMET_API_KEY` | Disabled by default. Set `enabled=True` on the `CometCallback` in the launch script to use. |
| HuggingFace | `HF_TOKEN` | Only required if a data mix or tokenizer download needs it. |

### `USER_PROJECT_SPECS`

Each launch script has a `USER_PROJECT_SPECS` dict near the top. Fill these in once before your first run so you don't have to pass them on every command line:

| Key | Meaning |
|---|---|
| `DEFAULT_SAVE_PATH` | Parent directory under which `{run_name}/` checkpoints and configs are written. Defaults to `<repo>/models`. |
| `DATAROOT` | Root URL or directory containing the **training** data mix shards. |
| `VALID_DATA_DIR` | Root URL or directory containing the **validation** mix shards. |
| `DATA_WORK_DIR` | Local scratch directory used to materialize/cached the data mix. |
| `WANDB_ENTITY` / `WANDB_PROJECT` | W&B entity and project for logging. |
| `PROJECT_DIR` | Repo root; auto-derived from the script's path. |
| `NAME_KEYS` | Unused at the moment; kept for forward compatibility. |

CLI flags (e.g. `--data_root`) override the matching entry at run time.

---

## Data

We use the pretraining data mix from OLMoE. Follow the [instructions](https://github.com/allenai/OLMoE/blob/main/README.md#pretraining) in their repo to download and preprocess the data, or adapt the shell scripts in `scripts/data/` (`get_0824_preprocessed_mix.sh`, `get_1124_preprocessed_mix.sh`, `preprocess_data.sh`, …) to your storage layout. The shell scripts assume `wget` (and, for the AWS variant, the `aws` CLI with credentials). `preprocess_data.sh` calls the `dolma` CLI.

---

## Running a single training job

```bash
torchrun --nproc-per-node=1 scripts/single_train_launch.py my_run_name \
    --model_name=olmo2_ml_80M \
    --moe_num_experts_list=32 \
    --moe_hidden_multipliers_list=0.25 \
    --moe_router_top_ks_list=4 \
    --moe_generalist_hidden_multiplier=0 \
    --moe_type=dropless \
    --train_datamix_name=OLMoE_mix_0824 \
    --train_module.optim.lr=4e-4
```

The positional `run_name` argument becomes the W&B run name and the subdirectory under `save_root` (i.e. `{save_root}/{run_name}/`).

---

## Running a single eval-only job

```bash
torchrun --nproc-per-node=1 scripts/single_eval_launch.py my_run_name \
    --model_name=olmo2_ml_80M \
    --moe_num_experts_list=32 \
    --moe_hidden_multipliers_list=0.25 \
    --moe_router_top_ks_list=4 \
    --moe_generalist_hidden_multiplier=0 \
    --moe_type=dropless \
    --train_datamix_name=OLMoE_mix_0824 \
    --train_module.optim.lr=4e-4
```

The `run_name` must match what was passed to `single_train_launch.py` — the script looks for the checkpoint under `{save_root}/{run_name}/`. `TrainerConfig.eval_only=True` is set, so the underlying `trainer.fit()` call runs the registered evaluators rather than training.

---

## MoE configuration

A variety of MoE architecture configuration choices may be specified through the command line. Length of the three `*_list` arguments must match — heterogeneous MoE just means each entry describes one expert group within a layer.

| Flag                                  | Type           | Default     | Meaning                                                                       |
|---------------------------------------|----------------|-------------|-------------------------------------------------------------------------------|
| `--moe_num_experts_list`              | csv ints       | `32,64`     | Experts per MoE layer. `1` means dense (no MoE). Multi-value = heterogeneous MoE (different expert counts per group within one layer). |
| `--moe_hidden_multipliers_list`       | csv floats     | `1024,2048` | Expert hidden-size multipliers (multiplied by `d_model`). Length must match `--moe_num_experts_list`. |
| `--moe_router_top_ks_list`            | csv ints       | `4,8`       | Top-k per router. Length must match `--moe_num_experts_list`.                 |
| `--moe_generalist_hidden_multiplier`  | float          | `1`         | Hybrid MoE: dense FFN width multiplier (0 = pure MoE, no dense path).         |
| `--moe_type`                          | `default` / `dropless` | `default` | `dropless` uses uncapped routing (no token drops); `default` uses a capacity factor. |
| `--moe_bias_gamma`                    | float or None  | None        | DeepSeek-v3-style router bias adjustment (γ).                                 |
| `--moe_z_loss_weight`                 | float          | `0.001`     | Router z-loss (entropy regularization) weight.                                |
| `--moe_lb_loss_weight`                | float          | `0.01`      | Load-balancing loss weight. Pass `0` to disable.                              |
| `--expert_assignment`                 | `learned` / `uniform` | `learned` | `uniform` bypasses the router and assigns tokens uniformly across experts (ablation only). |

Additional fine-grained knobs (learning rate, weight decay, …) are not exposed as top-level argparse flags — use the [override mechanism](#overriding-arbitrary-config-fields) below.

### Examples

**Dense baseline** (no MoE):

```bash
--moe_num_experts_list=1
```

**Homogeneous MoE** — 32 experts, top-4, dropless:

```bash
--moe_num_experts_list=32 --moe_hidden_multipliers_list=0.25 \
--moe_router_top_ks_list=4 --moe_generalist_hidden_multiplier=0 \
--moe_type=dropless
```

**MoE w/ Generalist** — 32 experts + a dense FFN with multiplier 0.5:

```bash
--moe_num_experts_list=32 --moe_hidden_multipliers_list=0.5 \
--moe_router_top_ks_list=2 --moe_generalist_hidden_multiplier=0.5
```

**Heterogeneous MoE** — two groups per layer, 8 wider experts + 32 narrower experts:

```bash
--moe_num_experts_list=8,32 --moe_hidden_multipliers_list=0.25,0.125 \
--moe_router_top_ks_list=2,4 --moe_generalist_hidden_multiplier=0
```

### Available model sizes

`olmo2_ml_10M`, `olmo2_ml_20M`, `olmo2_ml_50M`, `olmo2_ml_80M`, `olmo2_ml_100M`, `olmo2_ml_110M`, `olmo2_ml_200M`, `olmo2_ml_300M`, `olmo2_ml_500M` (see `MODEL_CONFIG_LOOKUP` in `single_train_launch.py`).

---

## Data, model, and runtime flags

| Flag                       | Type    | Default          | Meaning                                                                |
|----------------------------|---------|------------------|------------------------------------------------------------------------|
| `--tokenizer_name`         | str     | `dolma2`         | One of `dolma2`, `gpt_neox_olmo_dolma_v1_5` (see `TOKENIZER_LOOKUP`). |
| `--train_datamix_name`     | str     | `OLMoE_mix_0824` | Training mix (also supported: `OLMoE_mix_1124`).                       |
| `--valid_datamix_name`     | str     | `v3_small_ppl_validation` | Validation mix.                                              |
| `--data_root`              | str     | `USER_PROJECT_SPECS["DATAROOT"]` | Root URL/dir for training data.                       |
| `--valid_data_dir`         | str     | `USER_PROJECT_SPECS["VALID_DATA_DIR"]` | Root URL/dir for validation data.               |
| `--data_work_dir`          | str     | `USER_PROJECT_SPECS["DATA_WORK_DIR"]` | Local scratch for mix materialization.           |
| `--save_root`              | str     | `USER_PROJECT_SPECS["DEFAULT_SAVE_PATH"]` | Parent dir for checkpoints / configs.        |
| `--sequence_length`        | int     | `2048`           | Token sequence length.                                                 |
| `--global_batch_size`      | int     | `512`            | Sequences per global step.                                             |
| `--per_gpu_batch_size`     | int     | `16`             | Sequences per GPU. (Per-call programmatic default in `build_config` is `4` — the argparse default is what you get from the CLI.) |
| `--scheduler`              | `cosine` / `wsd` | `cosine` | LR schedule.                                                       |

---

## Overriding arbitrary config fields

Any field of the built `ExperimentConfig` can be overridden from the CLI via dotted-path flags. The launch script captures unknown args with `parse_known_args()` and applies them via `ExperimentConfig.merge(overrides)`:

```bash
--train_module.optim.lr=4e-4
--train_module.optim.weight_decay=0.1
--train_module.scheduler.warmup_steps=2000
--trainer.max_duration.value=1600000000   # tokens
--trainer.save_interval=200
```

Nested keys resolve against the `ExperimentConfig` dataclass tree (`model`, `dataset`, `data_loader`, `train_module`, `trainer`, `init_seed`).

---

## Citation

If you use this code, please cite our paper (BibTeX TODO) along with the upstream OLMo-core paper. This fork inherits the Apache 2.0 license from OLMo-core; see `LICENSE`.
