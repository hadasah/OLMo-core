# Slicing and Dicing — Training & Eval Scripts

This directory contains the code for the *Slicing and Dicing: Configuring Optimal Mixture-of-Experts Models* paper. The training/eval stack is a hard fork of [OLMo-core](https://github.com/allenai/OLMo-core). See `README_olmo_core.md` for those docs, which we have not edited.

Script used to launch our trianing and eval runs live in the `scripts` folder:
- `single_train_launch.py` — pretraining entry point
- `single_eval_launch.py` — eval-only entry point that loads a checkpoint and runs the same LM + downstream evaluators that fire during training.
- `data/` — shell helpers for downloading and preprocessing the OLMoE data mixes. **Provided as-is**: they contain cluster-specific paths and will need to be edited to your environment before they run.

---

## Setup

```bash
ENV_NAME=sandd_olmo_core
SANDD_PATH=$YOUR_DIR/slicing_and_dicing

mamba create -n $ENV_NAME python=3.11
mamba activate $ENV_NAME

git clone https://github.com/hadasah/slicing_and_dicing.git $SANDD_PATH
cd $SANDD_PATH
pip install -e .[all]

# Optional but recommended for fast MoE kernels.
git clone --recursive https://github.com/sneha-rk/grouped_gemm
cd grouped_gemm
GROUPED_GEMM_CUTLASS=1 pip install .
```
We use cuda 12.6.3.

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
| `PROJECT_DIR` | Repo root, auto-derived from the script's path. |

CLI flags (e.g. `--data_root`) override the matching entry at run time.

---

## Data

We use the pretraining data mix from OLMoE. Follow the [instructions](https://github.com/allenai/OLMoE/blob/main/README.md#pretraining) in their repo to download and preprocess the data, or adapt the shell scripts in `scripts/data/` (`get_0824_preprocessed_mix.sh`, `get_1124_preprocessed_mix.sh`, `preprocess_data.sh`, …) to your storage layout. The shell scripts assume `wget` (and, for the AWS variant, the `aws` CLI with credentials). `preprocess_data.sh` calls the `dolma` CLI.

---

## Running a single training job

```bash
torchrun --nproc-per-node=1 scripts/single_train_launch.py run_name \
    --model_name=olmo2_ml_80M \
    --moe_num_experts_list=32 \
    --moe_hidden_multipliers_list=0.25 \
    --moe_router_top_ks_list=4 \
    --moe_generalist_hidden_multiplier=0 \
    --moe_type=dropless \
    --train_datamix_name=OLMoE_mix_0824 \
    --train_module.optim.lr=4e-4
```

The positional `run_name` argument becomes the WandB run name. Checkpoints and logs are saved in a subdirectory under `save_root` (i.e. at `{save_root}/{run_name}`).

---

## Running a single eval-only job

```bash
torchrun --nproc-per-node=1 scripts/single_eval_launch.py run_name \
    --model_name=olmo2_ml_80M \
    --moe_num_experts_list=32 \
    --moe_hidden_multipliers_list=0.25 \
    --moe_router_top_ks_list=4 \
    --moe_generalist_hidden_multiplier=0 \
    --moe_type=dropless \
    --train_datamix_name=OLMoE_mix_0824 \
    --train_module.optim.lr=4e-4
```

This script looks for the checkpoint under `{save_root}/{run_name}`, so make sure the `run_name` and `save_root` match what was passed to `single_train_launch.py`. 

---

## MoE configuration

A variety of our MoE architecture configuration choices may be specified through the command line. Length of the three `*_list` arguments must match — heterogeneous MoE just means each entry describes one expert group within a layer.

| Flag                                  | Type           | Default     | Meaning                                                                       |
|---------------------------------------|----------------|-------------|-------------------------------------------------------------------------------|
| `--moe_num_experts_list`              | str            | `32,64`     | Experts per MoE layer. `1` means dense. Multiple values, comma-separated = heterogeneous MoE (different expert counts per group within one layer). |
| `--moe_hidden_multipliers_list`       | str            | `0.25,0.125` | Expert hidden-size multiplier(s), multiplied by `d_model` to get expert FFN intermediate dimension. Length must match `--moe_num_experts_list`.  |
| `--moe_router_top_ks_list`            | str            | `2,4`       | Top-k value(s) per router. Length must match `--moe_num_experts_list`.                 |
| `--moe_generalist_hidden_multiplier`  | float          | `1`         | Generalist / shared expert hidden-size multipliers. A `0` value indicates no generalist.         |
| `--moe_type`                          | `default` / `dropless` | `default` | `dropless` uses uncapped routing (no token drops); `default` uses a capacity factor. |
| `--moe_bias_gamma`                    | float or None  | None        | DeepSeek-v3-style loss-free load balancing bias γ                                 |
| `--moe_z_loss_weight`                 | float          | `0.001`     | Router z-loss weight                                |
| `--moe_lb_loss_weight`                | float          | `0.01`      | Load-balancing loss weight. A `0` value indicates no load balancing loss                              |
| `--expert_assignment`                 | `learned` / `uniform` | `learned` | `uniform` bypasses the router and weights all expert outputs equally. _Only_ used for ablations. |

Other hyperparameters and settings (e.g., learning rate, weight decay, etc.) can also be set with the [override mechanism](#overriding-arbitrary-config-fields) below.

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

`olmo2_ml_10M`, `olmo2_ml_20M`, `olmo2_ml_50M`, `olmo2_ml_80M`, `olmo2_ml_100M`, `olmo2_ml_110M`, `olmo2_ml_200M`, `olmo2_ml_300M`, `olmo2_ml_500M` (see `MODEL_CONFIG_LOOKUP` in `single_train_launch.py`). Paper does not show results for `olmo2_ml_500M`, since our resource constraints resulted in a limited number of completed runs at that model parameter scale.

---

## Data, model, and runtime flags

| Flag                       | Type    | Default          | Meaning                                                                |
|----------------------------|---------|------------------|------------------------------------------------------------------------|
| `--tokenizer_name`         | str     | `dolma2`         | Tokenizer to use. We only use `dolma2`, but additional tokenizers may be added to `TOKENIZER_LOOKUP` and used |
| `--train_datamix_name`     | str     | `OLMoE_mix_0824` | Training data mix                       |
| `--valid_datamix_name`     | str     | `v3_small_ppl_validation` | Validation data                                              |
| `--data_root`              | str     | `USER_PROJECT_SPECS["DATAROOT"]` | Root URL/dir for training data                       |
| `--valid_data_dir`         | str     | `USER_PROJECT_SPECS["VALID_DATA_DIR"]` | Root URL/dir for validation data               |
| `--data_work_dir`          | str     | `USER_PROJECT_SPECS["DATA_WORK_DIR"]` | Local scratch directory for mix materialization           |
| `--save_root`              | str     | `USER_PROJECT_SPECS["DEFAULT_SAVE_PATH"]` | Parent directory for saving checkpoints, configs, and logs        |
| `--sequence_length`        | int     | `2048`           | Sequence length                                                 |
| `--global_batch_size`      | int     | `512`            | Number of sequences per batch                                             |
| `--per_gpu_batch_size`     | int     | `16`             | Number of sequences per GPU. (Per-call programmatic default in `build_config` is `4` — the argparse default is what you get from the CLI.) |
| `--scheduler`              | `cosine` / `wsd` | `cosine` | LR schedule. We use `cosine` in all main experiments.                                                      |

---

## Overriding other config fields

Any field of `ExperimentConfig` can be overridden from the CLI with dotted-path flags. The launch script captures unknown args with `parse_known_args()` and applies them via `ExperimentConfig.merge(overrides)`:

```bash
--train_module.optim.lr=4e-4
--train_module.optim.weight_decay=0.1
--train_module.scheduler.warmup_steps=2000
--trainer.max_duration.value=1600000000   # tokens
--trainer.save_interval=200
```

---

## Citation
```bibtex
@misc{slicinganddicing,
      title={Slicing and Dicing: Configuring Optimal Mixtures of Experts}, 
      author={Margaret Li and Sneha Kudugunta and Danielle Rothermel and Luke Zettlemoyer},
      year={2026},
      eprint={2605.11689},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.11689}, 
}
```

Citation for OLMo-core can be found in `README_olmo_core.md`.
The original OLMo-core repo is licensed with Apache 2.0. See `LICENSE`.
