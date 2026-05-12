# `ml/` — MoE Training Pipeline

Tools for training OLMo-Core MoE (and dense) models on SLURM clusters. Wraps `olmo_core.train` with a parameter-sweep system, hyperparameter defaults, and CLI plumbing for the project-specific knobs we care about (MoE shape, data repetition, shared-block parameter sharing).

## Companion docs

- [`SHARED_MOE_TODOS.md`](./SHARED_MOE_TODOS.md) — backlog of open follow-ups for the `SharedBlockConfig` feature.
- [`SHARED_MOE_TOP3_PLANS.md`](./SHARED_MOE_TOP3_PLANS.md) — implementation plans for the highest-priority items.
- [`SHARED_MOE_FIX_NOTES.md`](./SHARED_MOE_FIX_NOTES.md) — walkthroughs of each correctness fix shipped on the shared-MoE feature.

---

## Setup

```bash
ENV_NAME=olmoe-core
OLMO_CORE_PATH=/path/to/OLMo-core

module load cuda/12.6.3
mamba create -n $ENV_NAME python=3.11
mamba activate $ENV_NAME

git clone https://github.com/sneha-rk/OLMo-core.git
cd OLMo-core
pip install -e .[all]

# Optional but recommended for fast MoE kernels.
git clone --recursive https://github.com/sneha-rk/grouped_gemm
cd grouped_gemm
GROUPED_GEMM_CUTLASS=1 pip install .
```

### Per-user config

Add an entry for `$USER` to `PROJECT_SPECS` in `ml/scripts/constants.py`:

```python
PROJECT_SPECS = {
    "yourname": {
        'DEFAULT_SAVE_PATH': '/path/to/save/models',
        'DATA_WORK_DIR':    '/path/to/data/work',
        'VALID_DATA_DIR':   '/path/to/data/preprocessed',
        "WANDB_PROJECT": "unimoe",
        "WANDB_ENTITY":  "ml-moe",
        "CONDA_ENV_NAME": "olmoe-core",
        "PROJECT_DIR":   DEFAULT_DIR_PATH,
        "COMMAND_PREFIX": f"{DEFAULT_DIR_PATH}/ml/scripts/single_train_launch.py",
        "NUM_GPUS": 4,
        "DATAROOT": "https://olmo-data.org/",
        "DATA_DIR": "/path/to/data",
        "NAME_KEYS": [],
    },
    ...
}
```

`ml/scripts/train_sweep.py` looks up the entry by `$USER` and uses these for save paths, W&B logging, and default SLURM resources.

---

## Pipeline overview

Three layered scripts under `ml/scripts/`:

| File                     | Role                                                                 |
|--------------------------|----------------------------------------------------------------------|
| `train_sweep.py`         | Top-level entry point. Defines a *grid* of hyperparameters and *subgrids* that bundle related settings. Submits one SLURM job per grid permutation. |
| `slurm_job.py`           | Internals of the sweep: expands the grid cartesian product, builds `--key=value` CLI strings, writes SLURM job scripts, and submits them. Generic — not project-specific. |
| `single_train_launch.py` | What each SLURM job runs. Parses CLI args (argparse + arbitrary `--nested.config.overrides`), builds `TransformerConfig` + `ExperimentConfig`, calls `trainer.fit()`. |

Argument flow:

```
train_sweep.py grid          (Python dict)
        │
        ▼  c_prod() + unroll_args()
slurm_job.py per-job command (string)
        │
        ▼  shell invocation
single_train_launch.py args  (argparse)
        │
        ▼  build_config()
TransformerConfig            (dataclass)
        │
        ▼  .build()
Transformer model            (nn.Module)
        │
        ▼  trainer.fit()
training run
```

`slurm_job.py` flattens any nested key (e.g. `train_module.optim.lr`) into `--train_module.optim.lr=value`. `single_train_launch.py` accepts known flags via argparse and captures the rest in `overrides` (via `parse_known_args()`), which get applied to the built config dict via `.merge(overrides)`. So you can override **any** field in the config tree from the sweep without adding a new argparse flag.

---

## Running a sweep

1. Edit `MODELS` (line ~13) and the `subgrids` dict (line ~158) in `ml/scripts/train_sweep.py`.
2. Run:

```bash
python ml/scripts/train_sweep.py \
    -n my_sweep_name \
    -a $SLURM_ACCOUNT \
    -p $SLURM_PARTITION \
    --gpus 4 \
    --cpus 5 \
    --mem 120G \
    -t 24:00:00
```

Useful flags:

| Flag                       | Default       | Description                                                    |
|----------------------------|---------------|----------------------------------------------------------------|
| `-n, --sweep-name`         | `data_rep_AC` | Prefix for the sweep directory.                                |
| `-a, --slurm-account`      | (required)    | SLURM account.                                                 |
| `-p, --slurm-partition`    | (required)    | SLURM partition.                                               |
| `-t, --job-time`           | `24:00:00`    | Wallclock per job.                                             |
| `--gpus / --cpus / --mem`  | from user spec| Per-job resources; overrides defaults in `HARDWARE_SPECS_DICT`.|
| `--debug`                  | off           | Caps job time at 1h; useful for quick sanity checks.           |
| `--dry-mode`               | off           | Generates job scripts but doesn't submit. Inspect before launch.|
| `-rn, --relaunch-name`     | None          | Resume an existing sweep by name. Re-uses its `grid.json`.     |

### Anatomy of a grid

Each model in `MODELS` gets its own sweep. The grid has two top-level keys:

```python
grid = {
    "main_grid": {
        # Applied to every job. Lists of values produce cartesian products.
        "model_name": [model],
        "save_root": [f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}"],
        "moe_type":  ["dropless"],
        "train_datamix_name": ["dclm_only"],
        "train_module": {"optim": {"lr": [4e-4]}},  # nested → --train_module.optim.lr
    },
    "subgrids": {
        # Each subgrid is a named bundle of settings for one experiment.
        "moe32_rep1x": {
            "moe_num_experts_list": ["32"],
            "moe_hidden_multipliers_list": ["0.25"],
            "moe_router_top_ks_list": ["4"],
            "moe_generalist_hidden_multiplier": ["0"],
            "unique_data_fraction": ["1.0"],
            "num_repetitions": ["1"],
        },
        "moe32_rep4x": {
            "moe_num_experts_list": ["32"], ...
            "unique_data_fraction": ["0.25"], "num_repetitions": ["4"],
        },
    },
}
```

Subgrids inherit `main_grid` values; settings in the subgrid override. The sweep launches `|main_grid permutations| × |subgrid permutations|` jobs per subgrid name.

Many existing subgrids in `train_sweep.py` are commented out — uncomment the ones you want or add new ones following the same shape.

---

## Running a single job (without SLURM)

For debugging, you can invoke `single_train_launch.py` directly:

```bash
torchrun --nproc-per-node=1 ml/scripts/single_train_launch.py my_run_name \
    --model_name=olmo2_ml_80M \
    --moe_num_experts_list=32 \
    --moe_hidden_multipliers_list=0.25 \
    --moe_router_top_ks_list=4 \
    --moe_generalist_hidden_multiplier=0 \
    --moe_type=dropless \
    --train_datamix_name=dclm_only \
    --train_module.optim.lr=4e-4
```

The positional `run_name` argument becomes the W&B run name and the subdirectory under `save_root`.

---

## MoE configuration

All MoE flags in `single_train_launch.py`:

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

### Examples

**Dense baseline** (no MoE):

```bash
--moe_num_experts_list=1
```

**Pure MoE** (no dense FFN), 32 experts, top-4, dropless:

```bash
--moe_num_experts_list=32 --moe_hidden_multipliers_list=0.25 \
--moe_router_top_ks_list=4 --moe_generalist_hidden_multiplier=0 \
--moe_type=dropless
```

**Hybrid MoE** (32 experts + a dense FFN with multiplier 0.5):

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

## Shared MoE blocks

A variant where routers, experts, and/or attention/norms are **physically shared** across a contiguous range of transformer layers — i.e. multiple `block.feed_forward_moe` attributes point to the same `nn.Module`, so the parameters live in memory exactly once and are *applied* N times during the forward pass.

Useful for parameter-efficient MoE experiments where you want the inner FLOPs of an N-layer network but the parameter footprint of a 1-layer (shared) middle stack.

### Configuration

Disabled by default. Enable by setting `--shared_blocks_start_layer` to an integer. The shared range is `[start_layer, start_layer + n_layers - 1]` (inclusive).

| Flag                                 | Type      | Default | Meaning                                                                 |
|--------------------------------------|-----------|---------|-------------------------------------------------------------------------|
| `--shared_blocks_start_layer`        | int / None| None    | First layer index in shared range. None = sharing disabled.             |
| `--shared_blocks_n_layers`           | int       | 2       | Number of contiguous layers in the shared range. Must be ≥ 2.           |
| `--shared_blocks_share_routers`      | bool      | False   | Share the MoE routers across the shared range.                          |
| `--shared_blocks_share_experts`      | bool      | False   | Share the MoE expert MLPs across the shared range.                      |
| `--shared_blocks_share_shared_mlp`   | bool      | False   | Share the dense `shared_mlp` (only present in hybrid MoE blocks).       |
| `--shared_blocks_share_attention`    | bool      | False   | Share attention/sequence-mixer across the shared range.                 |
| `--shared_blocks_share_norms`        | bool      | False   | Share `attention_norm` and `feed_forward_norm`.                         |

Boolean flags accept `True/False`, `1/0`, `yes/no`, `t/f` (case-insensitive). Anything else raises.

### Examples

Share routers and experts across the middle 4 layers of an 80M model (~12 layers deep):

```bash
--shared_blocks_start_layer=4 --shared_blocks_n_layers=4 \
--shared_blocks_share_routers=True --shared_blocks_share_experts=True
```

Share *everything* (routers, experts, shared MLP, attention, norms) across the middle 6 layers — extreme parameter sharing, only the outermost layers are unique:

```bash
--shared_blocks_start_layer=3 --shared_blocks_n_layers=6 \
--shared_blocks_share_routers=True --shared_blocks_share_experts=True \
--shared_blocks_share_shared_mlp=True --shared_blocks_share_attention=True \
--shared_blocks_share_norms=True
```

Share only routers (keep per-layer experts, study what routing-policy reuse looks like):

```bash
--shared_blocks_start_layer=4 --shared_blocks_n_layers=4 \
--shared_blocks_share_routers=True --shared_blocks_share_experts=False
```

### Pre-flight smoke test

Before submitting a real training run, you can validate the config-build path locally — no GPU, no data, no SLURM required:

```bash
python ml/scripts/smoke_test_shared_moe.py
```

This builds a small `olmo2_ml_10M` model with `shared_blocks` enabled (on `init_device="meta"`), then checks module identity, parameter counts (`num_params` and `num_param_uses`), `state_dict` dedup, and W&B tag generation. Exits 0 on success. Useful before sinking time into a SLURM submission.

### Example sweep subgrids

`train_sweep.py` includes commented-out example subgrids for shared-block runs:

```python
"moe32_shared_RE_mid4": {
    "moe_num_experts_list": ["32"],
    "moe_hidden_multipliers_list": ["0.25"],
    "moe_router_top_ks_list": ["4"],
    "moe_generalist_hidden_multiplier": ["0"],
    "shared_blocks_start_layer": ["4"],
    "shared_blocks_n_layers":    ["4"],
    "shared_blocks_share_routers": ["True"],
    "shared_blocks_share_experts": ["True"],
},
```

Uncomment, adjust `start_layer` / `n_layers` for your model depth, and run the sweep.

### What gets reported

- **W&B tags.** Runs with `shared_blocks` enabled are tagged `sharedMoE` and `share[RE]@4+4` (letters encode which flags are on: R=routers, E=experts, S=shared_mlp, A=attention, N=norms; `@start+n_layers` shows the range).
- **Parameter counts.** `model.num_params` reports the *unique* parameter count (shared params counted once). `model.num_param_uses` reports the *usage-weighted* count (shared params counted once per referencing block). Use the latter for FLOPs estimation: `flops ≈ 6 × num_active_param_uses × tokens`.

### What it interacts with

- **EP / FSDP / TP / CP.** All four parallelism strategies are dedup'd internally — each shared submodule is parallelized/sharded exactly once, not N times. See `SHARED_MOE_FIX_NOTES.md` for details.
- **Init schemes.** `Transformer.init_weights` dedups so depth-scaled init (`InitMethod.llama_depth`) lands the canonical (lowest) `block_idx`'s std on shared params, not the highest. See fix notes for the worked example.
- **Checkpoints.** `state_dict()` emits only canonical paths (smaller checkpoint files). `load_state_dict()` fans the canonical entry out to alias slots so `strict=True` works, and **raises** if the source has per-layer values that disagree for what's now a single shared Parameter (catches silent corruption when warm-starting from an unshared pretrained checkpoint).
- **FSDP fine-grained wrapping.** Not yet supported with `shared_blocks` — `apply_fsdp(wrapping_strategy="fine_grained")` raises a configuration error in that case. Use `"blocks"` or `"full"`.

---

## Data and repetition

| Flag                       | Type    | Default          | Meaning                                                                |
|----------------------------|---------|------------------|------------------------------------------------------------------------|
| `--train_datamix_name`     | str     | `OLMoE_mix_0824` | Training mix (`dclm_only`, `c4_only`, `OLMoE_mix_0824`, etc.).         |
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

---

## File layout

```
ml/
├── README.md                      # this file
├── SHARED_MOE_TODOS.md            # backlog for SharedBlockConfig follow-ups
├── SHARED_MOE_TOP3_PLANS.md       # implementation plans for top priorities
├── SHARED_MOE_FIX_NOTES.md        # per-fix walkthroughs
├── run_train.sh                   # one-shot launch helper
├── run_train_scale.sh             # multi-scale launch helper
└── scripts/
    ├── constants.py               # PROJECT_SPECS, HARDWARE_SPECS_DICT, MODEL_HP_DEFAULTS
    ├── train_sweep.py             # top-level sweep launcher
    ├── slurm_job.py               # grid expansion + SLURM submission
    ├── single_train_launch.py     # what each job runs
    ├── routing_analysis.py        # post-training routing inspection
    ├── utils.py                   # small helpers (dict_update, etc.)
    └── data/                      # data-mix definitions
```
