"""
Standalone single-job smoke sweep that exercises the ``SharedBlockConfig`` feature
end-to-end on real hardware. Separate from ``train_sweep.py`` so it doesn't
disrupt the data-repetition sweep config in use there.

What this submits:

- One SLURM job
- Model: ``olmo2_ml_20M`` (4 layers, ~20M params)
- Sharing: ``share_routers=True, share_experts=True`` over layers 1-2 (canonical=1).
  Layers 0 and 3 are unshared. Validates the full sharing surface on a tiny model.
- MoE: 32 experts, top-4, dropless, no generalist FFN (pure MoE)
- Short token budget (~50M tokens) — meant to validate the pipeline, not converge

What this smoke test confirms when it succeeds:

- The shared-MoE build path works on CUDA (not just meta-device unit tests)
- Apply_ep / apply_fsdp dedup hold under real distributed setup
- The DCP checkpoint round-trip works with the state_dict dedup override
- W&B logging picks up the ``sharedMoE`` + ``share[RE]@1+2`` tags

Usage::

    python ml/scripts/smoke_sweep.py \\
        -a $SLURM_ACCOUNT \\
        -p $SLURM_PARTITION \\
        --gpus 2 \\
        -t 1:00:00

Add ``--dry-mode`` to generate the SLURM scripts without submitting them
(useful for inspecting the launch command before committing GPU time).
"""

import argparse
import os
from copy import copy
from datetime import datetime

from constants import HARDWARE_SPECS_DICT, MODEL_HP_DEFAULTS, PROJECT_SPECS
from slurm_job import run_grid
from utils import dict_update

MODEL = "olmo2_ml_20M"  # 4-layer MoE; shared range [1, 2] leaves 0 and 3 unshared
SWEEP_NAME_DEFAULT = "smoke_shared_moe"
# ~50M tokens. Plenty to surface kernel/parallelism/checkpoint issues without
# wasting GPU time. Adjust upward if you want to see a clear loss curve.
SMOKE_MAX_TOKENS = 50_000_000


def main(
    account,
    partition,
    gpus=None,
    cpus=None,
    mem=None,
    job_time="1:00:00",
    dry_mode=False,
    debug=False,
):
    user = os.environ.get("USER")
    if user not in PROJECT_SPECS:
        raise ValueError(f"User {user!r} not found in PROJECT_SPECS in ml/scripts/constants.py.")
    user_specs = PROJECT_SPECS[user]

    time_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    sweep_name = f"{time_str}_{SWEEP_NAME_DEFAULT}_{MODEL}"

    specs = copy(user_specs)
    specs = dict_update(specs, HARDWARE_SPECS_DICT.get("all", {}))
    specs = dict_update(specs, HARDWARE_SPECS_DICT.get(partition, {}))
    specs = dict_update(specs, HARDWARE_SPECS_DICT.get(MODEL, {}).get(partition, {}))
    specs["NUM_GPUS"] = gpus or specs["NUM_GPUS"]
    specs["NUM_CPUS"] = cpus or specs.get("NUM_CPUS", 4)
    specs["MEM_GB"] = mem or specs.get("MEM_GB", 64)

    grid = {
        "main_grid": {
            "model_name": [MODEL],
            "save_root": [f"{specs['DEFAULT_SAVE_PATH']}/{sweep_name}"],
            "moe_type": ["dropless"],
            "train_datamix_name": ["dclm_only"],
            # Override the model's default token budget for a quick smoke run.
            "trainer": {
                "max_duration": {
                    "value": [SMOKE_MAX_TOKENS],
                },
            },
        },
        "subgrids": {
            # Single subgrid → single job. Shares routers + experts across the
            # middle 2 layers of the 4-layer model.
            "smoke_RE": {
                "moe_num_experts_list": ["32"],
                "moe_hidden_multipliers_list": ["0.25"],
                "moe_router_top_ks_list": ["4"],
                "moe_generalist_hidden_multiplier": ["0"],
                "shared_blocks_start_layer": ["1"],
                "shared_blocks_n_layers": ["2"],
                "shared_blocks_share_routers": ["True"],
                "shared_blocks_share_experts": ["True"],
            },
        },
    }

    run_grid(
        grid,
        default_grid=dict_update(copy(MODEL_HP_DEFAULTS["all"]), MODEL_HP_DEFAULTS.get(MODEL, {})),
        sweep_name=sweep_name,
        specs=specs,
        name_keys=specs.get("NAME_KEYS", []),
        prefix=specs["COMMAND_PREFIX"],
        gpus=specs["NUM_GPUS"],
        cpus=specs["NUM_CPUS"],
        nodes=((specs["NUM_GPUS"] - 1) // 8 + 1),
        node_exclude=None,
        account=account,
        partition=partition,
        DIR_PATH=specs["PROJECT_DIR"],
        jobtime=job_time,
        include_job_id=False,
        hashname=False,
        saveroot=f"{specs['DEFAULT_SAVE_PATH']}/{sweep_name}",
        logroot=f"{specs['DEFAULT_SAVE_PATH']}/{sweep_name}",
        mem_gb=specs["MEM_GB"],
        # Don't auto-requeue — if the smoke test fails, surface the failure.
        requeue=False,
        data_parallel=False,
        comment=None,
        copy_env=True,
        copy_dirs=[],
        max_num_jobs=None,
        num_copies=1,
        job_id_start=1,
        debug_mode=debug,
        dry_mode=dry_mode,
        dependencies=[],
        repo_name="olmoe-core",
        conda_env_name=specs.get("CONDA_ENV_NAME"),
        include_jobs_indices=None,
        # Don't filter — we want to run even if a same-named job exists.
        filter_succeeded=False,
        filter_running=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-a", "--slurm-account", type=str, required=True)
    parser.add_argument("-p", "--slurm-partition", type=str, required=True)
    parser.add_argument("-t", "--job-time", type=str, default="1:00:00")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--mem", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Caps job time at 1h.")
    parser.add_argument(
        "--dry-mode",
        action="store_true",
        help="Generate SLURM scripts but don't submit. Inspect ${save_root} before launching.",
    )
    args = parser.parse_args()

    main(
        account=args.slurm_account,
        partition=args.slurm_partition,
        gpus=args.gpus,
        cpus=args.cpus,
        mem=args.mem,
        job_time=args.job_time,
        dry_mode=args.dry_mode,
        debug=args.debug,
    )
