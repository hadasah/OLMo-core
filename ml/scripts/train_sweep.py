import argparse
import json
import os
from copy import copy
from datetime import datetime

from constants import HARDWARE_SPECS_DICT, MODEL_HP_DEFAULTS, PROJECT_SPECS
from slurm_job import run_grid
from utils import dict_update

SWEEP_NAME_DEFAULT = "data_rep_AC"
project = "moe"
MODELS = [
    # 'olmo2_ml_10M',
    "olmo2_ml_80M",
    # 'olmo2_ml_200M',
]

# === Family A / A+rep / B: two-domain data-mixing sweeps (additive) ===
# When one of the RUN_FAMILY_* toggles is True, the sweep's subgrids are replaced (per model) by
# the corresponding grid produced below; set all False to run the original single-source
# repetition subgrids defined inline in the grid literal. The families are mutually exclusive
# (each replaces the whole subgrid set). INCLUDE_GRANULARITY additionally sweeps 16- and
# 128-expert variants (fixed active params: top_k=4, hidden_mult=0.25).
#   Family A:     DCLM<->Dolma3 interpolation (composition ratios x architecture, rep=1).
#   Family A+rep: the same 5 interpolation ratios x repetition {1,2,4,8,16,32} x architecture,
#                 unique pool nested across rep (higher R => smaller head-prefix), ratio fixed.
#   Family B:     generalist primary + repeated specialist minority (rep ladder x architecture).
# All three are False so the inline single-source subgrids below are used. The
# sparsity/granularity sweep (moe64s8, moe64s32) lives there; leaving RUN_FAMILY_B
# True would REPLACE those subgrids and silently drop the whole experiment.
RUN_FAMILY_A = False
RUN_FAMILY_A_REP = False
RUN_FAMILY_B = False
# Not used for the granularity sweep: moe16/moe128 are defined explicitly in the
# inline subgrids instead, so they get the same flat naming as the other arms.
INCLUDE_GRANULARITY = False


def build_family_b_subgrids():
    """Generate the Family B subgrids: a generalist primary blended with a repeated
    specialist minority, swept over mixing cell x minority-repetition x architecture."""
    # (cell label, primary datamix, minority datamix, minority fraction)
    cells = [
        ("sc50", "dclm_only", "starcoder_only", "0.5"),
        ("sc90", "dclm_only", "starcoder_only", "0.1"),
        ("p2o50", "dclm_only", "pes2o_only", "0.5"),
        # p2o10 mirrors sc90 for pes2o, so both specialists are covered at a 50% and a
        # 10% minority share. Appended last, so its 18 subgrids are indices 54-71.
        ("p2o10", "dclm_only", "pes2o_only", "0.1"),
    ]
    reps = ["1", "2", "4", "8", "16", "32"]
    arches = [
        ("dense", {"moe_num_experts_list": ["1"]}),
        ("moe32", {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"],
                   "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]}),
        ("moe64", {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"],
                   "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]}),
    ]
    if INCLUDE_GRANULARITY:
        arches += [
            ("moe16", {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.25"],
                       "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]}),
            ("moe128", {"moe_num_experts_list": ["128"], "moe_hidden_multipliers_list": ["0.25"],
                        "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]}),
        ]
    subgrids = {}
    for cell, primary, minority, frac in cells:
        for rep in reps:
            for arch_name, arch in arches:
                sg = {
                    "mix_primary": [primary],
                    "mix_minority": [minority],
                    "minority_fraction": [frac],
                    "minority_repetition": [rep],
                    "primary_repetition": ["1"],
                }
                sg.update(arch)
                subgrids[f"famB_{cell}_rep{rep}x_{arch_name}"] = sg
    return subgrids


def build_family_a_subgrids():
    """Generate the Family A subgrids: DCLM<->Dolma3 interpolation (the DataDecide DCLM<->Dolma
    recipe with the architecture axis added), at rep=1 (pure composition, no repetition).

    Five DCLM:Dolma3 composition ratios {100/0, 75/25, 50/50, 25/75, 0/100} x architecture.
    (Dolma 1.7 is not hosted on olmo-data.org; Dolma3 = OLMo-mix-0925-official is the hosted,
    faithful substitute -- same DCLM-vs-curated-mix contrast; dolma3-tokenizer == dolma2 vocab.)
    The endpoints (100/0, 0/100) are single-source runs (just set train_datamix_name to
    dclm_only / dolma3, which overrides the main_grid value). The three interior blends reuse
    the Family B two-source mixture build path with minority=dolma3 and minority_repetition=1,
    so minority_fraction = the dolma3 share of the fixed token budget.
    """
    arches = [
        ("dense", {"moe_num_experts_list": ["1"]}),
        ("moe32", {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"],
                   "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]}),
        ("moe64", {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"],
                   "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]}),
    ]
    if INCLUDE_GRANULARITY:
        arches += [
            ("moe16", {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.25"],
                       "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]}),
            ("moe128", {"moe_num_experts_list": ["128"], "moe_hidden_multipliers_list": ["0.25"],
                        "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]}),
        ]
    # DCLM percentage of the blend (Dolma3 is the remaining 1 - pct/100).
    dclm_pcts = [100, 75, 50, 25, 0]
    subgrids = {}
    for pct in dclm_pcts:
        for arch_name, arch in arches:
            if pct == 100:
                sg = {"train_datamix_name": ["dclm_only"]}        # pure DCLM endpoint
            elif pct == 0:
                sg = {"train_datamix_name": ["dolma3"]}           # pure Dolma3 endpoint
            else:
                sg = {
                    "mix_primary": ["dclm_only"],
                    "mix_minority": ["dolma3"],
                    "minority_fraction": [f"{(100 - pct) / 100:.2f}"],
                    "minority_repetition": ["1"],
                    "primary_repetition": ["1"],
                }
            sg.update(arch)
            subgrids[f"famA_dclm{pct}_{arch_name}"] = sg
    return subgrids


def build_family_a_rep_subgrids():
    """Family A + repetition: the DCLM<->Dolma3 interpolation crossed with the data-repetition
    axis. Each of the 5 interpolation ratios {100/0, 75/25, 50/50, 25/75, 0/100} is run at
    repetition {1,2,4,8,16,32}x, with the *unique* pool strictly nested across rep levels (higher
    R => smaller file-order head-prefix) and the DCLM:Dolma3 ratio held constant at every R.

    Interiors (75/25, 50/50, 25/75) use the two-source re-epoch path (mix_unique_fraction=1/R): the
    mixture is built at the unique budget T/R with take_ratio<1 (no in-mixture tiling) and the data
    loader re-epochs R times to fill the budget T. Endpoints (100/0 = dclm_only, 0/100 = dolma3) use
    the single-source repetition path (unique_data_fraction=1/R), matching the original single-
    dataset repetition experiment (and, at rep=1, the plain single-source full run).
    """
    arches = [
        ("dense", {"moe_num_experts_list": ["1"]}),
        ("moe32", {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"],
                   "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]}),
        ("moe64", {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"],
                   "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]}),
    ]
    if INCLUDE_GRANULARITY:
        arches += [
            ("moe16", {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.25"],
                       "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]}),
            ("moe128", {"moe_num_experts_list": ["128"], "moe_hidden_multipliers_list": ["0.25"],
                        "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]}),
        ]
    # repetition R -> unique fraction 1/R of the token budget (both as strings for the CLI).
    reps = [("1", "1.0"), ("2", "0.5"), ("4", "0.25"), ("8", "0.125"), ("16", "0.0625"), ("32", "0.03125")]
    # interior blends: (cell label, dolma3 minority fraction)
    interiors = [("dclm75", "0.25"), ("dclm50", "0.50"), ("dclm25", "0.75")]
    # endpoints: (cell label, single-source datamix name)
    endpoints = [("dclm100", "dclm_only"), ("dclm0", "dolma3")]
    subgrids = {}
    for rep, frac in reps:
        for cell, minority_frac in interiors:
            for arch_name, arch in arches:
                sg = {
                    "mix_primary": ["dclm_only"],
                    "mix_minority": ["dolma3"],
                    "minority_fraction": [minority_frac],
                    "mix_unique_fraction": [frac],
                    "num_repetitions": [rep],
                }
                sg.update(arch)
                subgrids[f"famAr_{cell}_rep{rep}x_{arch_name}"] = sg
        for cell, datamix in endpoints:
            for arch_name, arch in arches:
                sg = {
                    "train_datamix_name": [datamix],
                    "unique_data_fraction": [frac],
                    "num_repetitions": [rep],
                }
                sg.update(arch)
                subgrids[f"famAr_{cell}_rep{rep}x_{arch_name}"] = sg
    return subgrids


def main(
    sweep_name=SWEEP_NAME_DEFAULT,
    relaunch_path=None,
    relaunch_name=None,
    add_time_to_name="front",
    add_model_to_name="end",
    debug=False,
    dry_mode=False,
    account=None,
    partition=None,
    job_time="24:00:00",
    gpus=None,
    cpus=None,
    mem=None,
    include_jobs_indices=None,
    ignore_specs_check_keys=["NUM_CPUS", "MEM_GB", "JOBTIME"],
    filter_succeeded=True,
    filter_running=True,
    **kwargs,
):
    if account is None or partition is None:
        raise RuntimeError("Must specify account and partition")

    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    job_time = "1:00:00" if debug else job_time
    user = os.environ.get("USER")
    if user not in PROJECT_SPECS:
        raise ValueError(
            f"User {user} not found in PROJECT_SPECS. Please add your user to the PROJECT_SPECS dictionary."
        )
    USER_SPECS = PROJECT_SPECS[user]

    if relaunch_path or relaunch_name:
        if relaunch_name and relaunch_path:
            raise ValueError("Cannot specify both relaunch_name and relaunch_path")
        if relaunch_name:
            relaunch_path = os.path.join(PROJECT_SPECS[user]["DEFAULT_SAVE_PATH"], relaunch_name)
        relaunch_path = relaunch_path.rstrip("/")
        model_sweep_name = os.path.basename(relaunch_path)
        path_to_grid_file = os.path.join(relaunch_path, "grid.json")
        path_to_specs = os.path.join(relaunch_path, "specs.json")
        if not os.path.exists(path_to_grid_file):
            raise FileNotFoundError(f"Grid file {path_to_grid_file} does not exist.")
        grid = json.load(open(path_to_grid_file, "r"))
        model = grid.get("main_grid", {}).get("model_name", [None])[0]

        SPECS = copy(USER_SPECS)
        SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT.get("all", {}))
        SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT.get(partition, {}))
        SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT[model].get(partition, {}))
        SPECS["NUM_GPUS"] = gpus or SPECS["NUM_GPUS"]
        SPECS["NUM_CPUS"] = cpus or SPECS["NUM_CPUS"]
        SPECS["MEM_GB"] = mem or SPECS["MEM_GB"]

        if os.path.exists(path_to_specs):
            old_specs = json.load(open(path_to_specs, "r"))
            for key in old_specs:
                if key not in ignore_specs_check_keys:
                    assert (
                        SPECS.get(key) == old_specs[key]
                    ), f"Specs mismatch for {key}: {SPECS.get(key)} != {old_specs[key]}"

        run_grid(
            grid,
            default_grid=dict_update(
                copy(MODEL_HP_DEFAULTS["all"]), MODEL_HP_DEFAULTS.get(model, {})
            ),
            sweep_name=model_sweep_name,
            specs=SPECS,
            name_keys=SPECS.get("NAME_KEYS", []),
            prefix=SPECS["COMMAND_PREFIX"],
            gpus=SPECS["NUM_GPUS"],
            cpus=SPECS["NUM_CPUS"],
            nodes=((SPECS["NUM_GPUS"] - 1) // 8 + 1),
            node_exclude=None,
            account=account,
            partition=partition,
            DIR_PATH=SPECS["PROJECT_DIR"],
            jobtime=(job_time if job_time else SPECS.get("JOBTIME", "24:00:00")),
            include_job_id=False,
            hashname=False,
            saveroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
            logroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
            mem_gb=SPECS["MEM_GB"],
            requeue=True,
            data_parallel=False,
            comment=None,
            copy_env=True,
            copy_dirs=[],
            max_num_jobs=None,
            num_copies=1,
            job_id_start=1,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            dependencies=[],
            repo_name="olmoe-core",
            conda_env_name=SPECS.get("CONDA_ENV_NAME"),
            include_jobs_indices=include_jobs_indices,
            filter_succeeded=filter_succeeded,
            filter_running=filter_running,
            # append_to_sbatch_str=None,
        )

    else:
        SWEEP_NAME = sweep_name
        if add_time_to_name == "front":
            time_str = str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
            SWEEP_NAME = f"{time_str}_{SWEEP_NAME}" if SWEEP_NAME else time_str
        for model in MODELS:
            model_sweep_name = f"{SWEEP_NAME}_{model}" if add_model_to_name == "end" else SWEEP_NAME

            SPECS = copy(USER_SPECS)
            SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT.get("all", {}))
            SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT.get(partition, {}))
            SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT[model].get(partition, {}))
            SPECS["NUM_GPUS"] = gpus or SPECS["NUM_GPUS"]
            SPECS["NUM_CPUS"] = cpus or SPECS["NUM_CPUS"]
            SPECS["MEM_GB"] = mem or SPECS["MEM_GB"]
            grid = {
                # main_grid is the top-level grid, the sweep will run over all combinations of these hyperparameters,
                # combined with the subgrids
                "main_grid": {
                    "model_name": [model],
                    "save_root": [f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}"],
                    # "scheduler": ["wsd"],
                    "moe_type": ["dropless"],
                    # Single-source mixes — pick ONE per sweep. Available options:
                    #   "starcoder_only"  (100 files, code)
                    #   "wikipedia_only"  (63 files, encyclopedic prose)
                    #   "pes2o_only"      (26 files, academic papers)
                    #   "dclm_only"       (945 files, filtered web)
                    # Use "OLMoE_mix_0824" for the default full OLMo mix.
                    # ("c4_only" is registered but C4 is not on olmo-data.org — won't work.)
                    # The granularity sweep runs on the OLMoE mix, which is where the
                    # existing dense/moe32/moe64 ladder lives, so the new arms are
                    # directly comparable to it. Leaving this set to a single domain
                    # would have trained every arm on that domain instead.
                    "train_datamix_name": ["OLMoE_mix_0824"],
                    # "train_datamix_name": ["starcoder_only"],
                    # "train_datamix_name": ["dclm_only"],
                    # "train_datamix_name": ["wikipedia_only"],
                    # "train_datamix_name": ["pes2o_only"],
                    # "moe_bias_gamma": [0.001],  # None for default, or specify a float value
                    # "moe_lb_loss_weight": [0.0001],  # Weight for the lb-loss in MoE
                    "train_module": {
                        "optim": {
                            # 'lr': [4e-3, 1e-2],
                            "lr": [4e-4],
                        },
                    },
                    "trainer": {
                        "max_duration": {
                            # "value": [2000000000], #just testing
                        },
                    },
                },
                # allows you to bundle multiple hyperparameters together
                "subgrids": {
                    # "e1x1c1": {"moe_num_experts_list": ["1"]},
                    ### no generalist models
                    # "e2x1c1_nogen": {"moe_num_experts_list": ["2"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e4x1c1_nogen": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e8x1c1_nogen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e16x1c1_nogen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e4x0.5c2_nogen": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e8x0.5c2_nogen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e16x0.5c2_nogen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e8x0.25c4_nogen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e16x0.25c4_nogen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e16x0.125c8_nogen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e4,8x0.5,0.25c1,2_nogen": {"moe_num_experts_list": ["4,8"], "moe_hidden_multipliers_list": ["0.5,0.25"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e8,16x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["8,16"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e4,16x0.5,0.125c1,4_nogen": {"moe_num_experts_list": ["4,16"], "moe_hidden_multipliers_list": ["0.5,0.125"], "moe_router_top_ks_list": ["1,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e32x0.25c4_nogen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e64x0.25c4_nogen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e32x0.125c8_nogen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e64x0.125c8_nogen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e8,32x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["8,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e16,32x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["16,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e32,32x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["32,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e32,64x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["32,64"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    ## 0.5 generalist models
                    # "e4x0.5c1_0.5gen": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e8x0.5c1_0.5gen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e16x0.5c1_0.5gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e8x0.25c2_0.5gen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e16x0.25c2_0.5gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e16x0.125c4_0.5gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32x0.25c2_0.5gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e64x0.25c2_0.5gen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32x0.125c4_0.5gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e64x0.125c4_0.5gen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e8,32x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["8,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e16,32x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["16,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32,32x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["32,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32,64x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["32,64"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e8,16x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["8,16"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e8x0.25c3_0.25gen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e16x0.25c3_0.25gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e16x0.125c6_0.25gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["6"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e4,16x0.5,0.125c1,2_0.25gen": {"moe_num_experts_list": ["4,16"], "moe_hidden_multipliers_list": ["0.5,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e8,16x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["8,16"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},"e32x0.125c4_0.5gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32x0.25c3_0.25gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e64x0.25c3_0.25gen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e32x0.125c6_0.25gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["6"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e64x0.125c6_0.25gen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["6"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e8,32x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["8,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e16,32x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["16,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e32,32x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["32,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e32,64x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["32,64"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    ## === Data repetition experiments (A+C) ===
                    ## Dense baselines at different repetition levels
#                     "dense_rep1x": {
#                         "moe_num_experts_list": ["1"],
#                         "unique_data_fraction": ["1.0"],
#                         "num_repetitions": ["1"],
#                     },
#                     "dense_rep2x": {
#                         "moe_num_experts_list": ["1"],
#                         "unique_data_fraction": ["0.5"],
#                         "num_repetitions": ["2"],
#                     },
#                     "dense_rep4x": {
#                         "moe_num_experts_list": ["1"],
#                         "unique_data_fraction": ["0.25"],
#                         "num_repetitions": ["4"],
#                     },
#                     "dense_rep8x": {
#                         "moe_num_experts_list": ["1"],
#                         "unique_data_fraction": ["0.125"],
#                         "num_repetitions": ["8"],
#                     },
#                     "dense_rep16x": {
#                         "moe_num_experts_list": ["1"],
#                         "unique_data_fraction": ["0.0625"],
#                         "num_repetitions": ["16"],
#                     },
#                     "dense_rep32x": {
#                         "moe_num_experts_list": ["1"],
#                         "unique_data_fraction": ["0.03125"],
#                         "num_repetitions": ["32"],
#                     },
                    # "dense_rep64x": {"moe_num_experts_list": ["1"], "unique_data_fraction": ["0.015625"], "num_repetitions": ["64"]},
                    # "dense_rep128x": {"moe_num_experts_list": ["1"], "unique_data_fraction": ["0.0078125"], "num_repetitions": ["128"]},
                    # "dense_rep256x": {"moe_num_experts_list": ["1"], "unique_data_fraction": ["0.00390625"], "num_repetitions": ["256"], "global_batch_size": ["64"], "per_gpu_batch_size": ["8"]},
                    # "dense_rep512x": {"moe_num_experts_list": ["1"], "unique_data_fraction": ["0.001953125"], "num_repetitions": ["512"], "global_batch_size": ["64"], "per_gpu_batch_size": ["8"]},
                    # "dense_rep1024x": {"moe_num_experts_list": ["1"], "unique_data_fraction": ["0.0009765625"], "num_repetitions": ["1024"], "global_batch_size": ["64"], "per_gpu_batch_size": ["8"]},
                    ## MoE 32 experts at different repetition levels
#                     "moe32_rep1x": {
#                         "moe_num_experts_list": ["32"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["1.0"],
#                         "num_repetitions": ["1"],
#                     },
#                     "moe32_rep2x": {
#                         "moe_num_experts_list": ["32"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.5"],
#                         "num_repetitions": ["2"],
#                     },
#                     "moe32_rep4x": {
#                         "moe_num_experts_list": ["32"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.25"],
#                         "num_repetitions": ["4"],
#                     },
#                     "moe32_rep8x": {
#                         "moe_num_experts_list": ["32"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.125"],
#                         "num_repetitions": ["8"],
#                     },
#                     "moe32_rep16x": {
#                         "moe_num_experts_list": ["32"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.0625"],
#                         "num_repetitions": ["16"],
#                     },
#                     "moe32_rep32x": {
#                         "moe_num_experts_list": ["32"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.03125"],
#                         "num_repetitions": ["32"],
#                     },
                    # "moe32_rep64x": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.015625"], "num_repetitions": ["64"]},
                    # "moe32_rep128x": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.0078125"], "num_repetitions": ["128"]},
                    # "moe32_rep256x": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.00390625"], "num_repetitions": ["256"], "global_batch_size": ["64"], "per_gpu_batch_size": ["8"]},
                    # "moe32_rep512x": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.001953125"], "num_repetitions": ["512"], "global_batch_size": ["64"], "per_gpu_batch_size": ["8"]},
                    # "moe32_rep1024x": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.0009765625"], "num_repetitions": ["1024"], "global_batch_size": ["64"], "per_gpu_batch_size": ["8"]},
                    ## MoE 64 experts at different repetition levels
#                     "moe64_rep1x": {
#                         "moe_num_experts_list": ["64"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["1.0"],
#                         "num_repetitions": ["1"],
#                     },
#                     "moe64_rep2x": {
#                         "moe_num_experts_list": ["64"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.5"],
#                         "num_repetitions": ["2"],
#                     },
#                     "moe64_rep4x": {
#                         "moe_num_experts_list": ["64"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.25"],
#                         "num_repetitions": ["4"],
#                     },
#                     "moe64_rep8x": {
#                         "moe_num_experts_list": ["64"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.125"],
#                         "num_repetitions": ["8"],
#                     },
#                     "moe64_rep16x": {
#                         "moe_num_experts_list": ["64"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.0625"],
#                         "num_repetitions": ["16"],
#                     },
#                     "moe64_rep32x": {
#                         "moe_num_experts_list": ["64"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.03125"],
#                         "num_repetitions": ["32"],
#                     },
                    # "moe64_rep64x": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.015625"], "num_repetitions": ["64"]},
                    # "moe64_rep128x": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.0078125"], "num_repetitions": ["128"]},
                    # "moe64_rep256x": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.00390625"], "num_repetitions": ["256"], "global_batch_size": ["64"], "per_gpu_batch_size": ["8"]},
                    # "moe64_rep512x": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.001953125"], "num_repetitions": ["512"], "global_batch_size": ["64"], "per_gpu_batch_size": ["8"]},
                    # "moe64_rep1024x": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.0009765625"], "num_repetitions": ["1024"], "global_batch_size": ["64"], "per_gpu_batch_size": ["8"]},
                    ## MoE 64 experts of g=1/8 at different repetition levels
                    "moe64s8_rep1x": {
                        "moe_num_experts_list": ["64"],
                        "moe_hidden_multipliers_list": ["0.125"],
                        "moe_router_top_ks_list": ["8"],
                        "moe_generalist_hidden_multiplier": ["0"],
                        "unique_data_fraction": ["1.0"],
                        "num_repetitions": ["1"],
                    },
                    "moe64s8_rep2x": {
                        "moe_num_experts_list": ["64"],
                        "moe_hidden_multipliers_list": ["0.125"],
                        "moe_router_top_ks_list": ["8"],
                        "moe_generalist_hidden_multiplier": ["0"],
                        "unique_data_fraction": ["0.5"],
                        "num_repetitions": ["2"],
                    },
                    "moe64s8_rep4x": {
                        "moe_num_experts_list": ["64"],
                        "moe_hidden_multipliers_list": ["0.125"],
                        "moe_router_top_ks_list": ["8"],
                        "moe_generalist_hidden_multiplier": ["0"],
                        "unique_data_fraction": ["0.25"],
                        "num_repetitions": ["4"],
                    },
                    "moe64s8_rep8x": {
                        "moe_num_experts_list": ["64"],
                        "moe_hidden_multipliers_list": ["0.125"],
                        "moe_router_top_ks_list": ["8"],
                        "moe_generalist_hidden_multiplier": ["0"],
                        "unique_data_fraction": ["0.125"],
                        "num_repetitions": ["8"],
                    },
                    "moe64s8_rep16x": {
                        "moe_num_experts_list": ["64"],
                        "moe_hidden_multipliers_list": ["0.125"],
                        "moe_router_top_ks_list": ["8"],
                        "moe_generalist_hidden_multiplier": ["0"],
                        "unique_data_fraction": ["0.0625"],
                        "num_repetitions": ["16"],
                    },
                    "moe64s8_rep32x": {
                        "moe_num_experts_list": ["64"],
                        "moe_hidden_multipliers_list": ["0.125"],
                        "moe_router_top_ks_list": ["8"],
                        "moe_generalist_hidden_multiplier": ["0"],
                        "unique_data_fraction": ["0.03125"],
                        "num_repetitions": ["32"],
                    },
                    ## MoE 64 experts of g = 1/2 at different repetition levels
                    "moe64s32_rep1x": {
                        "moe_num_experts_list": ["64"],
                        "moe_hidden_multipliers_list": ["0.5"],
                        "moe_router_top_ks_list": ["2"],
                        "moe_generalist_hidden_multiplier": ["0"],
                        "unique_data_fraction": ["1.0"],
                        "num_repetitions": ["1"],
                    },
                    "moe64s32_rep2x": {
                        "moe_num_experts_list": ["64"],
                        "moe_hidden_multipliers_list": ["0.5"],
                        "moe_router_top_ks_list": ["2"],
                        "moe_generalist_hidden_multiplier": ["0"],
                        "unique_data_fraction": ["0.5"],
                        "num_repetitions": ["2"],
                    },
                    "moe64s32_rep4x": {
                        "moe_num_experts_list": ["64"],
                        "moe_hidden_multipliers_list": ["0.5"],
                        "moe_router_top_ks_list": ["2"],
                        "moe_generalist_hidden_multiplier": ["0"],
                        "unique_data_fraction": ["0.25"],
                        "num_repetitions": ["4"],
                    },
                    "moe64s32_rep8x": {
                        "moe_num_experts_list": ["64"],
                        "moe_hidden_multipliers_list": ["0.5"],
                        "moe_router_top_ks_list": ["2"],
                        "moe_generalist_hidden_multiplier": ["0"],
                        "unique_data_fraction": ["0.125"],
                        "num_repetitions": ["8"],
                    },
                    "moe64s32_rep16x": {
                        "moe_num_experts_list": ["64"],
                        "moe_hidden_multipliers_list": ["0.5"],
                        "moe_router_top_ks_list": ["2"],
                        "moe_generalist_hidden_multiplier": ["0"],
                        "unique_data_fraction": ["0.0625"],
                        "num_repetitions": ["16"],
                    },
                    "moe64s32_rep32x": {
                        "moe_num_experts_list": ["64"],
                        "moe_hidden_multipliers_list": ["0.5"],
                        "moe_router_top_ks_list": ["2"],
                        "moe_generalist_hidden_multiplier": ["0"],
                        "unique_data_fraction": ["0.03125"],
                        "num_repetitions": ["32"],
                    },
                    ## MoE 128 experts at different repetition levels
#                     "moe128_rep1x": {
#                         "moe_num_experts_list": ["128"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["1.0"],
#                         "num_repetitions": ["1"],
#                     },
#                     "moe128_rep2x": {
#                         "moe_num_experts_list": ["128"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.5"],
#                         "num_repetitions": ["2"],
#                     },
#                     "moe128_rep4x": {
#                         "moe_num_experts_list": ["128"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.25"],
#                         "num_repetitions": ["4"],
#                     },
#                     "moe128_rep8x": {
#                         "moe_num_experts_list": ["128"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.125"],
#                         "num_repetitions": ["8"],
#                     },
#                     "moe128_rep16x": {
#                         "moe_num_experts_list": ["128"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.0625"],
#                         "num_repetitions": ["16"],
#                     },
#                     "moe128_rep32x": {
#                         "moe_num_experts_list": ["128"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.03125"],
#                         "num_repetitions": ["32"],
#                     },
                    ## MoE 64 experts at different repetition levels
#                     "moe16_rep1x": {
#                         "moe_num_experts_list": ["16"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["1.0"],
#                         "num_repetitions": ["1"],
#                     },
#                     "moe16_rep2x": {
#                         "moe_num_experts_list": ["16"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.5"],
#                         "num_repetitions": ["2"],
#                     },
#                     "moe16_rep4x": {
#                         "moe_num_experts_list": ["16"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.25"],
#                         "num_repetitions": ["4"],
#                     },
#                     "moe16_rep8x": {
#                         "moe_num_experts_list": ["16"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.125"],
#                         "num_repetitions": ["8"],
#                     },
#                     "moe16_rep16x": {
#                         "moe_num_experts_list": ["16"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.0625"],
#                         "num_repetitions": ["16"],
#                     },
#                     "moe16_rep32x": {
#                         "moe_num_experts_list": ["16"],
#                         "moe_hidden_multipliers_list": ["0.25"],
#                         "moe_router_top_ks_list": ["4"],
#                         "moe_generalist_hidden_multiplier": ["0"],
#                         "unique_data_fraction": ["0.03125"],
#                         "num_repetitions": ["32"],
#                     },
                },
            }

            if sum([RUN_FAMILY_A, RUN_FAMILY_A_REP, RUN_FAMILY_B]) > 1:
                raise ValueError(
                    "RUN_FAMILY_A / RUN_FAMILY_A_REP / RUN_FAMILY_B are mutually exclusive; enable only one."
                )
            if RUN_FAMILY_A:
                # DCLM<->Dolma3 interpolation (composition, rep=1).
                grid["subgrids"] = build_family_a_subgrids()
            elif RUN_FAMILY_A_REP:
                # DCLM<->Dolma3 interpolation crossed with the repetition axis {1,2,4,8,16,32}x.
                grid["subgrids"] = build_family_a_rep_subgrids()
            elif RUN_FAMILY_B:
                # Generalist primary + repeated specialist minority. Set all toggles False to
                # restore the original single-source repetition sweep defined in the grid literal.
                grid["subgrids"] = build_family_b_subgrids()

            run_grid(
                grid,
                default_grid=dict_update(
                    copy(MODEL_HP_DEFAULTS["all"]), MODEL_HP_DEFAULTS.get(model, {})
                ),
                sweep_name=model_sweep_name,
                specs=SPECS,
                name_keys=SPECS.get("NAME_KEYS", []),
                prefix=SPECS["COMMAND_PREFIX"],
                gpus=SPECS["NUM_GPUS"],
                cpus=SPECS["NUM_CPUS"],
                nodes=((SPECS["NUM_GPUS"] - 1) // 8 + 1),
                node_exclude=None,
                account=account,
                partition=partition,
                DIR_PATH=SPECS["PROJECT_DIR"],
                jobtime=(job_time if job_time else SPECS.get("JOBTIME", "24:00:00")),
                include_job_id=False,
                hashname=False,
                saveroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
                logroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
                mem_gb=SPECS["MEM_GB"],
                requeue=True,
                data_parallel=False,
                comment=None,
                copy_env=True,
                copy_dirs=[],
                max_num_jobs=None,
                num_copies=1,
                job_id_start=1,
                debug_mode=DEBUG_MODE,
                dry_mode=DRY_MODE,
                dependencies=[],
                repo_name="olmoe-core",
                conda_env_name=SPECS.get("CONDA_ENV_NAME"),
                include_jobs_indices=include_jobs_indices,
                filter_succeeded=filter_succeeded,
                filter_running=filter_running,
                # append_to_sbatch_str=None,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--sweep-name", type=str, default=SWEEP_NAME_DEFAULT)
    parser.add_argument(
        "-rp",
        "--relaunch-path",
        type=str,
        default=None,
        help="Path to the sweep directory containing grid.json and specs.json. Used to restart jobs from a previous sweep.",
    )
    parser.add_argument(
        "-rn",
        "--relaunch-name",
        type=str,
        default=None,
        help="Name of sweep, also base of sweep directory containing grid.json and specs.json. Used to restart jobs from a previous sweep.",
    )
    parser.add_argument("--add-time-to-name", type=str, default="front", choices=["front", "none"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-mode", action="store_true")
    parser.add_argument("-a", "--slurm-account", type=str)
    parser.add_argument("-p", "--slurm-partition", type=str)
    parser.add_argument("-t", "--job-time", type=str)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--mem", type=str)
    parser.add_argument("-i", "--include-jobs-indices", type=str, default=None)
    parser.add_argument(
        "-nf",
        "--no-filter",
        action="store_true",
        help="If set, will not filter out jobs that have already been run in the sweep. Useful for debugging.",
    )

    args = parser.parse_args()

    main(
        sweep_name=args.sweep_name,
        relaunch_path=args.relaunch_path,
        relaunch_name=args.relaunch_name,
        add_time_to_name=args.add_time_to_name,
        debug=args.debug,
        dry_mode=args.dry_mode,
        account=args.slurm_account,
        partition=args.slurm_partition,
        job_time=args.job_time,
        gpus=args.gpus,
        cpus=args.cpus,
        mem=args.mem,
        include_jobs_indices=(
            [int(i) for i in args.include_jobs_indices.split(",")]
            if args.include_jobs_indices
            else None
        ),
        filter_running=not args.no_filter,
        filter_succeeded=not args.no_filter,
    )
