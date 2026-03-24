import argparse
import json
import os
from datetime import datetime
from copy import copy
from slurm_job import run_grid
from constants import PROJECT_SPECS, HARDWARE_SPECS_DICT, MODEL_HP_DEFAULTS, JOB_SPEC_KEYS
from utils import dict_update, get_specs_for_user_and_model


SWEEP_NAME_DEFAULT = 'lr_sweep_5XD'
project = 'moe'
MODELS = [
    # 'olmo2_1_0B',
    # 'olmo2_400M',
    # 'olmo2_100M',
    # 'olmo2_50M',
    # 'olmo2_20M',
    # 'olmo2_10M',
    # 'olmo2_5M',
    # 'olmo2_1M',
    # 'olmo2b_500M',
    'olmo2b_50M',
    'olmo2b_20M',
    'olmo2b_10M',
]
SPECS_KEYS_TO_IGNORE = ['per_gpu_batch_size', 'MEM_GB']

def main(
    sweep_name=SWEEP_NAME_DEFAULT,
    relaunch_path=None,
    relaunch_name=None,
    add_time_to_name='front',
    add_model_to_name='end',
    debug=False, 
    dry_mode=False,
    account=None, 
    partition=None,
    job_time='24:00:00',
    gpus=None,
    cpus=None,
    mem=None,
    include_jobs_indices=None,
    ignore_specs_check_keys=["NUM_CPUS", "MEM_GB", "JOBTIME", "CONDA_ENV_NAME"],
    filter_succeeded=True,
    filter_running=True,
    **kwargs,
):
    if account is None or partition is None:
        raise RuntimeError("Must specify account and partition")

    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    job_time = '00:30:00' if debug else job_time
    user = os.environ.get('USER')
    if user not in PROJECT_SPECS:
        raise ValueError(f"User {user} not found in PROJECT_SPECS. Please add your user to the PROJECT_SPECS dictionary.")
    USER_SPECS = PROJECT_SPECS[user]

    if relaunch_path or relaunch_name:
        if relaunch_name and relaunch_path:
            raise ValueError("Cannot specify both relaunch_name and relaunch_path")
        if relaunch_name:
            relaunch_path = os.path.join(PROJECT_SPECS[user]['DEFAULT_SAVE_PATH'], relaunch_name)
        relaunch_path = relaunch_path.rstrip('/')
        model_sweep_name = os.path.basename(relaunch_path)
        path_to_grid_file = os.path.join(relaunch_path, 'grid.json')
        path_to_specs = os.path.join(relaunch_path, 'specs.json')
        if not os.path.exists(path_to_grid_file):
            raise FileNotFoundError(f"Grid file {path_to_grid_file} does not exist.")
        grid = json.load(open(path_to_grid_file, 'r'))
        model = grid.get('main_grid', {}).get('model_name', [None])[0]

        SPECS = get_specs_for_user_and_model(USER_SPECS, HARDWARE_SPECS_DICT, model, partition, gpus, cpus, mem)

        if os.path.exists(path_to_specs):
            old_specs = json.load(open(path_to_specs, 'r'))
            for key in old_specs:
                if key not in ignore_specs_check_keys:
                    if key in SPECS_KEYS_TO_IGNORE:
                        continue
                    if key not in SPECS:
                        continue
                    assert SPECS.get(key) == old_specs[key], f"Specs mismatch for {key}: {SPECS.get(key)} != {old_specs[key]}"
        
        run_grid(
            grid,
            default_grid=dict_update(copy(MODEL_HP_DEFAULTS['all']), MODEL_HP_DEFAULTS.get(model, {})),
            sweep_name=model_sweep_name,
            specs=SPECS,
            name_keys=SPECS.get("NAME_KEYS", []),
            prefix=SPECS['TRAIN_COMMAND_PREFIX'],
            gpus=SPECS['NUM_GPUS'],
            cpus=SPECS["NUM_CPUS"],
            nodes=((SPECS['NUM_GPUS'] - 1) // 8 + 1),
            node_exclude=None,
            account=account,
            partition=partition,
            DIR_PATH=SPECS["PROJECT_DIR"],
            jobtime=(job_time if job_time else SPECS.get("JOBTIME", '24:00:00')),            
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
            bash_setup_file=SPECS.get("BASH_SETUP_FILE", None),
            include_jobs_indices=include_jobs_indices,
            filter_succeeded=filter_succeeded,
            filter_running=filter_running,
            # append_to_sbatch_str=None,
        )
        
    else:
        SWEEP_NAME = sweep_name
        if add_time_to_name == 'front':
            time_str = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
            SWEEP_NAME = f"{time_str}_{SWEEP_NAME}" if SWEEP_NAME else time_str
        for model in MODELS:
            model_sweep_name = f"{SWEEP_NAME}_{model}" if add_model_to_name == 'end' else SWEEP_NAME

            SPECS = get_specs_for_user_and_model(USER_SPECS, HARDWARE_SPECS_DICT, model, partition, gpus, cpus, mem)
            
            grid = {
                # main_grid is the top-level grid, the sweep will run over all combinations of these hyperparameters, 
                # combined with the subgrids
                "main_grid": { 
                    "model_name": [model],
                    "save_root": [f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}"],
                    # "scheduler": ["wsd"],
                    "moe_type": ["dropless"],
                    # "moe_type": ["default"],
                    "moe_bias_gamma": [0.001],  # None for default, or specify a float value
                    "moe_lb_loss_weight": [0.01], # Weight for the lb-loss in MoE
                    # "moe_lb_loss_weight": [0.0001],  # Weight for the lb-loss in MoE
                    "expert_assignment": ["learned"],
                    # "expert_assignment": ["uniform"],
                    'train_module': {
                        'optim': {
                            # 'lr': [4e-3, 1e-2],
                            'lr': [4e-4],
                            # 'lr': [1e-3],
                        },
                    },
                    "trainer": {
                        "max_duration": {
                            # "value": [2000000000],
                        },
                    },
                },
                # allows you to bundle multiple hyperparameters together
                "subgrids": {
                    ### no generalist models
                    # ###
                    # "e1x1c1": {"moe_num_experts_list": ["1"]},
                    # "e2x0.5c2_nogen": {"moe_num_experts_list": ["2"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e4x0.25c4_nogen": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e8x0.125c8_nogen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    
                    # "e2x1c1_nogen": {"moe_num_experts_list": ["2"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e4x0.5c2_nogen": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e8x0.25c4_nogen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e16x0.125c8_nogen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e32x0.0625c16_nogen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["16"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e64x0.03125c32_nogen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.03125"], "moe_router_top_ks_list": ["32"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e128x0.015625c64_nogen": {"moe_num_experts_list": ["128"], "moe_hidden_multipliers_list": ["0.015625"], "moe_router_top_ks_list": ["64"], "moe_generalist_hidden_multiplier": ["0"]},
                    
                    # "e4x1c1_nogen": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e8x0.5c2_nogen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e16x0.25c4_nogen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e32x0.125c8_nogen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e64x0.0625c16_nogen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["16"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e128x0.03125c32_nogen": {"moe_num_experts_list": ["128"], "moe_hidden_multipliers_list": ["0.03125"], "moe_router_top_ks_list": ["32"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e256x0.015625c64_nogen": {"moe_num_experts_list": ["256"], "moe_hidden_multipliers_list": ["0.015625"], "moe_router_top_ks_list": ["64"], "moe_generalist_hidden_multiplier": ["0"]},
                    
                    # "e8x1c1_nogen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e16x0.5c2_nogen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e32x0.25c4_nogen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e64x0.125c8_nogen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e128x0.0625c16_nogen": {"moe_num_experts_list": ["128"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["16"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e256x0.03125c32_nogen": {"moe_num_experts_list": ["256"], "moe_hidden_multipliers_list": ["0.03125"], "moe_router_top_ks_list": ["32"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e512x0.015625c64_nogen": {"moe_num_experts_list": ["512"], "moe_hidden_multipliers_list": ["0.015625"], "moe_router_top_ks_list": ["64"], "moe_generalist_hidden_multiplier": ["0"]},
                    
                    # "e16x1c1_nogen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e32x0.5c2_nogen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e64x0.25c4_nogen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    
                    # "e128x0.125c8_nogen": {"moe_num_experts_list": ["128"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e256x0.0625c16_nogen": {"moe_num_experts_list": ["256"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["16"], "moe_generalist_hidden_multiplier": ["0"]},
                    
                    # "e64x0.5c2_nogen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e128x0.25c4_nogen": {"moe_num_experts_list": ["128"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e256x0.125c8_nogen": {"moe_num_experts_list": ["256"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e512x0.0625c16_nogen": {"moe_num_experts_list": ["512"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["16"], "moe_generalist_hidden_multiplier": ["0"]},
                    
                    # "e128x0.5c2_nogen": {"moe_num_experts_list": ["128"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e256x0.25c4_nogen": {"moe_num_experts_list": ["256"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e512x0.125c8_nogen": {"moe_num_experts_list": ["512"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e1024x0.0625c16_nogen": {"moe_num_experts_list": ["1024"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["16"], "moe_generalist_hidden_multiplier": ["0"]},
                    # ###

                    # "e256x0.5c2_nogen": {"moe_num_experts_list": ["256"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e512x0.25c4_nogen": {"moe_num_experts_list": ["512"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    
                    # "e512x0.5c2_nogen": {"moe_num_experts_list": ["512"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e1024x0.25c4_nogen": {"moe_num_experts_list": ["1024"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e4096x0.0625c16_nogen": {"moe_num_experts_list": ["4096"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["16"], "moe_generalist_hidden_multiplier": ["0"]},
                    
                    # "e1024x0.5c2_nogen": {"moe_num_experts_list": ["1024"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e4096x0.125c8_nogen": {"moe_num_experts_list": ["4096"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    
                    # ###
                    "e4,8x0.5,0.25c1,2_nogen": {"moe_num_experts_list": ["4,8"], "moe_hidden_multipliers_list": ["0.5,0.25"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e8,16x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["8,16"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e16,32x0.125,0.0625c4,8_nogen": {"moe_num_experts_list": ["16,32"], "moe_hidden_multipliers_list": ["0.125,0.0625"], "moe_router_top_ks_list": ["4,8"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e32,64x0.0625,0.03125c8,16_nogen": {"moe_num_experts_list": ["32,64"], "moe_hidden_multipliers_list": ["0.0625,0.03125"], "moe_router_top_ks_list": ["8,16"], "moe_generalist_hidden_multiplier": ["0"]},

                    "e8,16x0.5,0.25c1,2_nogen": {"moe_num_experts_list": ["8,16"], "moe_hidden_multipliers_list": ["0.5,0.25"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e16,32x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["16,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e32,64x0.125,0.0625c4,8_nogen": {"moe_num_experts_list": ["32,64"], "moe_hidden_multipliers_list": ["0.125,0.0625"], "moe_router_top_ks_list": ["4,8"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e64,128x0.0625,0.03125c8,16_nogen": {"moe_num_experts_list": ["64,128"], "moe_hidden_multipliers_list": ["0.0625,0.03125"], "moe_router_top_ks_list": ["8,16"], "moe_generalist_hidden_multiplier": ["0"]},

                    "e16,32x0.5,0.25c1,2_nogen": {"moe_num_experts_list": ["16,32"], "moe_hidden_multipliers_list": ["0.5,0.25"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e32,64x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["32,64"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e64,128x0.125,0.0625c4,8_nogen": {"moe_num_experts_list": ["64,128"], "moe_hidden_multipliers_list": ["0.125,0.0625"], "moe_router_top_ks_list": ["4,8"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e128,256x0.0625,0.03125c8,16_nogen": {"moe_num_experts_list": ["128,256"], "moe_hidden_multipliers_list": ["0.0625,0.03125"], "moe_router_top_ks_list": ["8,16"], "moe_generalist_hidden_multiplier": ["0"]},

                    "e32,64x0.5,0.25c1,2_nogen": {"moe_num_experts_list": ["32,64"], "moe_hidden_multipliers_list": ["0.5,0.25"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e64,128x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["64,128"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e128,256x0.125,0.0625c4,8_nogen": {"moe_num_experts_list": ["128,256"], "moe_hidden_multipliers_list": ["0.125,0.0625"], "moe_router_top_ks_list": ["4,8"], "moe_generalist_hidden_multiplier": ["0"]},
                    
                    
                    # ###

                    # "e4,16x0.5,0.125c1,4_nogen": {"moe_num_experts_list": ["4,16"], "moe_hidden_multipliers_list": ["0.5,0.125"], "moe_router_top_ks_list": ["1,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e16,16,32x0.5,0.25,0.125c1,2,4_nogen": {"moe_num_experts_list": ["16,16,32"], "moe_hidden_multipliers_list": ["0.5,0.25,0.125"], "moe_router_top_ks_list": ["1,2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e32,32,64x0.5,0.25,0.125c1,2,4_nogen": {"moe_num_experts_list": ["32,32,64"], "moe_hidden_multipliers_list": ["0.5,0.25,0.125"], "moe_router_top_ks_list": ["1,2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e8,32x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["8,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32,32x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["32,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e48,32x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["48,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e96,64x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["96,64"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e16,96x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["16,96"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    
                    ### 0.5 generalist models
                    # ###
                    # "e3x0.5c1_0.5gen": {"moe_num_experts_list": ["3"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e6x0.25c2_0.5gen": {"moe_num_experts_list": ["6"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e12x0.125c4_0.5gen": {"moe_num_experts_list": ["12"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    
                    # "e7x0.5c1_0.5gen": {"moe_num_experts_list": ["7"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e14x0.25c2_0.5gen": {"moe_num_experts_list": ["14"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e28x0.125c4_0.5gen": {"moe_num_experts_list": ["28"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    "e7,14x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["7,14"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    ## "e15,30x0.125,0.0625c2,4_0.5gen": {"moe_num_experts_list": ["15,30"], "moe_hidden_multipliers_list": ["0.125,0.0625"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    
                    # "e15x0.5c1_0.5gen": {"moe_num_experts_list": ["15"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e30x0.25c2_0.5gen": {"moe_num_experts_list": ["30"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e60x0.125c4_0.5gen": {"moe_num_experts_list": ["60"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    "e15,30x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["15,30"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    ## "e31,62x0.125,0.0625c2,4_0.5gen": {"moe_num_experts_list": ["31,62"], "moe_hidden_multipliers_list": ["0.125,0.0625"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0.5"]}
                    
                    # "e62x0.25c2_0.5gen": {"moe_num_experts_list": ["62"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    "e124x0.125c4_0.5gen": {"moe_num_experts_list": ["124"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    "e31,62x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["31,62"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    ## "e63,126x0.125,0.0625c2,4_0.5gen": {"moe_num_experts_list": ["63,126"], "moe_hidden_multipliers_list": ["0.125,0.0625"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0.5"]}
                    
                    # "e124x0.125c4_0.5gen": {"moe_num_experts_list": ["124"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # ###
                    ### 0.25 generalist models
                    # ###
                    # "e7x0.25c3_0.25gen": {"moe_num_experts_list": ["7"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e14x0.125c6_0.25gen": {"moe_num_experts_list": ["14"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["6"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    "e28x0.0625c12_0.25gen": {"moe_num_experts_list": ["28"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["12"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    
                    # "e15x0.25c3_0.25gen": {"moe_num_experts_list": ["15"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e30x0.125c6_0.25gen": {"moe_num_experts_list": ["30"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["6"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    "e60x0.0625c12_0.25gen": {"moe_num_experts_list": ["60"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["12"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    "e7,16x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["7,16"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    
                    # "e31x0.25c3_0.25gen": {"moe_num_experts_list": ["31"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e62x0.125c6_0.25gen": {"moe_num_experts_list": ["62"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["6"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    "e124x0.0625c12_0.25gen": {"moe_num_experts_list": ["124"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["12"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    "e15,32x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["15,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},   
                    
                    # "e63x0.25c3_0.25gen": {"moe_num_experts_list": ["63"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e127x0.25c3_0.25gen": {"moe_num_experts_list": ["127"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # ###
                    # # "e31,64x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["31,64"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # ####  0.125 generalist model
                    # ###
                    # "e15x0.125c7_0.125gen": {"moe_num_experts_list": ["15"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["7"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    # "e30x0.0625c14_0.125gen": {"moe_num_experts_list": ["30"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["14"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    # "e60x0.03125c28_0.125gen": {"moe_num_experts_list": ["60"], "moe_hidden_multipliers_list": ["0.03125"], "moe_router_top_ks_list": ["28"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    # "e31x0.125c7_0.125gen": {"moe_num_experts_list": ["31"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["7"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    "e62x0.0625c14_0.125gen": {"moe_num_experts_list": ["62"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["14"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    # "e63x0.125c7_0.125gen": {"moe_num_experts_list": ["63"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["7"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    "e126x0.0625c14_0.125gen": {"moe_num_experts_list": ["126"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["14"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    # "e127x0.125c7_0.125gen": {"moe_num_experts_list": ["127"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["7"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    # ###
                    # "e256x0.125c7_0.125gen": {"moe_num_experts_list": ["256"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["7"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    # "e512x0.125c7_0.125gen": {"moe_num_experts_list": ["512"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["7"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    # "e1024x0.125c8_nogen": {"moe_num_experts_list": ["1024"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e4096x0.125c8_nogen": {"moe_num_experts_list": ["4096"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]}
                    # "e4096x0.0625c16_nogen": {"moe_num_experts_list": ["4096"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["16"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e128x0.015625c64_nogen": {"moe_num_experts_list": ["128"], "moe_hidden_multipliers_list": ["0.015625"], "moe_router_top_ks_list": ["64"], "moe_generalist_hidden_multiplier": ["0"]},
                    

                    # "e31x0.0625c15_0.0625gen": {"moe_num_experts_list": ["31"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["15"], "moe_generalist_hidden_multiplier": ["0.0625"]},


                    # ===================================================
                    ### 0.5 generalist models
                    # "e4x0.5c1_0.5gen": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e8x0.5c1_0.5gen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e16x0.5c1_0.5gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e8x0.25c2_0.5gen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e16x0.25c2_0.5gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e16x0.125c4_0.5gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32x0.25c2_0.5gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e64x0.25c2_0.5gen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e128x0.25c2_0.5gen": {"moe_num_experts_list": ["128"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32x0.125c4_0.5gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e64x0.125c4_0.5gen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e8,32x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["8,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e16,32x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["16,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32,32x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["32,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32,64x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["32,64"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e8,16x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["8,16"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    ### 0.25 generalist models
                    # "e8x0.25c3_0.25gen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e16x0.25c3_0.25gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e16x0.125c6_0.25gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["6"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e4,16x0.5,0.125c1,2_0.25gen": {"moe_num_experts_list": ["4,16"], "moe_hidden_multipliers_list": ["0.5,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e8,16x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["8,16"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e32x0.25c3_0.25gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    ## "e64x0.25c3_0.25gen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    ## "e128x0.25c3_0.25gen": {"moe_num_experts_list": ["128"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e32x0.125c6_0.25gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["6"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e64x0.125c6_0.25gen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["6"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e8,32x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["8,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e16,32x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["16,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e32,32x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["32,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e32,64x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["32,64"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    ####  0.125 generalist model
                    # "e16x0.125c7_0.125gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["7"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    # "e32x0.125c7_0.125gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["7"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    # "e64x0.125c7_0.125gen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["7"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    # "e128x0.125c7_0.125gen": {"moe_num_experts_list": ["128"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["7"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    # "e256x0.125c7_0.125gen": {"moe_num_experts_list": ["256"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["7"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    # "e512x0.125c7_0.125gen": {"moe_num_experts_list": ["512"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["7"], "moe_generalist_hidden_multiplier": ["0.125"]},
                    # "e1024x0.125c8_nogen": {"moe_num_experts_list": ["1024"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e4096x0.125c8_nogen": {"moe_num_experts_list": ["4096"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"], "per_gpu_batch_size": [2]},
                    # "e32x0.0625c16_nogen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["16"], "moe_generalist_hidden_multiplier": ["0"], "per_gpu_batch_size": [8]},
                    # "e4096x0.0625c16_nogen": {"moe_num_experts_list": ["4096"], "moe_hidden_multipliers_list": ["0.0625"], "moe_router_top_ks_list": ["16"], "moe_generalist_hidden_multiplier": ["0"], "per_gpu_batch_size": [8]},
                    # "e64x0.03125c32_nogen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.03125"], "moe_router_top_ks_list": ["16"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e128x0.015625c64_nogen": {"moe_num_experts_list": ["128"], "moe_hidden_multipliers_list": ["0.015625"], "moe_router_top_ks_list": ["64"], "moe_generalist_hidden_multiplier": ["0"], "per_gpu_batch_size": [8]},
                    
                },
            }

            run_grid(
                grid,
                default_grid=dict_update(copy(MODEL_HP_DEFAULTS['all']), MODEL_HP_DEFAULTS.get(model, {})),
                sweep_name=model_sweep_name,
                specs=SPECS,
                job_spec_keys=JOB_SPEC_KEYS,
                name_keys=SPECS.get("NAME_KEYS", []),
                prefix=SPECS['TRAIN_COMMAND_PREFIX'],
                gpus=SPECS['NUM_GPUS'],
                cpus=SPECS["NUM_CPUS"],
                nodes=((SPECS['NUM_GPUS'] - 1) // 8 + 1),
                node_exclude=None,
                account=account,
                partition=partition,
                DIR_PATH=SPECS["PROJECT_DIR"],
                jobtime=(job_time if job_time else SPECS.get("JOBTIME", '24:00:00')),      
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
                bash_setup_file=SPECS.get("BASH_SETUP_FILE", None),
                include_jobs_indices=include_jobs_indices,
                filter_succeeded=filter_succeeded,
                filter_running=filter_running,
                # append_to_sbatch_str=None,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sn', '--sweep-name', type=str, default=SWEEP_NAME_DEFAULT)
    parser.add_argument('-rp', '--relaunch-path', type=str, default=None, help="Path to the sweep directory containing grid.json and specs.json. Used to restart jobs from a previous sweep.")
    parser.add_argument('-rn', '--relaunch-name', type=str, default=None, help="Name of sweep, also base of sweep directory containing grid.json and specs.json. Used to restart jobs from a previous sweep.")
    parser.add_argument('--add-time-to-name', type=str, default='front', choices=['front', 'none'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-mode', action='store_true')
    parser.add_argument('-a', '--slurm-account', type=str)
    parser.add_argument('-p', '--slurm-partition', type=str)
    parser.add_argument('-t', '--job-time', type=str)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--cpus', type=int)
    parser.add_argument('--mem', type=str)
    parser.add_argument('-i', '--include-jobs-indices', type=str, default=None)
    parser.add_argument('-nf', '--no-filter', action='store_true', help="If set, will not filter out jobs that have already been run in the sweep. Useful for debugging.")
    parser.add_argument('-or', '--override', )

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
        include_jobs_indices=([int(i) for i in args.include_jobs_indices.split(",")] if args.include_jobs_indices else None),
        filter_running=not args.no_filter,
        filter_succeeded=not args.no_filter,
    )
