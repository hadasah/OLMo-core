import argparse
import json
import os
from datetime import datetime
from copy import copy
from eval_slurm_job import run_grid
from constants import PROJECT_SPECS, HARDWARE_SPECS_DICT, MODEL_HP_DEFAULTS, JOB_SPEC_KEYS
from utils import dict_update, get_specs_for_user_and_model, get_all_sweeps

SWEEP_NAME_DEFAULT = "evals"
SPECS_KEYS_TO_IGNORE = ['per_gpu_batch_size', 'MEM_GB']

def main(
    SWEEP_NAME=SWEEP_NAME_DEFAULT,
    sweep_name_substring="",
    add_time_to_name='front',
    add_model_to_name='end',
    debug=False, 
    dry_mode=False,
    account=None, 
    partition=None,
    job_time='1:00:00',
    gpus=None,
    cpus=None,
    mem=None,
    include_jobs_indices=None,
    ignore_specs_check_keys=["NUM_CPUS", "MEM_GB", "JOBTIME", "CONDA_ENV_NAME"],
    **kwargs,
):
    if account is None or partition is None:
        raise RuntimeError("Must specify account and partition")

    time_str = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))

    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    job_time = '00:30:00' if debug else job_time
    user = os.environ.get('USER')
    if user not in PROJECT_SPECS:
        raise ValueError(f"User {user} not found in PROJECT_SPECS. Please add your user to the PROJECT_SPECS dictionary.")
    USER_SPECS = PROJECT_SPECS[user]

    eval_path = get_all_sweeps(USER_SPECS['DEFAULT_SAVE_PATH'], sweep_name_substring)
    # os.path.join(PROJECT_SPECS[user]['DEFAULT_SAVE_PATH'], relaunch_name)
    print(f"Found the following sweeps to eval: {eval_path}")
    for eval_path in eval_path:
        eval_path = eval_path.rstrip('/')
        eval_path_name = os.path.basename(eval_path)
        sweep_name = f"eval/{eval_path_name}/{time_str}"
        path_to_grid_file = os.path.join(eval_path, 'grid.json')
        path_to_specs = os.path.join(eval_path, 'specs.json')
        if not os.path.exists(path_to_grid_file):
            raise FileNotFoundError(f"Grid file {path_to_grid_file} does not exist.")
        grid = json.load(open(path_to_grid_file, 'r'))
        model = grid.get('main_grid', {}).get('model_name', [None])[0]

        SPECS = get_specs_for_user_and_model(USER_SPECS, HARDWARE_SPECS_DICT, model, partition, gpus, cpus, mem)
        run_grid(
            grid,
            default_grid=dict_update(copy(MODEL_HP_DEFAULTS['all']), MODEL_HP_DEFAULTS.get(model, {})),
            sweep_name=sweep_name,
            train_sweep_path=eval_path,
            specs=SPECS,
            name_keys=SPECS.get("NAME_KEYS", []),
            prefix=SPECS['EVAL_COMMAND_PREFIX'],
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
            saveroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{sweep_name}",
            logroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{sweep_name}",
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
            # append_to_sbatch_str=None,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sns', '--sweep-name-substring', type=str, default="")
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
        SWEEP_NAME=SWEEP_NAME_DEFAULT,
        sweep_name_substring=args.sweep_name_substring,
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
    )
