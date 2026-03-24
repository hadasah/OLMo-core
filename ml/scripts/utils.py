import collections.abc
import os
import time
from copy import copy

# def recursive_dict_update(original_dict, update_dict):
#     """
#     Recursively updates a dictionary with values from another dictionary.
#     Handles nested dictionaries by merging them.
#     """
#     for key, value in update_dict.items():
#         if key in original_dict and isinstance(original_dict[key], dict) and isinstance(value, dict):
#             # If both values are dictionaries, recurse
#             original_dict[key] = recursive_dict_update(original_dict[key], value)
#         else:
#             # Otherwise, update or add the key-value pair
#             original_dict[key] = value
#     return original_dict


def dict_update(d, u):
    """
    Recursively update a dict with another dict.
    This is a deep update, meaning that if a key in the first dict
    has a dict as its value, and the second dict has a key with
    the same name, the value in the first dict will be updated
    with the value from the second dict.
    Keys in the second dict are the ones iterated over. 
    If the value in the second dict is not a dict, it will
    overwrite the value in the first dict.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_specs_for_user_and_model(USER_SPECS, HARDWARE_SPECS_DICT, model, partition, gpus, cpus, mem):
    SPECS = copy(USER_SPECS)
    SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT.get('all', {}))
    SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT.get(partition, {}))
    SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT[model].get("all", {}))
    SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT[model].get(partition, {}))
    SPECS['NUM_GPUS'] = gpus or SPECS['NUM_GPUS']
    SPECS["NUM_CPUS"] = cpus or SPECS["NUM_CPUS"]
    SPECS["MEM_GB"] = mem or SPECS["MEM_GB"]
    return SPECS

def has_file_been_modified_recently(filepath, recent_threshold_seconds=3600):
    """
    Checks if a file has been modified within a specified recent time threshold.

    Args:
        filepath (str): The path to the file.
        recent_threshold_seconds (int): The number of seconds defining "recently".
                                       Defaults to 3600 seconds (1 hour).

    Returns:
        bool: True if the file was modified within the threshold, False otherwise.
    """
    if not os.path.exists(filepath):
        return False  # File does not exist

    last_modified_time = os.path.getmtime(filepath)
    current_time = time.time()

    return (current_time - last_modified_time) < recent_threshold_seconds


def get_all_sweeps(run_top_dir, run_dir_substring):
    import glob
    run_top_dir = "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/models"
    pattern = os.path.join(run_top_dir, f"**{run_dir_substring}**", '')
    # Get the list of all subfolders recursively
    subfolders = glob.glob(pattern, recursive=True)
    # subfolders = glob.glob("/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/models/**{run_dir_substring}**/", recursive=True)
    return subfolders


def filter_training_incomplete(train_sweep_dir, final_jobs_dict, final_jobs_names):
    import re
    import glob

    filtered_job_names = []
    filtered_job_dict = {}

    run_dirs = glob.glob(os.path.join(train_sweep_dir, '*{sweep_name}*', ''), recursive=True)

    step_count_lookup = {
        '1CD':{
            'olmo2_10M': 954,
            'olmo2_20M': 1908,
            'olmo2_50M': 4769,
            'olmo2_100M': 9537,
            'olmo2b_10M': 191,
            'olmo2b_20M': 382,
            'olmo2b_50M': 954,
        },
        '5CD': {
            'olmo2b_20M': 1908,
        }
    }
    pattern = "(202\d_\d\d_\d\d-\d\d_\d\d_\d\d_(\w+)_(olmo2[b]?_\d+M|1_0B))_e(.+)x(.+)[ec](\d+,\d+|\d+)"

    for job_name in final_jobs_names:
    
        match = re.match(pattern, job_name)
        if match is None:
            continue
        sweep_name = match.group(1)
        run_dir = os.path.join(train_sweep_dir, job_name)
        # if '2026_02_26-10_58_23_default_lf_olmo2b_10M_e62x0.25c2_0.5gen' in job_name:
        #     continue
        if '5CD' not in job_name:
            if (
                os.path.isdir(
                    f"{run_dir}/step{step_count_lookup['1CD'].get(match.group(3), 0)}"
            )):
                filtered_job_names.append(job_name)
                filtered_job_dict[job_name] = final_jobs_dict[job_name]
        else:
            if (
                os.path.isdir(
                    f"{run_dir}/step{step_count_lookup['5CD'].get(match.group(3), 0)}"
            )):
                filtered_job_names.append(job_name)
                filtered_job_dict[job_name] = final_jobs_dict[job_name]
    return filtered_job_names, filtered_job_dict

def filter_eval_done(train_sweep_dir, final_jobs_dict, final_jobs_names):
    """
    Checks if the final evaluation for a run has been completed by looking for a specific file.

    Args:
        run_dir (str): The directory of the run.
        step_count (int): The step count to check for.
        final_eval_filename (str): The name of the file that indicates final evaluation is done.
        recent_threshold_seconds (int): The number of seconds defining "recently" for file modification.

    Returns:
        bool: True if the final evaluation is done, False otherwise.
    """
    import glob
    import re


    filtered_job_names = []
    filtered_job_dict = {}

    run_dirs = glob.glob(os.path.join(train_sweep_dir, '*{sweep_name}*', ''), recursive=True)
    pattern = "(202\d_\d\d_\d\d-\d\d_\d\d_\d\d_(\w+)_olmo2[b]?_(\d+M|1_0B))_e(.+)x(.+)[ec](\d+,\d+|\d+)"

    for job_name in final_jobs_names:
    
        match = re.match(pattern, job_name)
        if match is None:
            continue
        sweep_name = match.group(1)
        run_dir = os.path.join(train_sweep_dir, job_name)
        if not (
            os.path.exists(
                f"{run_dir}/final_eval_done.txt"
            )
        ):
            filtered_job_names.append(job_name)
            filtered_job_dict[job_name] = final_jobs_dict[job_name]
    return filtered_job_names, filtered_job_dict

    # import re
    # import wandb
    # import glob
    # import shutil
    
    # filtered_job_names = []
    # filtered_job_dict = {}

    # # final_eval_file_path = os.path.join(run_dir, f"step{step_count}", final_eval_filename)
    # #     return has_file_been_modified_recently(final_eval_file_path, recent_threshold_seconds)

    # api = wandb.Api(timeout=60)

    # # # Define the filter
    # # filters = {"display_name": run_name_to_find}

    # # # Get runs matching the filter
    # # runs = api.runs(path=f"{entity_name}/{project_name}", filters=filters)

    # # Project is specified by <entity/project-name>
    # runs = api.runs("ml-moe/moe")
    # pattern = "(202\d_\d\d_\d\d-\d\d_\d\d_\d\d_(\w+)_olmo2[b]?_(\d+M|1_0B))_e(.+)x(.+)[ec](\d+,\d+|\d+)"
    # # pattern = "(2026_\d\d_\d\d-\d\d_\d\d_\d\d_(\w+)_olmo2[b]?_(\d+M|1_0B))_e(.+)x(.+)[ec](\d+,\d+|\d+)"
    # for run in runs:
    #     if run.name not in final_jobs_names:
    #         continue
    #     # if sweep_name not in run.name:
    #     #     continue
    #     # if run.state != "finished": 
    #     #     continue

    #     tags_to_ignore = ["hidden", "_no_show", "notCM", "notEmatched", "notContMoE"]
    #     for tag in tags_to_ignore:
    #         if tag in run.tags:
    #             continue
        
    #     if sweep_name not in run.name:
    #         continue
        
    #     match = re.match(pattern, run.name)
    #     if match is None:
    #         continue

    #     if "final_eval/lm/c4_en-validation/CE loss" in run.summary:
    #         continue

    #     filtered_job_names.append(run.name)
    #     filtered_job_dict[run.name] = final_jobs_dict[run.name]

    # return filtered_job_names, filtered_job_dict
