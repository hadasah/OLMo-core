import re
import wandb
import glob
import os
import shutil

run_top_dir = "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/models"
api = wandb.Api(timeout=60)

# Project is specified by <entity/project-name>
runs = api.runs("ml-moe/moe")
# pattern = "(202\d_\d\d_\d\d-\d\d_\d\d_\d\d_(\w+)_olmo2[b]?_(\d+M|1_0B))_e(.+)x(.+)[ec](\d+,\d+|\d+)"
pattern = "(2026_\d\d_\d\d-\d\d_\d\d_\d\d_(\w+)_(olmo2[b]?_\d+M|1_0B))_e(.+)x(.+)[ec](\d+,\d+|\d+)"
for run in runs:
    if run.state != "finished": 
        continue
    
    match = re.match(pattern, run.name)
    if match is None:
        continue
    run_dir = f"{run_top_dir}/{match.group(1)}/{run.name}"
    print(run_dir)


    if "hidden" in run.tags or "_no_show" in run.tags or "notCM" in run.tags or not "notEmatched" in run.tags:
        folder_pattern = f'{run_dir}/step*0'
        folders_to_delete = glob.glob(folder_pattern)

        # Iterate over the list of files and delete each one
        for folder_path in folders_to_delete:
            try:
                shutil.rmtree(folder_path)
                print(f"Removed: {folder_path}")
            except OSError as e:
                print(f"Error removing {folder_path}: {e}")
        continue

    if (
        not(os.path.isdir(f"{run_dir}/step191")) and 
        not(os.path.isdir(f"{run_dir}/step382")) and 
        not(os.path.isdir(f"{run_dir}/step954")) and 
        not(os.path.isdir(f"{run_dir}/step1908")) and 
        not(os.path.isdir(f"{run_dir}/step477")) and 
        not(os.path.isdir(f"{run_dir}/step4769")) and
        not(os.path.isdir(f"{run_dir}/step9537"))
    ):
        continue
    folder_pattern = f'{run_dir}/step*0'
    folders_to_delete = glob.glob(folder_pattern)

    # Iterate over the list of files and delete each one
    for folder_path in folders_to_delete:
        # if "_10M" in folder_path and "step800" in folder_path:
        #     continue
        # if "_20M" in folder_path and "step1600" in folder_path:
        #     continue
        try:
            # import pdb;pdb.set_trace()
            shutil.rmtree(folder_path)
            print(f"Removed: {folder_path}")
        except OSError as e:
            print(f"Error removing {folder_path}: {e}")
