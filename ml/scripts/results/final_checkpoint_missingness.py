import re
import wandb
import glob
import os
import shutil


step_count_lookup = {
    'olmo2_10M': 954,
    'olmo2_20M': 1908,
    'olmo2_50M': 4769,
    'olmo2_100M': 9537,
    'olmo2b_10M': 191,
    'olmo2b_20M': 382, # for 5xdata, use 1908
    'olmo2b_50M': 954,
}

run_top_dir = "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/models"
api = wandb.Api(timeout=60)

sweeps_missing_runs = {k: set() for k in step_count_lookup}

# Project is specified by <entity/project-name>
runs = api.runs("ml-moe/moe")
pattern = "(202\d_\d\d_\d\d-\d\d_\d\d_\d\d_(\w+)_(olmo2[b]?_\d+M|1_0B))_e(.+)x(.+)[ec](\d+,\d+|\d+)"
# pattern = "(2026_\d\d_\d\d-\d\d_\d\d_\d\d_(\w+)_olmo2[b]?_(\d+M|1_0B))_e(.+)x(.+)[ec](\d+,\d+|\d+)"
for run in runs:
    if run.state != "finished": 
        continue

    tags_to_ignore = set(["hidden", "_no_show", "notCM", "notEmatched", "notContMoE"])
    # for tag in tags_to_ignore:
    #     if tag in run.tags:
    #         continue
    if set(run.tags).intersection(tags_to_ignore):
        continue
    
    match = re.match(pattern, run.name)
    if match is None:
        continue
    if match.group(3) not in step_count_lookup:
        continue

    run_dir = f"{run_top_dir}/{match.group(1)}/{run.name}"

    if (
        not(os.path.isdir(
            f"{run_dir}/step{step_count_lookup.get(match.group(3), 0)}"
    ))):
        sweeps_missing_runs[match.group(3)].add(f"{run_top_dir}/{match.group(1)}")

print("Sweeps missing runs:")
for k in sweeps_missing_runs:
    print(f"{k}")
    for run_dir in sweeps_missing_runs[k]:
        print(f"  {run_dir}")