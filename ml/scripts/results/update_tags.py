import re
import pandas as pd
import wandb


api = wandb.Api(timeout=600)

# Project is specified by <entity/project-name>
runs = api.runs("ml-moe/moe")
bad_sweeps = set()
good_sweeps = set()



summary_list, config_list, name_list = [], [], []
for run in runs:
    # if "total_expert_dim_mult" in run.config:
    #     continue
    # if run.state != "finished": 
    #     continue
    # if "2026" not in run.name and "5CD" not in run.name:
    #     continue
    if "_no_show" in run.tags or "notCM" in run.tags or "notEmatched" in run.tags:
        continue
    # print(run.config)
    # import pdb;pdb.set_trace()
    pattern = "(202\d_\d\d_\d\d-\d\d_\d\d_\d\d_(\w+)_(olmo2[b]?_\d+M|1_0B))_e(.+)x(.+)[ec](\d+,\d+|\d+)(?:_(.+)gen|)(?:_lr=(.+)|)?"
    # pattern = "(2026_03_\d\d-\d\d_\d\d_\d\d_(\w+)_(olmo2[b]?_\d+M|1_0B))_e(.+)x(.+)[ec](\d+,\d+|\d+)(?:_(.+)gen|)(?:_lr=(.+)|)?"

    match = re.match(pattern, run.name)
    if match is None:
        continue
        
    if "hidden" not in run.tags: continue

    if "hidden" in run.tags: 
        if ("total_expert_dim_mult" in run.config or 
            ("wsd" not in run.name and ("lr" not in run.name or run.config.get("lr") == 0.0004))
        ):
            run.tags.remove("hidden"); #run.update()
    for tag in ["Data=1C", "Data=5C"]:
        if tag in run.tags:
            run.tags.remove(tag)
    if "5XD" in run.name or ("olmo_2b" in run.name and "5CD" not in run.name):
        run.tags.append("Data=1C")
    
    run.update()
   