import re
import pandas as pd
import wandb


api = wandb.Api(timeout=600)

# Project is specified by <entity/project-name>
runs = api.runs("ml-moe/moe")
bad_sweeps = set()
good_sweeps = set()

MODEL_NAME_LOOKUP = {
    'olmo2_10M': ('80M', 81898320),
    'olmo2_20M': ('110M', 113593968),
    'olmo2_50M': ('200M', 194012800),
    'olmo2_100M': ('300M', 299933504),
    'olmo2b_10M': ('10M', 9745008),
    'olmo2b_20M': ('20M', 19859040),
    'olmo2b_50M': ('50M', 52782000),
}

summary_list, config_list, name_list = [], [], []
for run in runs:
    # if "total_expert_dim_mult" in run.config:
    #     continue
    # if run.state != "finished": 
    #     continue
    # if "2026" not in run.name and "5CD" not in run.name:
    #     continue
    if "hidden" in run.tags or "_no_show" in run.tags or "notCM" in run.tags or "notEmatched" in run.tags:
        continue
    # print(run.config)
    # import pdb;pdb.set_trace()

    all_dims = []
    expt_dims = []
    pattern = "(202\d_\d\d_\d\d-\d\d_\d\d_\d\d_(\w+)_(olmo2[b]?_\d+M|1_0B))_e(.+)x(.+)[ec](\d+,\d+|\d+)(?:_(.+)gen|)(?:_lr=(.+)|)?"
    # pattern = "(2026_03_\d\d-\d\d_\d\d_\d\d_(\w+)_(olmo2[b]?_\d+M|1_0B))_e(.+)x(.+)[ec](\d+,\d+|\d+)(?:_(.+)gen|)(?:_lr=(.+)|)?"

    match = re.match(pattern, run.name)
    if match is None:
        print(f"==No regex match for run {run.name}")
        # if "hidden" not in run.tags: run.tags.append("hidden"); run.update()
        continue

    try:
        num_expts = run.config["model"]["block"]["feed_forward_moe"]["num_experts_list"]
        hidden_sizes = run.config["model"]["block"]["feed_forward_moe"]["hidden_sizes_list"]
        ffn_hsz = float(run.config["model"]["d_model"]) * 4
        # routers_list = run.config["model"]["block"]["feed_forward_moe"]["routers_list"]
        generalist_hsz = run.config.get("model", {}).get("block", {}).get("feed_forward", {}).get("hidden_size")
        for num_expt, hsz in zip(num_expts, hidden_sizes):
            all_dims.append(float(num_expt) * float(hsz))
            expt_dims.append(float(hsz)/ffn_hsz)
    except:
        generalist_hsz = run.config.get("model", {}).get("block", {}).get("feed_forward", {}).get("hidden_size")
        ffn_hsz = generalist_hsz
        if ffn_hsz is not None and "x1c1" in run.name:
            all_dims.append(ffn_hsz)
        else:
            print(f"==No experts found  for run {run.name}: {run.config['model']['block']}, {run.config['model']['d_model']}")
            print(run.tags)
            # if "hidden" not in run.tags: run.tags.append("hidden"); run.update()
            continue

    try:
        generalist_mult = float(match.group(7))
    except:
        generalist_mult = 0.0
    total_dim = sum(all_dims)

    if total_dim <= ffn_hsz and "2026_01_1" not in run.name:
        if "e1x1c1" not in run.name and "e2x0.5c2" not in run.name and "e4x0.25c4" not in run.name and "e8x0.125c8" not in run.name:
            print("Total expert dim less than ffn_hsz", run.name)
            # bad_sweeps.add(match.group(1))
            # if "hidden" not in run.tags: run.tags.append("hidden"); run.update()
            import pdb;pdb.set_trace()
            continue

    try:
        num_expts = match.group(4).split(',')
        hidden_mults = match.group(5).split(',')
        total_mult_2 = sum([float(e) * float(h) for e, h in zip(num_expts, hidden_mults)])
    except:
        total_mult_2 = 0.0
        print(f"==No total_mult_2 found for run {run.name}")
        
    if total_mult_2 != total_dim / ffn_hsz and 'e64x0.0625e16' not in run.name and 'e64x0.0625e32' not in run.name:
        # print(f"==Mismatch in total expert dim for run {run.name}: regex {total_mult_2}, computed {total_dim / ffn_hsz}")
        bad_sweeps.add(match.group(1))
        import pdb;pdb.set_trace()
        continue
        # if "hidden" not in run.tags: run.tags.append("hidden"); run.update()

    if generalist_mult != 0.0:
        if generalist_hsz / ffn_hsz != generalist_mult:
            print(f"==Mismatch in generalist mult for run {run.name}: regex {generalist_mult}, computed {generalist_hsz / ffn_hsz}")
            bad_sweeps.add(match.group(1))
            import pdb;pdb.set_trace()
            continue
            # if "hidden" not in run.tags: run.tags.append("hidden"); run.update()

    # import pdb;pdb.set_trace()
    good_sweeps.add(match.group(1))
    model_size_name, num_active_params = MODEL_NAME_LOOKUP.get(match.group(3), (None, None))
    run.config["model_size_name"] = model_size_name
    run.config["num_active_params"] = num_active_params
    run.config["total_expert_dim_mult"] = total_dim / ffn_hsz
    run.config["generalist_dim_mult"] = generalist_mult
    run.config["total_moe_dim_mult"] = (total_dim ) / ffn_hsz + generalist_mult
    run.config["expert_mults_list"] = expt_dims

    bias_gamma = run.config["model"]["block"].get("feed_forward_moe", {}).get("routers_list", [{}])[0].get("bias_gamma")
    if bias_gamma:
        run.config["moe_bias_gamma"] = bias_gamma

    if (total_dim ) / ffn_hsz + generalist_mult != int((total_dim ) / ffn_hsz + generalist_mult):
        run.tags.append("notEmatched") 
        
    run.update()

print(bad_sweeps)
# print(good_sweeps)
# print(bad_sweeps - good_sweeps)