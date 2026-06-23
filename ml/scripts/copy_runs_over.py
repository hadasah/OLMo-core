import wandb

# Set your API key
wandb.login()

# Set the source and destination projects
src_entity = "ml-moe"
src_project = "moe"
dst_entity = "ml-moe"
dst_project = "slicing-and-dicing"

# Initialize the wandb API
api = wandb.Api()

# Get the runs from the source project
runs = api.runs(f"{src_entity}/{src_project}")

# Iterate through the runs and copy them to the destination project

for run in runs:
    if "hidden" in run.tags or "_no_show" in run.tags or "notCM" in run.tags or "notEmatched" in run.tags:
    continue
    
    # Get the run history and files
    history = run.history()
    files = run.files()



    new_run_config = copy(run.config)
    new_run_config["original_run_name"] = run.name
    new_run_name = 

    # Create a new run in the destination project
    new_run = wandb.init(project=dst_project, entity=dst_entity, config=new_run_config, name=new_run_name, resume="allow")
    
    # Log the history to the new run
    for index, row in history.iterrows():
        new_run.log(row.to_dict())

    # Upload the files to the new run
    for file in files:
        file.download(replace=True)
        new_run.save(file.name,policy = "now")

    # Finish the new run
    new_run.finish()