import os
import subprocess


MODELS_FOLDER = '/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/models'
ids = [
    '2026_02_12-14_54_00_dropless_lowlb_olmo2b_10M/2026_02_12-14_54_00_dropless_lowlb_olmo2b_10M_e7x0.25c3_0.25gen'
]

for id in ids:

    for root, dirnames, filenames in os.walk(os.path.join(MODELS_FOLDER, id, 'wandb/wandb')):
        # new_id = id.split('/')[1] + "_repl"
        new_id = id.split('/')[1]
        for subfolder in dirnames:
            # Check if the subfolder starts with the desired prefix
            if subfolder.startswith("run-"):
                try:
                    run_path = os.path.join(root, subfolder)
        # Execute the wandb sync command
                    subprocess.check_call(["wandb", "sync", run_path, "--id", new_id])
                    print(f"Successfully synced runs in {run_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to sync W&B runs: {e}")
                except FileNotFoundError:
                    print("The 'wandb' command was not found. Please ensure the W&B CLI is installed and in your PATH.")
