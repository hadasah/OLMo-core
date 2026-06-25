"""Update wandb run configs to add model_size_name based on run name."""

import argparse

import wandb


def main(entity: str, project: str, dry_run: bool = False):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    updated = 0
    for run in runs:
        name = run.name or ""
        model_size_name = None

        if "200M" in name:
            model_size_name = "200M"
        elif "80M" in name:
            model_size_name = "80M"

        if model_size_name is None:
            continue

        existing = run.config.get("model_size_name")
        if existing == model_size_name:
            continue

        if dry_run:
            print(f"[dry run] {run.name}: model_size_name={model_size_name}")
        else:
            run.config["model_size_name"] = model_size_name
            run.update()
            print(f"Updated {run.name}: model_size_name={model_size_name}")

        updated += 1

    print(f"\n{'Would update' if dry_run else 'Updated'} {updated} runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update wandb run configs with model_size_name")
    parser.add_argument("--entity", type=str, default="ml-moe")
    parser.add_argument("--project", type=str, default="data_rep_moe")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without applying them")
    args = parser.parse_args()

    main(entity=args.entity, project=args.project, dry_run=args.dry_run)
