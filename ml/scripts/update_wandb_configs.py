"""Update wandb run configs and tags based on run name."""

import argparse

import wandb

# Rules for setting config values based on run name substrings.
# Each entry: (substring_to_match, config_key, config_value)
CONFIG_RULES = [
    ("200M", "model_size_name", "200M"),
    ("80M", "model_size_name", "80M"),
]

# Rules for adding tags based on run name substrings.
# Each entry: (substring_to_match, tag_to_add)
TAG_RULES = [
    ("moe64", "MoE64"),
]


def main(entity: str, project: str, dry_run: bool = False):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    updated = 0
    for run in runs:
        name = run.name or ""
        changes = []

        # Apply config rules (first match per config_key wins).
        seen_keys = set()
        for substring, config_key, config_value in CONFIG_RULES:
            if config_key in seen_keys:
                continue
            if substring in name and run.config.get(config_key) != config_value:
                run.config[config_key] = config_value
                changes.append(f"config {config_key}={config_value}")
                seen_keys.add(config_key)

        # Apply tag rules.
        existing_tags = set(run.tags or [])
        for substring, tag in TAG_RULES:
            if substring in name and tag not in existing_tags:
                run.tags.append(tag)
                changes.append(f"tag +{tag}")

        if not changes:
            continue

        if dry_run:
            print(f"[dry run] {run.name}: {', '.join(changes)}")
        else:
            run.update()
            print(f"Updated {run.name}: {', '.join(changes)}")

        updated += 1

    print(f"\n{'Would update' if dry_run else 'Updated'} {updated} runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update wandb run configs and tags based on run name")
    parser.add_argument("--entity", type=str, default="ml-moe")
    parser.add_argument("--project", type=str, default="data_rep_moe")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without applying them")
    args = parser.parse_args()

    main(entity=args.entity, project=args.project, dry_run=args.dry_run)
