import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import math
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from olmo_core.data.data_loader import NumpyDataLoaderConfig
    from olmo_core.data.mixes import DataMix
    from olmo_core.data.numpy_dataset import NumpyFSLDatasetConfig
    from olmo_core.data.source_subset_cache import (
        _assign_target_instances as assign_target_instances,
        _collect_source_entries as collect_source_entries,
    )
    from olmo_core.data.tokenizer import TokenizerConfig

    alt.data_transformers.disable_max_rows()


@app.cell
def _():
    output_root = Path("~/drotherm/data/olmo/olmo_mix_0824/").expanduser()
    data_root = "https://olmo-data.org"
    source_mix = DataMix.OLMoE_mix_0824
    source_sequence_length = 4096
    data_seed = 34521
    epoch = 1
    max_tokens = 10_000_000_000
    merge_gap_instances = [0, 4, 16, 64]
    curve_token_steps = [
        1_000_000,
        10_000_000,
        100_000_000,
        1_000_000_000,
        max_tokens,
    ]
    planner_workers = 32
    planner_window_sizes = [1_000, 10_000, 100_000, 500_000]
    planner_merge_extra_bytes = [0, 16_384, 65_536, 262_144, 1_048_576]
    planner_stage_budgets_gib = [0, 50, 100, 250, 500]
    subset_example_tokens = 100_000_000
    return (
        curve_token_steps,
        data_root,
        data_seed,
        epoch,
        max_tokens,
        merge_gap_instances,
        output_root,
        planner_merge_extra_bytes,
        planner_stage_budgets_gib,
        planner_window_sizes,
        planner_workers,
        source_mix,
        source_sequence_length,
        subset_example_tokens,
    )


@app.cell(hide_code=True)
def _(
    curve_token_steps,
    data_root,
    data_seed,
    epoch,
    max_tokens,
    merge_gap_instances,
    output_root,
    planner_merge_extra_bytes,
    planner_stage_budgets_gib,
    planner_window_sizes,
    planner_workers,
    source_mix,
    source_sequence_length,
    subset_example_tokens,
):
    def gib(num_bytes: int | float) -> float:
        return float(num_bytes) / (1024**3)


    def short_path(path: str) -> str:
        parts = str(path).split("/")
        if len(parts) <= 5:
            return str(path)
        return "/".join(parts[-5:])


    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        source_mix,
        tokenizer=TokenizerConfig.dolma2(),
        mix_base_dir=data_root,
        sequence_length=source_sequence_length,
        max_target_sequence_length=max(8192, source_sequence_length),
        work_dir=str(output_root / "work"),
    )
    dataset = dataset_config.build()
    loader = NumpyDataLoaderConfig(
        global_batch_size=source_sequence_length,
        work_dir=str(output_root / "work"),
        seed=data_seed,
        shuffle=True,
        num_workers=0,
    ).build(dataset)
    loader.reshuffle(epoch=epoch, in_memory=False)

    global_indices_file = str(loader._global_indices_file)
    item_size = dataset.dtype(0).itemsize
    sequence_bytes = source_sequence_length * item_size
    instances_needed = math.ceil(max_tokens / source_sequence_length)
    global_indices = np.asarray(
        loader.get_global_indices()[:instances_needed], dtype=np.int64
    )

    offsets = np.asarray(dataset.offsets, dtype=np.int64)
    starts = offsets[:, 0]
    ends = offsets[:, 1]
    file_sizes = np.asarray(dataset.file_sizes, dtype=np.int64)
    instance_capacity = ends - starts
    labels = np.asarray(
        [metadata.get("label", "unknown") for metadata in dataset.metadata],
        dtype=object,
    )
    paths = np.asarray([str(path) for path in dataset.paths], dtype=object)
    path_names = np.asarray([Path(path).name for path in paths], dtype=object)

    source_ids = np.searchsorted(ends, global_indices, side="right")
    local_indices = global_indices - starts[source_ids]

    touches = np.bincount(source_ids, minlength=len(paths)).astype(np.int64)
    requested_bytes = touches * sequence_bytes
    requested_tokens = touches * source_sequence_length
    coverage_fraction = np.divide(
        requested_bytes,
        file_sizes,
        out=np.zeros_like(requested_bytes, dtype=float),
        where=file_sizes > 0,
    )

    source_df = pd.DataFrame(
        {
            "source_id": np.arange(len(paths)),
            "label": labels,
            "path": paths,
            "path_name": path_names,
            "file_size_bytes": file_sizes,
            "file_size_gib": [gib(value) for value in file_sizes],
            "instance_capacity": instance_capacity,
            "token_capacity": instance_capacity * source_sequence_length,
            "token_capacity_billions": (instance_capacity * source_sequence_length)
            / 1e9,
            "touches": touches,
            "requested_tokens": requested_tokens,
            "requested_gib": [gib(value) for value in requested_bytes],
            "coverage_fraction": coverage_fraction,
        }
    )

    label_summary_df = (
        source_df.groupby("label", as_index=False)
        .agg(
            num_paths=("path", "size"),
            total_size_bytes=("file_size_bytes", "sum"),
            total_instances=("instance_capacity", "sum"),
            touched_paths=(
                "touches",
                lambda values: int((np.asarray(values) > 0).sum()),
            ),
            touched_instances=("touches", "sum"),
            requested_tokens=("requested_tokens", "sum"),
        )
        .assign(
            total_size_gib=lambda df: df["total_size_bytes"] / (1024**3),
            total_tokens_billions=lambda df: (
                df["total_instances"] * source_sequence_length / 1e9
            ),
            requested_gib=lambda df: (
                df["requested_tokens"] * item_size / (1024**3)
            ),
            replay_share=lambda df: df["requested_tokens"] / max_tokens,
        )
        .sort_values("total_size_bytes", ascending=False)
    )

    planner_df = (
        source_df.loc[source_df["touches"] > 0]
        .copy()
        .sort_values("requested_gib", ascending=False)
    )
    planner_label_df = (
        planner_df.groupby("label", as_index=False)
        .agg(
            touched_paths=("path", "size"),
            touches=("touches", "sum"),
            requested_tokens=("requested_tokens", "sum"),
            requested_gib=("requested_gib", "sum"),
        )
        .assign(replay_share=lambda df: df["requested_tokens"] / max_tokens)
        .sort_values("requested_gib", ascending=False)
    )

    preview_count = min(25, len(global_indices))
    access_preview_df = pd.DataFrame(
        {
            "output_rank": np.arange(preview_count),
            "global_instance_index": global_indices[:preview_count],
            "source_id": source_ids[:preview_count],
            "label": labels[source_ids[:preview_count]],
            "path_name": path_names[source_ids[:preview_count]],
            "local_instance_index": local_indices[:preview_count],
            "byte_start": local_indices[:preview_count] * sequence_bytes,
            "byte_end": (local_indices[:preview_count] + 1) * sequence_bytes - 1,
        }
    )

    planner_request_df = pd.DataFrame(
        {
            "output_position": np.arange(len(global_indices), dtype=np.int32),
            "source_id": source_ids.astype(np.int32, copy=False),
            "local_instance_index": local_indices.astype(np.int32, copy=False),
            "byte_start": (
                local_indices.astype(np.int64, copy=False) * sequence_bytes
            ),
            "request_bytes": np.full(
                len(global_indices), sequence_bytes, dtype=np.int32
            ),
        }
    )


    def ceil_div(numerator: int, denominator: int) -> int:
        return int((numerator + denominator - 1) // denominator)


    merge_rows = []
    for gap in merge_gap_instances:
        merged_requests = 0
        merged_instances = 0
        for sid in np.flatnonzero(touches > 0):
            shard_locals = np.sort(local_indices[source_ids == sid])
            if shard_locals.size == 0:
                continue
            interval_start = int(shard_locals[0])
            interval_end = interval_start
            for loc in shard_locals[1:]:
                loc = int(loc)
                if loc - interval_end - 1 <= gap:
                    interval_end = loc
                else:
                    merged_requests += 1
                    merged_instances += interval_end - interval_start + 1
                    interval_start = interval_end = loc
            merged_requests += 1
            merged_instances += interval_end - interval_start + 1
        merged_bytes = merged_instances * sequence_bytes
        merge_rows.append(
            {
                "gap_instances": gap,
                "merged_requests": merged_requests,
                "merged_gib": gib(merged_bytes),
                "exact_requested_gib": gib(len(global_indices) * sequence_bytes),
                "overfetch_gib": gib(
                    merged_bytes - len(global_indices) * sequence_bytes
                ),
            }
        )
    merge_df = pd.DataFrame(merge_rows)

    exact_requested_bytes = int(len(global_indices) * sequence_bytes)
    planner_summary_rows = [
        {
            "planner_family": "baseline_output_parallel",
            "window_size": 1,
            "merge_extra_bytes": 0,
            "stage_budget_gib": 0,
            "remote_requests": int(len(global_indices)),
            "remote_bytes_fetched_bytes": exact_requested_bytes,
            "peak_reorder_buffer_instances": 0,
            "staged_shards": 0,
            "staged_bytes_bytes": 0,
        }
    ]
    planner_locality_rows = []

    for window_size in planner_window_sizes:
        repeated_requests = 0
        unique_shards_total = 0
        window_count = 0
        merged_requests_by_budget = {
            budget: 0 for budget in planner_merge_extra_bytes
        }
        merged_extra_instances_by_budget = {
            budget: 0 for budget in planner_merge_extra_bytes
        }
        mergeable_reduction_by_budget = {
            budget: 0 for budget in planner_merge_extra_bytes
        }

        for window_start in range(0, len(global_indices), window_size):
            window_end = min(window_start + window_size, len(global_indices))
            window_sources = source_ids[window_start:window_end]
            window_locals = local_indices[window_start:window_end]
            if window_sources.size == 0:
                continue

            window_count += 1
            _, window_counts = np.unique(window_sources, return_counts=True)
            repeated_requests += int(window_counts[window_counts > 1].sum())
            unique_shards_total += int(window_counts.size)

            order = np.lexsort((window_locals, window_sources))
            sorted_sources = window_sources[order]
            sorted_locals = window_locals[order]
            group_starts = np.concatenate(
                ([0], np.flatnonzero(np.diff(sorted_sources)) + 1)
            )
            group_ends = np.concatenate((group_starts[1:], [len(sorted_sources)]))

            for group_start, group_end in zip(group_starts, group_ends):
                group_size = int(group_end - group_start)
                if group_size == 1:
                    for budget in planner_merge_extra_bytes:
                        merged_requests_by_budget[budget] += 1
                    continue

                group_locals = sorted_locals[group_start:group_end]
                gap_instances = np.diff(group_locals).astype(np.int64) - 1
                gap_bytes = gap_instances * sequence_bytes

                for budget in planner_merge_extra_bytes:
                    over_budget = gap_bytes > budget
                    over_budget_count = int(np.count_nonzero(over_budget))
                    merged_requests_by_budget[budget] += 1 + over_budget_count
                    mergeable_reduction_by_budget[budget] += (
                        group_size - 1 - over_budget_count
                    )
                    if over_budget_count != gap_instances.size:
                        merged_extra_instances_by_budget[budget] += int(
                            gap_instances[~over_budget].sum()
                        )

        avg_unique_shards_per_window = unique_shards_total / window_count
        repeated_request_fraction = repeated_requests / len(global_indices)

        planner_summary_rows.append(
            {
                "planner_family": "window_regrouping",
                "window_size": window_size,
                "merge_extra_bytes": 0,
                "stage_budget_gib": 0,
                "remote_requests": int(len(global_indices)),
                "remote_bytes_fetched_bytes": exact_requested_bytes,
                "peak_reorder_buffer_instances": window_size - 1,
                "staged_shards": 0,
                "staged_bytes_bytes": 0,
            }
        )

        for budget in planner_merge_extra_bytes:
            remote_requests = merged_requests_by_budget[budget]
            remote_bytes_fetched_bytes = int(
                exact_requested_bytes
                + merged_extra_instances_by_budget[budget] * sequence_bytes
            )
            planner_summary_rows.append(
                {
                    "planner_family": "window_merge",
                    "window_size": window_size,
                    "merge_extra_bytes": budget,
                    "stage_budget_gib": 0,
                    "remote_requests": remote_requests,
                    "remote_bytes_fetched_bytes": remote_bytes_fetched_bytes,
                    "peak_reorder_buffer_instances": window_size - 1,
                    "staged_shards": 0,
                    "staged_bytes_bytes": 0,
                }
            )
            planner_locality_rows.append(
                {
                    "window_size": window_size,
                    "merge_extra_bytes": budget,
                    "repeated_request_fraction": repeated_request_fraction,
                    "avg_unique_shards_per_window": avg_unique_shards_per_window,
                    "mergeable_request_fraction": (
                        mergeable_reduction_by_budget[budget] / len(global_indices)
                    ),
                }
            )

    touched_source_ids = np.flatnonzero(touches > 0)
    stage_scores = touches[touched_source_ids] / np.maximum(
        file_sizes[touched_source_ids] / (1024**3), 1e-9
    )
    stage_order = touched_source_ids[np.argsort(-stage_scores)]
    stage_sizes = file_sizes[stage_order]
    stage_requested_bytes = requested_bytes[stage_order]
    stage_touches = touches[stage_order]
    stage_cumulative_sizes = np.cumsum(stage_sizes, dtype=np.int64)
    stage_cumulative_requested = np.cumsum(stage_requested_bytes, dtype=np.int64)
    stage_cumulative_touches = np.cumsum(stage_touches, dtype=np.int64)

    for stage_budget_gib in planner_stage_budgets_gib:
        stage_budget_bytes = int(stage_budget_gib * (1024**3))
        staged_shards = int(
            np.searchsorted(
                stage_cumulative_sizes, stage_budget_bytes, side="right"
            )
        )
        if staged_shards > 0:
            staged_bytes_bytes = int(stage_cumulative_sizes[staged_shards - 1])
            staged_requested_bytes = int(
                stage_cumulative_requested[staged_shards - 1]
            )
            staged_touches_total = int(stage_cumulative_touches[staged_shards - 1])
        else:
            staged_bytes_bytes = 0
            staged_requested_bytes = 0
            staged_touches_total = 0

        remote_requests = (
            len(global_indices) - staged_touches_total + staged_shards
        )
        remote_bytes_fetched_bytes = (
            exact_requested_bytes - staged_requested_bytes + staged_bytes_bytes
        )
        planner_summary_rows.append(
            {
                "planner_family": "hot_shard_staging",
                "window_size": 1,
                "merge_extra_bytes": 0,
                "stage_budget_gib": stage_budget_gib,
                "remote_requests": remote_requests,
                "remote_bytes_fetched_bytes": int(remote_bytes_fetched_bytes),
                "peak_reorder_buffer_instances": 0,
                "staged_shards": staged_shards,
                "staged_bytes_bytes": staged_bytes_bytes,
            }
        )

    planner_summary_df = pd.DataFrame(planner_summary_rows).assign(
        remote_request_rounds_at_32_workers=lambda df: df["remote_requests"].map(
            lambda value: ceil_div(int(value), planner_workers)
        ),
        remote_bytes_fetched_gib=lambda df: (
            df["remote_bytes_fetched_bytes"] / (1024**3)
        ),
        overfetch_gib=lambda df: (
            (df["remote_bytes_fetched_bytes"] - exact_requested_bytes) / (1024**3)
        ),
        staged_bytes_gib=lambda df: df["staged_bytes_bytes"] / (1024**3),
        merge_budget_label=lambda df: df["merge_extra_bytes"].map(
            lambda value: f"{int(value / 1024):,} KiB"
        ),
        stage_budget_label=lambda df: df["stage_budget_gib"].map(
            lambda value: f"{int(value):,} GiB"
        ),
        window_label=lambda df: df["window_size"].map(
            lambda value: f"{int(value):,}"
        ),
    )

    planner_locality_df = pd.DataFrame(planner_locality_rows).assign(
        merge_budget_label=lambda df: df["merge_extra_bytes"].map(
            lambda value: f"{int(value / 1024):,} KiB"
        )
    )

    best_merge_configs_df = (
        planner_summary_df.loc[
            planner_summary_df["planner_family"] == "window_merge"
        ]
        .sort_values(
            [
                "window_size",
                "remote_request_rounds_at_32_workers",
                "remote_bytes_fetched_gib",
            ]
        )
        .groupby("window_size", as_index=False)
        .head(1)
        .copy()
    )
    regroup_configs_df = planner_summary_df.loc[
        planner_summary_df["planner_family"] == "window_regrouping"
    ].copy()
    staging_configs_df = planner_summary_df.loc[
        (planner_summary_df["planner_family"] == "hot_shard_staging")
        & (planner_summary_df["stage_budget_gib"] > 0)
    ].copy()
    baseline_config_df = planner_summary_df.loc[
        planner_summary_df["planner_family"] == "baseline_output_parallel"
    ].copy()

    best_merge_configs_df["conclusion_label"] = best_merge_configs_df[
        "window_size"
    ].map(lambda value: f"Best merge @ {int(value):,}")
    regroup_configs_df["conclusion_label"] = regroup_configs_df["window_size"].map(
        lambda value: f"Regroup @ {int(value):,}"
    )
    staging_configs_df["conclusion_label"] = staging_configs_df[
        "stage_budget_gib"
    ].map(lambda value: f"Stage {int(value):,} GiB")
    baseline_config_df["conclusion_label"] = "Baseline"

    planner_conclusion_df = pd.concat(
        [
            baseline_config_df.assign(family_display="Baseline"),
            regroup_configs_df.assign(family_display="Window regrouping"),
            best_merge_configs_df.assign(family_display="Best merge"),
            staging_configs_df.assign(family_display="Hot-shard staging"),
        ],
        ignore_index=True,
    ).assign(
        peak_reorder_buffer_display=lambda df: np.maximum(
            df["peak_reorder_buffer_instances"], 1
        )
    )

    curve_rows = []
    for token_count in curve_token_steps:
        instance_count = math.ceil(int(token_count) / source_sequence_length)
        unique_shards = np.unique(source_ids[:instance_count]).size
        curve_rows.append(
            {
                "token_count": int(token_count),
                "token_billions": int(token_count) / 1e9,
                "instance_count": instance_count,
                "unique_shards": int(unique_shards),
            }
        )
    unique_shards_curve_df = pd.DataFrame(curve_rows)

    planner_artifacts_df = pd.DataFrame(
        [
            {
                "artifact": "Resolved mix paths + labels",
                "used_by": "replay_cache and source_subset_cache",
                "why_it_matters": "Defines the shard universe and raw source labels",
                "needs_source_payload": False,
            },
            {
                "artifact": "Per-path file sizes + dataset offsets",
                "used_by": "replay_cache planner",
                "why_it_matters": "Maps global instance ids back to specific source shards",
                "needs_source_payload": False,
            },
            {
                "artifact": "dtype + sequence_length",
                "used_by": "replay_cache planner",
                "why_it_matters": "Determines bytes per instance and byte ranges for HTTP reads",
                "needs_source_payload": False,
            },
            {
                "artifact": "global_indices prefix",
                "used_by": "replay_cache planner",
                "why_it_matters": "Gives the exact seeded replay order for the requested token budget",
                "needs_source_payload": False,
            },
            {
                "artifact": "Source entries (label -> path capacity)",
                "used_by": "source_subset_cache",
                "why_it_matters": "Defines the hierarchy used for group/label/path apportionment",
                "needs_source_payload": False,
            },
            {
                "artifact": "Actual byte-range reads / local subset shards",
                "used_by": "both builders during materialization",
                "why_it_matters": "Only here do we touch source payload bytes",
                "needs_source_payload": True,
            },
        ]
    )

    workflow_df = pd.DataFrame(
        [
            {
                "builder": "replay_cache",
                "planning_unit": "global replay instances",
                "preserves_source_order": True,
                "local_layout": "append-only replay shards in seeded training order",
                "key_state": "global_indices prefix + replay_manifest",
            },
            {
                "builder": "source_subset_cache",
                "planning_unit": "group -> label -> path target instances",
                "preserves_source_order": False,
                "local_layout": "grouped local shards under sources/<group>/<label>/<path_key>/",
                "key_state": "source_entries + divisor-style target apportionment + subset manifest",
            },
        ]
    )

    source_entries = collect_source_entries(dataset, group_mapping=None)
    source_subset_df = pd.DataFrame(
        [
            {
                "raw_label": entry.raw_label,
                "group_name": entry.group_name,
                "num_paths": len(entry.path_entries),
                "available_instances": entry.available_instances,
                "available_tokens_billions": entry.available_instances
                * source_sequence_length
                / 1e9,
            }
            for entry in source_entries
        ]
    ).sort_values("available_instances", ascending=False)

    subset_example_labels = source_subset_df.head(4)["raw_label"].tolist()
    subset_example_target_mixture = {
        label: 1.0 / len(subset_example_labels) for label in subset_example_labels
    }
    subset_example_entries = [
        entry.model_copy(deep=True) for entry in source_entries
    ]
    subset_example_instances = math.ceil(
        subset_example_tokens / source_sequence_length
    )
    assign_target_instances(
        subset_example_entries,
        subset_example_target_mixture,
        subset_example_instances,
    )
    subset_example_df = pd.DataFrame(
        [
            {
                "raw_label": entry.raw_label,
                "group_name": entry.group_name,
                "num_paths": len(entry.path_entries),
                "available_tokens_billions": entry.available_instances
                * source_sequence_length
                / 1e9,
                "target_tokens_millions": entry.target_instances
                * source_sequence_length
                / 1e6,
            }
            for entry in subset_example_entries
            if entry.target_instances > 0
        ]
    ).sort_values("target_tokens_millions", ascending=False)

    summary = {
        "num_paths": len(paths),
        "num_labels": int(label_summary_df["label"].nunique()),
        "dtype": dataset.dtype.__name__,
        "item_size": item_size,
        "sequence_bytes": sequence_bytes,
        "total_source_gib": gib(file_sizes.sum()),
        "instances_needed": instances_needed,
        "requested_gib": gib(len(global_indices) * sequence_bytes),
        "touched_paths": int((touches > 0).sum()),
        "global_indices_file": global_indices_file,
        "planner_workers": planner_workers,
        "subset_example_labels": subset_example_labels,
    }
    return (
        access_preview_df,
        label_summary_df,
        merge_df,
        planner_artifacts_df,
        planner_conclusion_df,
        planner_df,
        planner_label_df,
        planner_locality_df,
        planner_summary_df,
        source_df,
        source_subset_df,
        subset_example_df,
        summary,
        unique_shards_curve_df,
        workflow_df,
    )


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    # Replay Cache Download Anatomy

    This notebook is a guided walk from **dataset shape** to **planner state** to **download tradeoffs**.

    Read it in lanes:

    - **Column 1** introduces the questions and the mental model.
    - **Columns 2-4** explain what `replay_cache` already knows before payload reads: source labels and shard sizes, then replay-prefix demand, then concrete byte-range examples and merge tradeoffs.
    - **Columns 5-6** switch to `source_subset_cache`, which plans from label/path capacity instead of `global_indices`.

    Two invariants matter throughout:

    1. `replay_cache` preserves the seeded training order, so it starts from a `global_indices` prefix and maps those instances back to shard-local byte ranges.
    2. `source_subset_cache` does **not** preserve that order; it apportions a target budget across `group -> label -> path` capacity and then samples deterministically within each path.

    As you read, the main questions are:

    - how large and heterogeneous is the source mix?
    - how quickly does the replay prefix spread across shard files?
    - which planning artifacts exist before any source-byte downloads?
    - where do the two builders diverge in what they optimize for?
    """)
    return


@app.cell(column=2, hide_code=True)
def _(label_summary_df, max_tokens, source_mix, summary):
    summary_table = label_summary_df[
        [
            "label",
            "num_paths",
            "total_size_gib",
            "total_tokens_billions",
            "touched_paths",
            "requested_gib",
            "replay_share",
        ]
    ].sort_values("total_size_gib", ascending=False)

    mo.vstack(
        [
            mo.md(f"""
    ## Planner Inputs Available Before Payload Reads

    **What this is.** This block summarizes the metadata that exists before `replay_cache` reads source payload bytes: the resolved mix, shard universe, dtype, bytes per fixed-sequence-length instance, and the saved permutation prefix.

    **Why it matters.** These are the ingredients the planner needs to convert replay positions into concrete shard-local byte ranges. Nothing here requires a full source download.

    **What to notice.**
    - `source_mix`: `{source_mix.value}`
    - source paths in mix: `{summary["num_paths"]:,}`
    - unique raw labels: `{summary["num_labels"]:,}`
    - dataset dtype: `{summary["dtype"]}` (`{summary["item_size"]}` bytes/token)
    - bytes per 4096-token instance: `{summary["sequence_bytes"]:,}`
    - total source size across all shard files: `{summary["total_source_gib"]:,.1f} GiB`
    - replay request: `{max_tokens / 1e9:,.1f}B` tokens -> `{summary["instances_needed"]:,}` instances -> `{summary["requested_gib"]:,.1f} GiB` of exact payload
    - touched shard files under this replay prefix: `{summary["touched_paths"]:,}` / `{summary["num_paths"]:,}`
    - cached permutation file: `{summary["global_indices_file"]}`

    The table below is the first bridge from **raw source capacity** to **replay demand** by label.
    """),
            summary_table,
        ]
    )
    return


@app.cell(column=3, hide_code=True)
def _(label_summary_df, source_df):
    top_labels = label_summary_df.head(20).copy()
    label_size_chart = (
        alt.Chart(top_labels)
        .mark_bar()
        .encode(
            x=alt.X("total_size_gib:Q", title="Source size (GiB)"),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=[
                "label",
                "num_paths",
                alt.Tooltip("total_size_gib:Q", format=",.1f"),
                alt.Tooltip("total_tokens_billions:Q", format=",.2f"),
            ],
        )
        .properties(
            title="Largest raw labels in the resolved source mix",
            height=420,
            width=700,
        )
    )

    shard_size_chart = (
        alt.Chart(source_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "file_size_gib:Q",
                bin=alt.Bin(maxbins=40),
                title="Per-shard size (GiB)",
            ),
            y=alt.Y("count():Q", title="Shard files"),
            tooltip=[alt.Tooltip("count():Q", title="shards")],
        )
        .properties(
            title="Shard-size distribution across all source files",
            height=220,
            width=700,
        )
    )

    source_mix_overview = alt.vconcat(label_size_chart, shard_size_chart)

    mo.vstack(
        [
            mo.md("""
    ## Source Mix Structure

    **What this is.** These charts describe the full resolved source mix before replay planning narrows anything down.

    **Why it matters.** Download strategy depends on whether the dataset is dominated by a few large pools or spread across many differently sized shard files.

    **What to notice.**
    - which raw labels dominate total source bytes
    - whether shard files cluster tightly or span a wide size range
    - whether the replay planner later seems proportional to raw capacity or skewed toward a subset of labels
    """),
            source_mix_overview,
        ]
    )
    return label_size_chart, shard_size_chart


@app.cell(hide_code=True)
def _(planner_df, planner_label_df, unique_shards_curve_df):
    planner_label_chart = (
        alt.Chart(planner_label_df.head(20))
        .mark_bar()
        .encode(
            x=alt.X("requested_gib:Q", title="Requested payload (GiB)"),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=[
                "label",
                "touched_paths",
                "touches",
                alt.Tooltip("requested_gib:Q", format=",.2f"),
                alt.Tooltip("replay_share:Q", format=".2%"),
            ],
        )
        .properties(
            title="Replay-prefix demand by raw label", height=420, width=700
        )
    )

    coverage_chart = (
        alt.Chart(planner_df)
        .mark_circle(size=70, opacity=0.6)
        .encode(
            x=alt.X("file_size_gib:Q", title="Shard size (GiB)"),
            y=alt.Y("requested_gib:Q", title="Requested payload from shard (GiB)"),
            color=alt.Color("label:N", legend=None),
            tooltip=[
                "label",
                "path_name",
                "touches",
                alt.Tooltip("coverage_fraction:Q", format=".2%"),
                alt.Tooltip("requested_gib:Q", format=",.3f"),
            ],
        )
        .properties(
            title="Touched shards under the replay prefix", height=360, width=700
        )
    )

    unique_shards_chart = (
        alt.Chart(unique_shards_curve_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "token_billions:Q", title="Replay budget (billions of tokens)"
            ),
            y=alt.Y("unique_shards:Q", title="Unique touched shards"),
            tooltip=[
                alt.Tooltip("token_billions:Q", format=",.3f"),
                "instance_count",
                "unique_shards",
            ],
        )
        .properties(
            title="How quickly the replay prefix spreads across source shards",
            height=240,
            width=700,
        )
    )

    replay_planner_overview = alt.vconcat(
        planner_label_chart, coverage_chart, unique_shards_chart
    )

    mo.vstack(
        [
            mo.md("""
    ## Replay Planner Demand

    **What this is.** This section takes the `global_indices` prefix and asks where those instances land in the source mix.

    **Why it matters.** This is the first place where the future download pattern becomes visible: which labels dominate the exact payload, how many individual shards become hot, and how quickly the replay prefix fans out across the full shard universe.

    **What to notice.**
    - whether payload demand is concentrated in a few labels or widely distributed
    - whether large shards also dominate requested bytes, or whether many small touches matter
    - how quickly the unique-shard curve saturates as replay budget grows
    """),
            replay_planner_overview,
        ]
    )
    return coverage_chart, planner_label_chart, unique_shards_chart


@app.cell(column=4, hide_code=True)
def _(access_preview_df):
    mo.vstack(
        [
            mo.md("""
    ## Concrete Access Preview

    **What this is.** These are the first planned replay reads after mapping the shuffled prefix back to shard-local positions.

    **Why it matters.** This table makes the planner's abstraction concrete: each output rank corresponds to one global instance index, one source shard, and one exact byte range.

    **What to notice.**
    - output order is preserved even though source shards repeat irregularly
    - local instance indices translate directly into byte offsets
    - this is the level where HTTP range requests or local cached reads would be issued
    """),
            access_preview_df,
        ]
    )
    return


@app.cell(hide_code=True)
def _(merge_df):
    merge_request_chart = (
        alt.Chart(merge_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("gap_instances:Q", title="Merge gap (instances)"),
            y=alt.Y("merged_requests:Q", title="Merged HTTP requests"),
            tooltip=[
                "gap_instances",
                "merged_requests",
                alt.Tooltip("merged_gib:Q", format=",.2f"),
                alt.Tooltip("overfetch_gib:Q", format=",.2f"),
            ],
        )
        .properties(
            title="Request-count sensitivity to merging nearby intervals",
            height=220,
            width=340,
        )
    )

    overfetch_chart = (
        alt.Chart(merge_df)
        .mark_line(point=True, color="#b91c1c")
        .encode(
            x=alt.X("gap_instances:Q", title="Merge gap (instances)"),
            y=alt.Y("overfetch_gib:Q", title="Extra bytes fetched (GiB)"),
            tooltip=[
                "gap_instances",
                alt.Tooltip("overfetch_gib:Q", format=",.2f"),
            ],
        )
        .properties(title="Overfetch tradeoff", height=220, width=340)
    )

    merge_tradeoff = alt.hconcat(merge_request_chart, overfetch_chart)

    mo.vstack(
        [
            mo.md("""
    ## Merge-Gap Tradeoff

    **What this is.** A simple planner thought experiment: if nearby instance ranges on the same shard are merged into larger reads, how much request count drops and how much extra payload gets fetched.

    **Why it matters.** This is the core tradeoff behind any local interval cache or block-prefetch scheme.

    **What to notice.**
    - small merge gaps reduce requests modestly with limited overfetch
    - aggressive gaps can explode extra transferred bytes
    - the useful operating region depends on whether the access pattern has enough within-shard locality to justify larger reads
    """),
            merge_tradeoff,
        ]
    )
    return merge_request_chart, overfetch_chart


@app.cell(hide_code=True)
def _(planner_summary_df, summary):
    planner_summary_table = planner_summary_df.sort_values(
        [
            "planner_family",
            "window_size",
            "merge_extra_bytes",
            "stage_budget_gib",
        ]
    )[
        [
            "planner_family",
            "window_size",
            "merge_budget_label",
            "stage_budget_label",
            "remote_requests",
            "remote_request_rounds_at_32_workers",
            "remote_bytes_fetched_gib",
            "overfetch_gib",
            "peak_reorder_buffer_instances",
            "staged_shards",
        ]
    ]

    mo.vstack(
        [
            mo.md(f"""
    ## Planner Simulation Testbed

    **What this is.** These rows compare four offline planner families on the same replay prefix: output-order parallel fetch, windowed regrouping, byte-budgeted merging, and opportunistic hot-shard staging.

    **Why it matters.** This is the decision surface for whether it is worth getting smarter than simple parallel sparse reads.

    **What to notice.**
    - all latency-proxy rows assume `{summary["planner_workers"]}` parallel remote workers
    - regrouping alone should not change request count or bytes; it only exposes locality that smarter planners might exploit
    - merging and staging are the only planners allowed to trade extra fetched bytes for fewer remote request rounds
    """),
            planner_summary_table,
        ]
    )
    return


@app.cell(hide_code=True)
def _(planner_conclusion_df, planner_locality_df, planner_summary_df, summary):
    log_window_scale = alt.Scale(type="log", base=10)

    repeated_request_chart = (
        alt.Chart(planner_locality_df)
        .mark_line(point=True, color="#1d4ed8")
        .encode(
            x=alt.X(
                "window_size:Q",
                title="Lookahead window (output instances)",
                scale=log_window_scale,
            ),
            y=alt.Y(
                "repeated_request_fraction:Q",
                title="Requests sharing a shard within window",
                axis=alt.Axis(format=".0%"),
            ),
            tooltip=[
                "window_size",
                alt.Tooltip("repeated_request_fraction:Q", format=".1%"),
                alt.Tooltip("avg_unique_shards_per_window:Q", format=",.1f"),
            ],
        )
        .properties(
            title="Windowed regrouping diagnostics: repeated shard hits",
            height=220,
            width=700,
        )
    )

    mergeable_fraction_chart = (
        alt.Chart(planner_locality_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "window_size:Q",
                title="Lookahead window (output instances)",
                scale=log_window_scale,
            ),
            y=alt.Y(
                "mergeable_request_fraction:Q",
                title="Potential request reduction under merge budget",
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "merge_budget_label:N", title="Merge extra-byte budget"
            ),
            tooltip=[
                "window_size",
                "merge_budget_label",
                alt.Tooltip("mergeable_request_fraction:Q", format=".1%"),
            ],
        )
        .properties(
            title="Windowed regrouping diagnostics: merge opportunity",
            height=220,
            width=700,
        )
    )

    avg_unique_shards_chart = (
        alt.Chart(planner_locality_df.drop_duplicates("window_size"))
        .mark_line(point=True, color="#b45309")
        .encode(
            x=alt.X(
                "window_size:Q",
                title="Lookahead window (output instances)",
                scale=log_window_scale,
            ),
            y=alt.Y(
                "avg_unique_shards_per_window:Q",
                title="Average unique shards per window",
            ),
            tooltip=[
                "window_size",
                alt.Tooltip("avg_unique_shards_per_window:Q", format=",.1f"),
            ],
        )
        .properties(
            title="Windowed regrouping diagnostics: shard fanout",
            height=220,
            width=700,
        )
    )

    planner_window_diagnostics = alt.vconcat(
        repeated_request_chart,
        mergeable_fraction_chart,
        avg_unique_shards_chart,
    )

    merge_frontier_df = planner_summary_df.loc[
        planner_summary_df["planner_family"] == "window_merge"
    ].copy()
    planner_merge_frontier_chart = (
        alt.Chart(merge_frontier_df)
        .mark_circle(size=110, opacity=0.75)
        .encode(
            x=alt.X(
                "remote_bytes_fetched_gib:Q", title="Remote bytes fetched (GiB)"
            ),
            y=alt.Y(
                "remote_request_rounds_at_32_workers:Q",
                title=f"Remote request rounds at {summary['planner_workers']} workers",
            ),
            color=alt.Color("window_label:N", title="Window size"),
            shape=alt.Shape(
                "merge_budget_label:N", title="Merge extra-byte budget"
            ),
            tooltip=[
                "window_size",
                "merge_budget_label",
                "remote_requests",
                "remote_request_rounds_at_32_workers",
                alt.Tooltip("remote_bytes_fetched_gib:Q", format=",.2f"),
                alt.Tooltip("overfetch_gib:Q", format=",.2f"),
                "peak_reorder_buffer_instances",
            ],
        )
        .properties(
            title="Merging frontier: extra bytes versus fewer remote rounds",
            height=280,
            width=700,
        )
    )

    staging_frontier_df = planner_summary_df.loc[
        planner_summary_df["planner_family"] == "hot_shard_staging"
    ].copy()
    planner_staging_frontier_chart = (
        alt.Chart(staging_frontier_df)
        .mark_circle(size=130, opacity=0.8)
        .encode(
            x=alt.X(
                "remote_bytes_fetched_gib:Q", title="Remote bytes fetched (GiB)"
            ),
            y=alt.Y(
                "remote_request_rounds_at_32_workers:Q",
                title=f"Remote request rounds at {summary['planner_workers']} workers",
            ),
            color=alt.Color("stage_budget_label:N", title="Stage budget"),
            tooltip=[
                "stage_budget_label",
                "staged_shards",
                alt.Tooltip("staged_bytes_gib:Q", format=",.1f"),
                "remote_requests",
                "remote_request_rounds_at_32_workers",
                alt.Tooltip("remote_bytes_fetched_gib:Q", format=",.2f"),
                alt.Tooltip("overfetch_gib:Q", format=",.2f"),
            ],
        )
        .properties(
            title="Hot-shard staging frontier: local budget versus remote rounds",
            height=280,
            width=700,
        )
    )

    planner_conclusion_chart = (
        alt.Chart(planner_conclusion_df)
        .mark_circle(opacity=0.85)
        .encode(
            x=alt.X(
                "remote_bytes_fetched_gib:Q", title="Remote bytes fetched (GiB)"
            ),
            y=alt.Y(
                "remote_request_rounds_at_32_workers:Q",
                title=f"Remote request rounds at {summary['planner_workers']} workers",
            ),
            color=alt.Color("family_display:N", title="Planner family"),
            shape=alt.Shape("family_display:N", title="Planner family"),
            size=alt.Size(
                "peak_reorder_buffer_display:Q",
                title="Reorder buffer size proxy",
                scale=alt.Scale(range=[60, 700]),
            ),
            tooltip=[
                "family_display",
                "conclusion_label",
                "window_size",
                "merge_budget_label",
                "stage_budget_label",
                "remote_requests",
                "remote_request_rounds_at_32_workers",
                alt.Tooltip("remote_bytes_fetched_gib:Q", format=",.2f"),
                alt.Tooltip("overfetch_gib:Q", format=",.2f"),
                "peak_reorder_buffer_instances",
                "staged_shards",
            ],
        )
        .properties(
            title="Conclusion plot: planner tradeoffs on the replay prefix",
            height=320,
            width=700,
        )
    )

    mo.vstack(
        [
            mo.md(f"""
    ## Planner Conclusions

    **What this is.** These plots compare the smarter planner families under the same `{summary["planner_workers"]}`-worker latency proxy.

    **Why it matters.** They show whether bounded regrouping, merging, or hot-shard staging buys a meaningful drop in remote request rounds, and what extra bytes or reorder pressure each strategy costs.

    **What to notice.**
    - if regrouping diagnostics stay weak as the window grows, the source-efficient planner has little locality to exploit
    - if the merging frontier moves right much faster than it moves down, extra fetched bytes are buying too little latency improvement
    - if staging points move down materially with moderate byte budgets, hot-shard staging becomes the first smart policy worth implementing after plain parallel reads
    """),
            planner_window_diagnostics,
            planner_merge_frontier_chart,
            planner_staging_frontier_chart,
            planner_conclusion_chart,
        ]
    )
    return (
        planner_conclusion_chart,
        planner_merge_frontier_chart,
        planner_staging_frontier_chart,
        planner_window_diagnostics,
    )


@app.cell(hide_code=True)
def _(workflow_df):
    mo.vstack(
        [
            mo.md("""
    ## Builder Comparison

    **What this is.** A compact side-by-side of the two builders in this repo.

    **Why it matters.** The rest of the notebook only makes sense if you keep straight which invariant each builder preserves.

    **What to notice.**
    - `replay_cache` plans in seeded replay order
    - `source_subset_cache` plans in mixture space first, then samples locally
    - the local outputs they materialize are organized for different downstream needs
    """),
            workflow_df,
        ]
    )
    return


@app.cell(column=5, hide_code=True)
def _(source_subset_df):
    mo.vstack(
        [
            mo.md("""
    ## source_subset_cache Source Entries

    **What this is.** The top rows of the `source_entries` view that `source_subset_cache` builds before any target mixture is applied.

    **Why it matters.** This is the hierarchy the subset planner sees: raw labels, their optional grouped names, and the amount of path-level capacity available under each label.

    **What to notice.**
    - this starts from capacity, not replay order
    - labels can be regrouped before apportionment
    - path counts hint at how much room the subset builder has to spread allocations within a label
    """),
            source_subset_df.head(25),
        ]
    )
    return


@app.cell(hide_code=True)
def _(source_subset_df, subset_example_df, summary):
    subset_capacity_chart = (
        alt.Chart(source_subset_df.head(20))
        .mark_bar()
        .encode(
            x=alt.X(
                "available_tokens_billions:Q", title="Available tokens (billions)"
            ),
            y=alt.Y("raw_label:N", sort="-x", title=None),
            tooltip=[
                "raw_label",
                "group_name",
                "num_paths",
                alt.Tooltip("available_tokens_billions:Q", format=",.2f"),
            ],
        )
        .properties(
            title="What source_subset_cache sees before apportionment",
            height=420,
            width=700,
        )
    )

    subset_example_chart = (
        alt.Chart(subset_example_df)
        .mark_bar(color="#2563eb")
        .encode(
            x=alt.X(
                "target_tokens_millions:Q",
                title="Allocated tokens in toy subset run (millions)",
            ),
            y=alt.Y("raw_label:N", sort="-x", title=None),
            tooltip=[
                "raw_label",
                "num_paths",
                alt.Tooltip("target_tokens_millions:Q", format=",.2f"),
            ],
        )
        .properties(
            title=f"Toy source_subset_cache allocation: equal mixture over {', '.join(summary['subset_example_labels'])}",
            height=220,
            width=700,
        )
    )

    subset_planner_overview = alt.vconcat(
        subset_capacity_chart, subset_example_chart
    )

    mo.vstack(
        [
            mo.md("""
    ## source_subset_cache Planning View

    **What this is.** The first chart shows the available capacity by raw label before apportionment. The second shows a toy allocation after a small target mixture is applied.

    **Why it matters.** This is the alternative planning lens to replay order: instead of asking which shuffled instances come next, the builder asks how to divide a target budget across labels and then across concrete paths.

    **What to notice.**
    - large available labels are not automatically assigned proportionally in the toy run
    - the planner can deliberately reshape the final local subset's composition
    - path multiplicity matters because allocation eventually has to land on concrete files
    """),
            subset_planner_overview,
        ]
    )
    return subset_capacity_chart, subset_example_chart


@app.cell(column=6, hide_code=True)
def _(planner_artifacts_df, subset_example_tokens):
    mo.vstack(
        [
            mo.md(f"""
    ## source_subset_cache Artifacts

    The newer source-subset path does **not** start from `global_indices`. Instead, it builds `source_entries` at the `raw_label -> path` hierarchy and then uses divisor-style apportionment to assign target instances.

    The toy chart above uses an equal-weight target mixture over the four largest labels in this mix for a `{subset_example_tokens / 1e6:,.0f}M` token subset budget.

    **How to read the artifact table below.**
    - the first rows are planning artifacts available before payload materialization
    - the replay-specific rows exist to preserve seeded order and exact byte ranges
    - the subset-specific rows exist to preserve target composition and append-friendly allocation behavior
    - only the final row actually requires touching source payload bytes
    """),
            planner_artifacts_df,
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    coverage_chart,
    label_size_chart,
    merge_request_chart,
    overfetch_chart,
    planner_conclusion_chart,
    planner_label_chart,
    planner_merge_frontier_chart,
    planner_staging_frontier_chart,
    planner_window_diagnostics,
    shard_size_chart,
    subset_capacity_chart,
    subset_example_chart,
    unique_shards_chart,
):
    charts_to_save = {
        "source_mix_structure": alt.vconcat(label_size_chart, shard_size_chart),
        "replay_planner_demand": alt.vconcat(
            planner_label_chart,
            coverage_chart,
            unique_shards_chart,
        ),
        "merge_tradeoff": alt.hconcat(merge_request_chart, overfetch_chart),
        "planner_window_diagnostics": planner_window_diagnostics,
        "planner_merge_frontier": planner_merge_frontier_chart,
        "planner_staging_frontier": planner_staging_frontier_chart,
        "planner_conclusion": planner_conclusion_chart,
        "source_subset_planning": alt.vconcat(
            subset_capacity_chart,
            subset_example_chart,
        ),
    }

    saved_plot_paths = []
    if mo.app_meta().mode == "script":
        output_dir = Path(__file__).resolve().parent / "data_download_artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, chart in charts_to_save.items():
            path = output_dir / f"{name}.html"
            chart.save(path)
            saved_plot_paths.append(str(path))
    return


@app.cell(column=7, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
