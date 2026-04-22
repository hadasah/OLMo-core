"""
Helpers for the ``data_download`` marimo notebook: dataset setup, planner simulation, and subset views.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from olmo_core.data.data_loader import NumpyDataLoaderConfig
from olmo_core.data.mixes import DataMix
from olmo_core.data.numpy_dataset import NumpyFSLDataset, NumpyFSLDatasetConfig
from olmo_core.data.source_subset_cache import (
    assign_target_instances,
    collect_source_entries,
)
from olmo_core.data.tokenizer import TokenizerConfig


def default_olmo_output_root() -> Path:
    """
    Default directory for OLMo mix work/cache data.

    Set environment variable ``OLMO_OUTPUT_ROOT`` to override (any path string;
    ``~`` is expanded). Default: ``~/drotherm/data/olmo/olmo_mix_0824/``.
    """
    return Path(
        os.environ.get(
            "OLMO_OUTPUT_ROOT",
            str(Path.home() / "drotherm" / "data" / "olmo" / "olmo_mix_0824"),
        )
    ).expanduser()


def gib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024**3)


def ceil_div(numerator: int, denominator: int) -> int:
    return int((numerator + denominator - 1) // denominator)


def build_dataset_loader(
    *,
    data_root: str,
    data_seed: int,
    epoch: int,
    max_tokens: int,
    output_root: Path,
    source_mix: DataMix,
    source_sequence_length: int,
) -> tuple[
    NumpyFSLDataset,
    Any,
    int,
    int,
    np.ndarray,
    str,
    int,
]:
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
    return (
        dataset,
        loader,
        item_size,
        sequence_bytes,
        global_indices,
        global_indices_file,
        instances_needed,
    )


def build_source_df(
    *,
    dataset: NumpyFSLDataset,
    global_indices: np.ndarray,
    source_sequence_length: int,
    item_size: int,
    max_tokens: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build source, label, and planning preview DataFrames for the replay prefix."""
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

    sequence_bytes = source_sequence_length * item_size
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
            requested_gib=lambda df: (df["requested_tokens"] * item_size / (1024**3)),
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
    return (
        access_preview_df,
        source_df,
        label_summary_df,
        planner_df,
        planner_label_df,
        planner_request_df,
        source_ids,
        local_indices,
        touches,
        file_sizes,
        requested_bytes,
    )


def run_planner_simulation(
    *,
    global_indices: np.ndarray,
    source_ids: np.ndarray,
    local_indices: np.ndarray,
    touches: np.ndarray,
    file_sizes: np.ndarray,
    requested_bytes: np.ndarray,
    sequence_bytes: int,
    merge_gap_instances: list[int],
    planner_window_sizes: list[int],
    planner_merge_extra_bytes: list[int],
    planner_stage_budgets_gib: list[int],
    planner_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge-gap sweep, windowed / merge planners, hot-shard staging, conclusion frames, and artifacts table.
    """
    exact_requested_bytes = int(len(global_indices) * sequence_bytes)
    n_global = int(len(global_indices))
    n_global_f = float(n_global) if n_global else 0.0

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
                "exact_requested_gib": gib(exact_requested_bytes),
                "overfetch_gib": gib(merged_bytes - exact_requested_bytes),
            }
        )
    merge_df = pd.DataFrame(merge_rows)

    planner_summary_rows: list[dict[str, int | str]] = [
        {
            "planner_family": "baseline_output_parallel",
            "window_size": 1,
            "merge_extra_bytes": 0,
            "stage_budget_gib": 0,
            "remote_requests": n_global,
            "remote_bytes_fetched_bytes": exact_requested_bytes,
            "peak_reorder_buffer_instances": 0,
            "staged_shards": 0,
            "staged_bytes_bytes": 0,
        }
    ]
    planner_locality_rows: list[dict[str, int | float]] = []

    for window_size in planner_window_sizes:
        repeated_requests = 0
        unique_shards_total = 0
        window_count = 0
        merged_requests_by_budget = {budget: 0 for budget in planner_merge_extra_bytes}
        merged_extra_instances_by_budget = {
            budget: 0 for budget in planner_merge_extra_bytes
        }
        mergeable_reduction_by_budget = {
            budget: 0 for budget in planner_merge_extra_bytes
        }

        for window_start in range(0, n_global, window_size):
            window_end = min(window_start + window_size, n_global)
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
                    mergeable_reduction_by_budget[budget] += group_size - 1 - over_budget_count
                    if over_budget_count != gap_instances.size:
                        merged_extra_instances_by_budget[budget] += int(
                            gap_instances[~over_budget].sum()
                        )

        avg_unique_shards_per_window = (
            unique_shards_total / window_count if window_count > 0 else 0.0
        )
        repeated_request_fraction = (
            repeated_requests / n_global_f if n_global else 0.0
        )

        planner_summary_rows.append(
            {
                "planner_family": "window_regrouping",
                "window_size": window_size,
                "merge_extra_bytes": 0,
                "stage_budget_gib": 0,
                "remote_requests": n_global,
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
            mergeable_request_fraction = (
                mergeable_reduction_by_budget[budget] / n_global_f
                if n_global
                else 0.0
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
                    "mergeable_request_fraction": mergeable_request_fraction,
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

        remote_requests = n_global - staged_touches_total + staged_shards
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
        remote_bytes_fetched_gib=lambda df: (df["remote_bytes_fetched_bytes"] / (1024**3)),
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
        window_label=lambda df: df["window_size"].map(lambda value: f"{int(value):,}"),
    )

    planner_locality_df = pd.DataFrame(planner_locality_rows).assign(
        merge_budget_label=lambda df: df["merge_extra_bytes"].map(
            lambda value: f"{int(value / 1024):,} KiB"
        )
    )

    best_merge_configs_df = (
        planner_summary_df.loc[planner_summary_df["planner_family"] == "window_merge"]
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

    best_merge_configs_df["conclusion_label"] = best_merge_configs_df["window_size"].map(
        lambda value: f"Best merge @ {int(value):,}"
    )
    regroup_configs_df["conclusion_label"] = regroup_configs_df["window_size"].map(
        lambda value: f"Regroup @ {int(value):,}"
    )
    staging_configs_df["conclusion_label"] = staging_configs_df["stage_budget_gib"].map(
        lambda value: f"Stage {int(value):,} GiB"
    )
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

    planner_artifacts_df = build_planner_artifacts_df()
    return (
        merge_df,
        planner_artifacts_df,
        planner_conclusion_df,
        planner_locality_df,
        planner_summary_df,
    )


def build_planner_artifacts_df() -> pd.DataFrame:
    return pd.DataFrame(
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


def build_workflow_df() -> pd.DataFrame:
    return pd.DataFrame(
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


def compute_unique_shards_curve(
    *,
    source_ids: np.ndarray,
    curve_token_steps: list[int],
    source_sequence_length: int,
) -> pd.DataFrame:
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
    return pd.DataFrame(curve_rows)


def build_source_subset(
    *,
    dataset: NumpyFSLDataset,
    source_sequence_length: int,
    subset_example_tokens: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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
    subset_example_entries = [entry.model_copy(deep=True) for entry in source_entries]
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
    return source_subset_df, subset_example_df, subset_example_labels


def build_replay_summary(
    *,
    dataset: NumpyFSLDataset,
    label_summary_df: pd.DataFrame,
    item_size: int,
    sequence_bytes: int,
    file_sizes: np.ndarray,
    global_indices: np.ndarray,
    touches: np.ndarray,
    global_indices_file: str,
    planner_workers: int,
    instances_needed: int,
    subset_example_labels: list[str],
) -> dict[str, int | str | float | list[str]]:
    return {
        "num_paths": len(dataset.paths),
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
