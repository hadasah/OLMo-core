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
        _assign_target_instances,
        _collect_source_entries,
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
    subset_example_tokens = 100_000_000
    return (
        curve_token_steps,
        data_root,
        data_seed,
        epoch,
        max_tokens,
        merge_gap_instances,
        output_root,
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

    source_entries = _collect_source_entries(dataset, group_mapping=None)
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
    _assign_target_instances(
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
        "subset_example_labels": subset_example_labels,
    }
    return (
        access_preview_df,
        label_summary_df,
        merge_df,
        planner_artifacts_df,
        planner_df,
        planner_label_df,
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

    This notebook is meant to make two planning views concrete for the `OLMoE-mix-0824` replay-cache run.

    1. **What the source dataset looks like before any payload reads**
       - how many source labels and shard files are in the mix
       - the size and shape of those shard files
       - how much capacity sits behind each label
    2. **What the replay planner actually uses up front**
       - the resolved mix paths and labels
       - per-path file sizes and dataset offsets
       - the dtype / bytes-per-instance implied by the tokenizer and sequence length
       - the `global_indices` prefix for the requested replay budget
    3. **How that differs from the newer `source_subset_cache` flow**
       - replay cache preserves the seeded training order
       - source subset cache allocates by label/group/path first, then samples deterministically within each path

    The charts below should make it easier to answer: how broad is the touched set, how quickly do we touch most shards, and what information is already available before source-byte downloads begin?
    """)
    return


@app.cell(column=2, hide_code=True)
def _(label_summary_df, max_tokens, source_mix, summary):
    mo.vstack(
        [
            mo.md(f"""
    ## Planner Inputs Available Before Payload Reads

    - `source_mix`: `{source_mix.value}`
    - source paths in mix: `{summary["num_paths"]:,}`
    - unique raw labels: `{summary["num_labels"]:,}`
    - dataset dtype: `{summary["dtype"]}` (`{summary["item_size"]}` bytes/token)
    - bytes per 4096-token instance: `{summary["sequence_bytes"]:,}`
    - total source size across all shard files: `{summary["total_source_gib"]:,.1f} GiB`
    - replay request: `{max_tokens / 1e9:,.1f}B` tokens -> `{summary["instances_needed"]:,}` instances -> `{summary["requested_gib"]:,.1f} GiB` of exact payload
    - touched shard files under this replay prefix: `{summary["touched_paths"]:,}` / `{summary["num_paths"]:,}`
    - cached permutation file: `{summary["global_indices_file"]}`

    The key point is that the planner already has enough information to map replay instances back to concrete shard files and byte ranges *before* it starts downloading source payload bytes.
    """),
            label_summary_df[
                [
                    "label",
                    "num_paths",
                    "total_size_gib",
                    "total_tokens_billions",
                    "touched_paths",
                    "requested_gib",
                    "replay_share",
                ]
            ].sort_values("total_size_gib", ascending=False),
        ]
    )
    return


@app.cell(column=3)
def _():
    return


@app.cell(hide_code=True)
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

    alt.vconcat(label_size_chart, shard_size_chart)
    return


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

    alt.vconcat(planner_label_chart, coverage_chart, unique_shards_chart)
    return


@app.cell(column=4, hide_code=True)
def _(access_preview_df):
    access_preview_df
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

    alt.hconcat(merge_request_chart, overfetch_chart)
    return


@app.cell(hide_code=True)
def _(workflow_df):
    workflow_df
    return


@app.cell(hide_code=True)
def _(source_subset_df):
    source_subset_df.head(25)
    return


@app.cell(column=5, hide_code=True)
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

    alt.vconcat(subset_capacity_chart, subset_example_chart)
    return


@app.cell(column=6)
def _():
    return


@app.cell(hide_code=True)
def _(planner_artifacts_df, subset_example_tokens):
    mo.vstack(
        [
            mo.md(f"""
    ## source_subset_cache Inputs

    The newer source-subset path does **not** start from `global_indices`.
    Instead, it builds `source_entries` at the `raw_label -> path` hierarchy and then uses divisor-style apportionment to assign target instances.

    The toy chart above uses an equal-weight target mixture over the four largest labels in this mix for a `{subset_example_tokens / 1e6:,.0f}M` token subset budget. The important part is the shape of the decision, not the particular toy weights:

    - group / label totals are assigned first
    - each label is then split across its concrete source paths
    - deterministic per-path sampling picks which local instances to materialize
    - append works cleanly because allocations are monotone as `max_tokens` grows
    """),
            planner_artifacts_df,
        ]
    )
    return


@app.cell(column=7, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
