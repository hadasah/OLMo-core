import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo

    from olmo_core.data.mixes import DataMix

    _nbs_dir = Path(__file__).resolve().parent
    if str(_nbs_dir) not in sys.path:
        sys.path.insert(0, str(_nbs_dir))

    from replay_download_pipeline import (
        build_dataset_loader,
        build_replay_summary,
        build_source_df,
        build_source_subset,
        build_workflow_df,
        compute_unique_shards_curve,
        default_olmo_output_root,
        run_planner_simulation,
    )

    alt.data_transformers.disable_max_rows()


@app.cell
def _():
    # Cache/output root for the mix: override with environment variable OLMO_OUTPUT_ROOT
    # (see replay_download_pipeline.default_olmo_output_root).
    output_root = default_olmo_output_root()
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
    (
        dataset,
        _loader,
        item_size,
        sequence_bytes,
        global_indices,
        global_indices_file,
        instances_needed,
    ) = build_dataset_loader(
        data_root=data_root,
        data_seed=data_seed,
        epoch=epoch,
        max_tokens=max_tokens,
        output_root=output_root,
        source_mix=source_mix,
        source_sequence_length=source_sequence_length,
    )

    (
        access_preview_df,
        source_df,
        label_summary_df,
        planner_df,
        planner_label_df,
        _planner_request_df,
        source_ids,
        local_indices,
        touches,
        file_sizes,
        requested_bytes,
    ) = build_source_df(
        dataset=dataset,
        global_indices=global_indices,
        source_sequence_length=source_sequence_length,
        item_size=item_size,
        max_tokens=max_tokens,
    )

    (
        merge_df,
        planner_artifacts_df,
        planner_conclusion_df,
        planner_locality_df,
        planner_summary_df,
    ) = run_planner_simulation(
        global_indices=global_indices,
        source_ids=source_ids,
        local_indices=local_indices,
        touches=touches,
        file_sizes=file_sizes,
        requested_bytes=requested_bytes,
        sequence_bytes=sequence_bytes,
        merge_gap_instances=merge_gap_instances,
        planner_window_sizes=planner_window_sizes,
        planner_merge_extra_bytes=planner_merge_extra_bytes,
        planner_stage_budgets_gib=planner_stage_budgets_gib,
        planner_workers=planner_workers,
    )

    unique_shards_curve_df = compute_unique_shards_curve(
        source_ids=source_ids,
        curve_token_steps=curve_token_steps,
        source_sequence_length=source_sequence_length,
    )
    workflow_df = build_workflow_df()

    source_subset_df, subset_example_df, subset_example_labels = build_source_subset(
        dataset=dataset,
        source_sequence_length=source_sequence_length,
        subset_example_tokens=subset_example_tokens,
    )
    summary = build_replay_summary(
        dataset=dataset,
        label_summary_df=label_summary_df,
        item_size=item_size,
        sequence_bytes=sequence_bytes,
        file_sizes=file_sizes,
        global_indices=global_indices,
        touches=touches,
        global_indices_file=global_indices_file,
        planner_workers=planner_workers,
        instances_needed=instances_needed,
        subset_example_labels=subset_example_labels,
    )
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
