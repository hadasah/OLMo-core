import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MoE Data Repetition Analysis

    Interactive exploration of how **data repetition** affects expert routing behavior
    in OLMo-2 80M Mixture-of-Experts models.

    **Sweep dimensions:** 3 datasets × 2 expert counts × 6 repetition levels = 36 experiments

    Each experiment produces three analyses:
    - **Knockout** — importance of each expert (loss increase when disabled)
    - **Ossification** — how quickly routing decisions lock in during training
    - **Co-activation** — which expert pairs tend to fire together
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import json
    import os
    import re

    return go, json, make_subplots, mo, np, os, pd, re


@app.cell(hide_code=True)
def _(json, np, os, pd, re):
    """Load all experiment data from analysis_results_2/."""

    # __file__ is set by marimo when running the notebook; fall back to cwd
    _this_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    _repo_root = os.path.dirname(_this_dir) if os.path.basename(_this_dir) == "notebooks" else _this_dir
    _BASE_DIR = os.path.join(_repo_root, "analysis_results_2")

    REP_LEVELS = ["1x", "2x", "4x", "8x", "16x", "32x"]
    _REP_INTS = {"1x": 1, "2x": 2, "4x": 4, "8x": 8, "16x": 16, "32x": 32}
    DATASETS = ["pes2o", "starcoder", "wiki_v2"]
    MOE_CONFIGS = ["moe32", "moe64"]
    N_LAYERS = 8

    # Storage
    _knockout_rows = []
    _ossification_rows = []
    _coactivation_rows = []
    coactivation_matrices = {}  # nested dict for on-demand access

    # Walk experiment directories
    for _parent_name in sorted(os.listdir(_BASE_DIR)):
        _parent_path = os.path.join(_BASE_DIR, _parent_name)
        if not os.path.isdir(_parent_path) or _parent_name == "logs":
            continue

        # Extract dataset from parent directory name
        _dataset_match = re.search(r"data_rep_AC_(\w+?)_olmo2", _parent_name)
        if not _dataset_match:
            continue
        _dataset = _dataset_match.group(1)

        for _exp_name in sorted(os.listdir(_parent_path)):
            _exp_path = os.path.join(_parent_path, _exp_name)
            if not os.path.isdir(_exp_path):
                continue

            # Extract moe_config and rep_level from experiment directory name
            _config_match = re.search(r"(moe\d+)_rep(\d+x)", _exp_name)
            if not _config_match:
                continue
            _moe_config = _config_match.group(1)
            _rep_level = _config_match.group(2)
            _rep_int = _REP_INTS[_rep_level]

            # --- Knockout ---
            _knockout_path = os.path.join(_exp_path, "knockout.json")
            if os.path.exists(_knockout_path):
                with open(_knockout_path) as _f:
                    _ko = json.load(_f)
                _baseline_loss = _ko["baseline_loss"]
                for _layer_key, _experts in _ko["knockout_losses"].items():
                    _layer = int(re.search(r"layer(\d+)", _layer_key).group(1))
                    for _expert_idx_str, _vals in _experts.items():
                        _knockout_rows.append({
                            "dataset": _dataset,
                            "moe_config": _moe_config,
                            "rep_level": _rep_level,
                            "rep_int": _rep_int,
                            "layer": _layer,
                            "expert_idx": int(_expert_idx_str),
                            "loss_increase": _vals["loss_increase"],
                            "baseline_loss": _baseline_loss,
                        })

            # --- Ossification ---
            _oss_path = os.path.join(_exp_path, "ossification.json")
            if os.path.exists(_oss_path):
                with open(_oss_path) as _f:
                    _oss = json.load(_f)
                _steps = _oss["steps"]
                for _layer_key, _rates in _oss["stability_rates"].items():
                    _layer = int(re.search(r"layer(\d+)", _layer_key).group(1))
                    for _step, _rate in zip(_steps, _rates):
                        _ossification_rows.append({
                            "dataset": _dataset,
                            "moe_config": _moe_config,
                            "rep_level": _rep_level,
                            "rep_int": _rep_int,
                            "layer": _layer,
                            "step": _step,
                            "stability_rate": _rate,
                        })

            # --- Co-activation ---
            _ca_path = os.path.join(_exp_path, "co_activation.json")
            if os.path.exists(_ca_path):
                with open(_ca_path) as _f:
                    _ca = json.load(_f)
                for _layer_key, _layer_data in _ca.items():
                    _layer = int(re.search(r"layer(\d+)", _layer_key).group(1))
                    _coactivation_rows.append({
                        "dataset": _dataset,
                        "moe_config": _moe_config,
                        "rep_level": _rep_level,
                        "rep_int": _rep_int,
                        "layer": _layer,
                        "co_activation_entropy": _layer_data["co_activation_entropy"],
                        "top_pairs": _layer_data["top_pairs"],
                    })
                    _mat_key = (_dataset, _moe_config, _rep_level, _layer)
                    coactivation_matrices[_mat_key] = np.array(
                        _layer_data["co_activation_matrix"]
                    )

    df_knockout = pd.DataFrame(_knockout_rows)
    df_ossification = pd.DataFrame(_ossification_rows)
    df_coactivation = pd.DataFrame(_coactivation_rows)

    # Sort rep_level categorically
    _rep_cat = pd.CategoricalDtype(categories=REP_LEVELS, ordered=True)
    for _df in [df_knockout, df_ossification, df_coactivation]:
        _df["rep_level"] = _df["rep_level"].astype(_rep_cat)

    print(f"Loaded: {len(df_knockout)} knockout rows, {len(df_ossification)} ossification rows, "
          f"{len(df_coactivation)} co-activation rows, {len(coactivation_matrices)} matrices")
    return (
        DATASETS,
        MOE_CONFIGS,
        N_LAYERS,
        REP_LEVELS,
        coactivation_matrices,
        df_coactivation,
        df_knockout,
        df_ossification,
    )


@app.cell(hide_code=True)
def _(DATASETS, mo):
    """Global controls for filtering across all sections."""
    dataset_selector = mo.ui.multiselect(
        options=DATASETS,
        value=DATASETS,
        label="Datasets",
    )
    moe_selector = mo.ui.dropdown(
        options={"Both": "both", "moe32": "moe32", "moe64": "moe64"},
        value="Both",
        label="Expert count",
    )
    mo.hstack([dataset_selector, moe_selector], justify="start", gap=1)
    return dataset_selector, moe_selector


@app.cell(hide_code=True)
def _(MOE_CONFIGS, dataset_selector, moe_selector):
    """Derive active filters from global controls."""
    active_datasets = list(dataset_selector.value) if dataset_selector.value else ["pes2o", "starcoder", "wiki_v2"]
    active_moe = MOE_CONFIGS if moe_selector.value == "both" else [moe_selector.value]
    return active_datasets, active_moe


@app.cell
def _(mo):
    mo.md("""
    ## Section A: Overview
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Key Metrics vs. Repetition** — High-level summary of how all three analysis
    dimensions respond to increasing data repetition. Each subplot shows one metric
    averaged across all layers and experts for each experiment configuration:

    - **Mean knockout impact**: the average loss increase when a single expert is disabled.
      Higher values mean individual experts are more critical to model performance.
    - **Mean co-activation entropy**: Shannon entropy of the pairwise expert co-activation
      distribution. Lower entropy indicates more rigid, stereotyped expert pairing patterns.
    - **End-of-training routing stability**: fraction of tokens whose top-expert assignment
      at the final checkpoint matches their assignment at the second-to-last checkpoint.
      Values near 1.0 mean routing decisions have fully locked in.
    """)
    return


@app.cell(hide_code=True)
def _(
    active_datasets,
    active_moe,
    df_coactivation,
    df_knockout,
    df_ossification,
    go,
    make_subplots,
):
    """Plot 1 — Key Metrics vs. Repetition (3 stacked subplots)."""

    DATASET_COLORS = {"pes2o": "#1f77b4", "starcoder": "#ff7f0e", "wiki_v2": "#2ca02c"}
    MOE_DASHES = {"moe32": "solid", "moe64": "dash"}

    _fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Mean Knockout Impact (loss increase)",
            "Mean Co-Activation Entropy",
            "End-of-Training Routing Stability",
        ],
    )

    for _ds in active_datasets:
        for _mc in active_moe:
            _label = f"{_ds} / {_mc}"
            _color = DATASET_COLORS[_ds]
            _dash = MOE_DASHES[_mc]

            # Knockout: mean loss_increase per rep_level
            _ko_sub = df_knockout[
                (df_knockout["dataset"] == _ds) & (df_knockout["moe_config"] == _mc)
            ]
            _ko_agg = _ko_sub.groupby("rep_level", observed=True)["loss_increase"].mean()

            _fig.add_trace(
                go.Scatter(
                    x=[str(_r) for _r in _ko_agg.index],
                    y=_ko_agg.values,
                    mode="lines+markers",
                    name=_label,
                    line=dict(color=_color, dash=_dash),
                    legendgroup=_label,
                    showlegend=True,
                ),
                row=1, col=1,
            )

            # Co-activation entropy: mean per rep_level
            _ca_sub = df_coactivation[
                (df_coactivation["dataset"] == _ds) & (df_coactivation["moe_config"] == _mc)
            ]
            _ca_agg = _ca_sub.groupby("rep_level", observed=True)["co_activation_entropy"].mean()

            _fig.add_trace(
                go.Scatter(
                    x=[str(_r) for _r in _ca_agg.index],
                    y=_ca_agg.values,
                    mode="lines+markers",
                    name=_label,
                    line=dict(color=_color, dash=_dash),
                    legendgroup=_label,
                    showlegend=False,
                ),
                row=2, col=1,
            )

            # Ossification: final stability (max step) averaged across layers
            _oss_sub = df_ossification[
                (df_ossification["dataset"] == _ds) & (df_ossification["moe_config"] == _mc)
            ]
            _oss_final = _oss_sub[_oss_sub["step"] == _oss_sub["step"].max()]
            _oss_agg = _oss_final.groupby("rep_level", observed=True)["stability_rate"].mean()

            _fig.add_trace(
                go.Scatter(
                    x=[str(_r) for _r in _oss_agg.index],
                    y=_oss_agg.values,
                    mode="lines+markers",
                    name=_label,
                    line=dict(color=_color, dash=_dash),
                    legendgroup=_label,
                    showlegend=False,
                ),
                row=3, col=1,
            )

    _fig.update_layout(
        height=800,
        title_text="Key Metrics vs. Data Repetition",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    _fig.update_xaxes(title_text="Repetition level", row=3, col=1)
    _fig.update_yaxes(title_text="Mean Δloss", row=1, col=1)
    _fig.update_yaxes(title_text="Entropy (nats)", row=2, col=1)
    _fig.update_yaxes(title_text="Stability rate", row=3, col=1)
    _fig
    return DATASET_COLORS, MOE_DASHES


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Baseline Loss vs. Repetition** — Model quality (evaluation loss with all experts active)
    at each repetition level. This contextualizes the knockout results above: if baseline loss
    itself changes substantially with repetition, knockout Δloss values should be interpreted
    relative to the baseline rather than in absolute terms.
    """)
    return


@app.cell(hide_code=True)
def _(
    DATASET_COLORS,
    MOE_DASHES,
    active_datasets,
    active_moe,
    df_knockout,
    go,
):
    """Plot 2 — Baseline Loss vs. Repetition."""

    _fig = go.Figure()

    for _ds in active_datasets:
        for _mc in active_moe:
            _sub = df_knockout[
                (df_knockout["dataset"] == _ds) & (df_knockout["moe_config"] == _mc)
            ]
            _agg = _sub.groupby("rep_level", observed=True)["baseline_loss"].first()
            _fig.add_trace(
                go.Scatter(
                    x=[str(_r) for _r in _agg.index],
                    y=_agg.values,
                    mode="lines+markers",
                    name=f"{_ds} / {_mc}",
                    line=dict(color=DATASET_COLORS[_ds], dash=MOE_DASHES[_mc]),
                )
            )

    _fig.update_layout(
        title="Baseline Loss vs. Data Repetition",
        xaxis_title="Repetition level",
        yaxis_title="Baseline loss",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section B: Expert Knockout Analysis

    **Knockout analysis** measures each expert's importance by disabling it (zeroing its
    contribution) and measuring the resulting increase in evaluation loss. An expert with
    a large **knockout Δloss** is critical — the model relies heavily on it. An expert
    whose removal barely changes loss (or even improves it) is redundant.
    """)
    return


@app.cell(hide_code=True)
def _(N_LAYERS, mo):
    """Local controls for knockout section."""
    ko_layer_slider = mo.ui.slider(
        start=0, stop=N_LAYERS - 1, step=1, value=0, label="Layer", show_value=True,
    )
    ko_topk_slider = mo.ui.slider(
        start=5, stop=20, step=1, value=10, label="Top-K experts (bump chart)", show_value=True,
    )
    mo.hstack([ko_layer_slider, ko_topk_slider], justify="start", gap=1)
    return ko_layer_slider, ko_topk_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Per-Layer Knockout Heatmap** — Mean knockout Δloss for each (layer, repetition level)
    combination, averaged across all experts in that layer. Brighter cells indicate layers
    where individual experts are more critical on average. Look for layers or repetition
    levels where importance concentrates.
    """)
    return


@app.cell(hide_code=True)
def _(REP_LEVELS, active_datasets, active_moe, df_knockout, go, make_subplots):
    """Plot 3 — Per-Layer Knockout Heatmap (Layer × Rep)."""

    _n_moe = len(active_moe)
    _n_ds = len(active_datasets)

    if _n_moe == 0 or _n_ds == 0:
        _fig = go.Figure()
        _fig.update_layout(title="No data selected")
    else:
        _fig = make_subplots(
            rows=_n_moe, cols=_n_ds,
            subplot_titles=[
                f"{_ds} / {_mc}" for _mc in active_moe for _ds in active_datasets
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.08,
        )

        for _r_idx, _mc in enumerate(active_moe):
            for _c_idx, _ds in enumerate(active_datasets):
                _sub = df_knockout[
                    (df_knockout["dataset"] == _ds) & (df_knockout["moe_config"] == _mc)
                ]
                _pivot = _sub.groupby(
                    ["rep_level", "layer"], observed=True
                )["loss_increase"].mean().reset_index()
                _heat = _pivot.pivot(index="rep_level", columns="layer", values="loss_increase")

                # Ensure proper ordering
                _heat = _heat.reindex(index=REP_LEVELS)

                _fig.add_trace(
                    go.Heatmap(
                        z=_heat.values,
                        x=[f"L{_c}" for _c in _heat.columns],
                        y=[str(_r) for _r in _heat.index],
                        colorscale="YlOrRd",
                        showscale=(_r_idx == 0 and _c_idx == _n_ds - 1),
                        text=_heat.values.round(4),
                        texttemplate="%{text:.4f}",
                        textfont={"size": 9},
                        hovertemplate="Layer %{x}<br>Rep %{y}<br>Mean Δloss: %{z:.5f}<extra></extra>",
                    ),
                    row=_r_idx + 1, col=_c_idx + 1,
                )

        _fig.update_layout(
            height=300 * _n_moe + 100,
            title_text="Per-Layer Knockout Impact — Mean Δloss (layer × rep)",
        )
        for _i in range(1, _n_moe * _n_ds + 1):
            _fig.update_xaxes(title_text="Layer", row=(_i - 1) // _n_ds + 1, col=(_i - 1) % _n_ds + 1)
        for _r_idx in range(_n_moe):
            _fig.update_yaxes(title_text="Rep level", row=_r_idx + 1, col=1)

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Expert Importance Distribution** — Box plots showing the full distribution of knockout
    Δloss values across all experts in the selected layer. Long upper whiskers or outliers
    indicate "superstar" experts that are disproportionately important. A tight distribution
    means experts share the workload more evenly.
    """)
    return


@app.cell(hide_code=True)
def _(
    active_datasets,
    active_moe,
    df_knockout,
    go,
    ko_layer_slider,
    make_subplots,
):
    """Plot 4 — Expert Importance Distribution (box plots)."""

    _selected_layer = ko_layer_slider.value
    _n_ds = len(active_datasets)

    if _n_ds == 0:
        _fig = go.Figure()
        _fig.update_layout(title="No data selected")
    else:
        _fig = make_subplots(
            rows=1, cols=_n_ds,
            subplot_titles=[f"{_ds}" for _ds in active_datasets],
            horizontal_spacing=0.06,
        )

        for _c_idx, _ds in enumerate(active_datasets):
            for _mc in active_moe:
                _sub = df_knockout[
                    (df_knockout["dataset"] == _ds)
                    & (df_knockout["moe_config"] == _mc)
                    & (df_knockout["layer"] == _selected_layer)
                ]
                _fig.add_trace(
                    go.Box(
                        x=_sub["rep_level"].astype(str),
                        y=_sub["loss_increase"],
                        name=_mc,
                        legendgroup=_mc,
                        showlegend=(_c_idx == 0),
                        marker_color={"moe32": "#636EFA", "moe64": "#EF553B"}.get(_mc, "#000"),
                    ),
                    row=1, col=_c_idx + 1,
                )

        _fig.update_layout(
            height=450,
            title_text=f"Expert Importance Distribution — Layer {_selected_layer}",
            boxmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        for _i in range(1, _n_ds + 1):
            _fig.update_xaxes(title_text="Rep level", row=1, col=_i)
        _fig.update_yaxes(title_text="Knockout Δloss", row=1, col=1)

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Expert Importance Ranking (Bump Chart)** — Tracks how the top-K most important
    experts (ranked by knockout Δloss at 1× repetition) change rank as repetition increases.
    Parallel lines mean the same experts stay dominant regardless of repetition.
    Crossing lines indicate that repetition reshuffles which experts the model relies on.
    Uses the first active dataset and expert count.
    """)
    return


@app.cell(hide_code=True)
def _(
    active_datasets,
    active_moe,
    df_knockout,
    go,
    ko_layer_slider,
    ko_topk_slider,
):
    """Plot 5 — Expert Importance Ranking (Bump Chart)."""

    _layer = ko_layer_slider.value
    _topk = ko_topk_slider.value

    # Use first active dataset/moe_config for the bump chart
    _ds = active_datasets[0] if active_datasets else "starcoder"
    _mc = active_moe[0] if active_moe else "moe64"

    _sub = df_knockout[
        (df_knockout["dataset"] == _ds)
        & (df_knockout["moe_config"] == _mc)
        & (df_knockout["layer"] == _layer)
    ].copy()

    # Rank experts within each rep_level (higher loss_increase = rank 1)
    _sub["rank"] = _sub.groupby("rep_level", observed=True)["loss_increase"].rank(
        ascending=False, method="min"
    )

    # Find top-K experts at the baseline (rep1x)
    _baseline = _sub[_sub["rep_level"] == "1x"].nsmallest(_topk, "rank")
    _top_experts = _baseline["expert_idx"].tolist()

    _fig = go.Figure()
    for _expert_id in _top_experts:
        _expert_data = _sub[_sub["expert_idx"] == _expert_id].sort_values("rep_int")
        _fig.add_trace(
            go.Scatter(
                x=[str(_r) for _r in _expert_data["rep_level"]],
                y=_expert_data["rank"],
                mode="lines+markers",
                name=f"Expert {_expert_id}",
                hovertemplate=(
                    f"Expert {_expert_id}<br>"
                    "Rep: %{x}<br>"
                    "Rank: %{y}<br>"
                    "<extra></extra>"
                ),
            )
        )

    _fig.update_layout(
        title=f"Expert Importance Ranking — {_ds} / {_mc} / Layer {_layer} (Top {_topk} at 1x)",
        xaxis_title="Repetition level",
        yaxis_title="Rank (1 = most important)",
        yaxis=dict(autorange="reversed"),
        height=500,
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Expert Importance Inequality (Gini Coefficient)** — The
    [Gini coefficient](https://en.wikipedia.org/wiki/Gini_coefficient) measures how
    unevenly knockout impact is distributed across experts. A Gini of 0 means all
    experts are equally important; a Gini approaching 1 means a small number of experts
    carry almost all the importance. Rising Gini with repetition would indicate that
    repeated data causes the model to concentrate its reliance on fewer experts.
    """)
    return


@app.cell(hide_code=True)
def _(
    DATASET_COLORS,
    MOE_DASHES,
    active_datasets,
    active_moe,
    df_knockout,
    go,
    np,
):
    """Plot 6 — Expert Importance Inequality (Gini Coefficient)."""

    def _gini(values):
        """Compute Gini coefficient of a 1-D array."""
        _v = np.sort(np.abs(values))
        _n = len(_v)
        if _n == 0 or _v.sum() == 0:
            return 0.0
        _index = np.arange(1, _n + 1)
        return (2 * np.sum(_index * _v) - (_n + 1) * np.sum(_v)) / (_n * np.sum(_v))

    _fig = go.Figure()

    for _ds in active_datasets:
        for _mc in active_moe:
            _sub = df_knockout[
                (df_knockout["dataset"] == _ds) & (df_knockout["moe_config"] == _mc)
            ]
            _gini_per_rep = (
                _sub.groupby("rep_level", observed=True)["loss_increase"]
                .apply(_gini)
                .reset_index()
            )
            _gini_per_rep.columns = ["rep_level", "gini"]

            _fig.add_trace(
                go.Scatter(
                    x=_gini_per_rep["rep_level"].astype(str),
                    y=_gini_per_rep["gini"],
                    mode="lines+markers",
                    name=f"{_ds} / {_mc}",
                    line=dict(color=DATASET_COLORS[_ds], dash=MOE_DASHES[_mc]),
                )
            )

    _fig.update_layout(
        title="Expert Importance Inequality (Gini Coefficient) vs. Repetition",
        xaxis_title="Repetition level",
        yaxis_title="Gini coefficient",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section C: Routing Ossification

    **Routing ossification** measures how quickly the router's expert assignments become
    fixed during training. At each checkpoint, we compare routing decisions to those at the
    final checkpoint. The **stability rate** is the fraction of tokens that are routed to
    the same expert as at the end of training. A stability rate of 1.0 means the routing
    pattern has fully converged (ossified). If high-repetition models reach high stability
    earlier, it suggests that repeated data causes premature lock-in of routing decisions.
    """)
    return


@app.cell(hide_code=True)
def _(N_LAYERS, mo):
    """Local controls for ossification section."""
    oss_threshold_slider = mo.ui.slider(
        start=0.80, stop=0.99, step=0.01, value=0.90,
        label="Stability threshold", show_value=True,
    )
    oss_layer_slider = mo.ui.slider(
        start=0, stop=N_LAYERS - 1, step=1, value=0,
        label="Layer (for per-layer plot)", show_value=True,
    )
    mo.hstack([oss_threshold_slider, oss_layer_slider], justify="start", gap=1)
    return oss_layer_slider, oss_threshold_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Ossification Trajectories** — Mean stability rate (averaged across all 8 layers)
    over the course of training, with one curve per repetition level. Lines that rise
    faster or plateau higher indicate earlier or stronger ossification. Darker colors
    correspond to higher repetition levels.
    """)
    return


@app.cell(hide_code=True)
def _(active_datasets, active_moe, df_ossification, go, make_subplots):
    """Plot 7 — Ossification Trajectories (faceted by dataset × moe)."""

    REP_COLORSCALE = {
        "1x": "#ffffcc", "2x": "#c7e9b4", "4x": "#7fcdbb",
        "8x": "#41b6c4", "16x": "#2c7fb8", "32x": "#253494",
    }

    _n_moe = len(active_moe)
    _n_ds = len(active_datasets)

    if _n_moe == 0 or _n_ds == 0:
        _fig = go.Figure()
        _fig.update_layout(title="No data selected")
    else:
        _fig = make_subplots(
            rows=_n_moe, cols=_n_ds,
            subplot_titles=[f"{_ds} / {_mc}" for _mc in active_moe for _ds in active_datasets],
            vertical_spacing=0.12,
            horizontal_spacing=0.06,
        )

        for _r_idx, _mc in enumerate(active_moe):
            for _c_idx, _ds in enumerate(active_datasets):
                _sub = df_ossification[
                    (df_ossification["dataset"] == _ds) & (df_ossification["moe_config"] == _mc)
                ]
                # Average across layers
                _mean_per_step = _sub.groupby(
                    ["rep_level", "step"], observed=True
                )["stability_rate"].mean().reset_index()

                for _rep in _mean_per_step["rep_level"].cat.categories:
                    _rep_data = _mean_per_step[_mean_per_step["rep_level"] == _rep]
                    if len(_rep_data) == 0:
                        continue
                    _fig.add_trace(
                        go.Scatter(
                            x=_rep_data["step"],
                            y=_rep_data["stability_rate"],
                            mode="lines+markers",
                            name=str(_rep),
                            line=dict(color=REP_COLORSCALE.get(str(_rep), "#888")),
                            legendgroup=str(_rep),
                            showlegend=(_r_idx == 0 and _c_idx == 0),
                        ),
                        row=_r_idx + 1, col=_c_idx + 1,
                    )

        _fig.update_layout(
            height=350 * _n_moe + 100,
            title_text="Ossification Trajectories — Mean Stability Over Training",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                title_text="Rep level",
            ),
        )
        for _c in range(1, _n_ds + 1):
            _fig.update_xaxes(title_text="Training step", row=_n_moe, col=_c)
        for _r in range(1, _n_moe + 1):
            _fig.update_yaxes(title_text="Stability rate", row=_r, col=1)

    _fig
    return (REP_COLORSCALE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Per-Layer Ossification** — Same as the trajectories above but for a single layer
    (selected with the slider), rather than averaged across layers. This reveals whether
    ossification speed varies by layer depth — earlier layers typically stabilize faster
    than later layers.
    """)
    return


@app.cell(hide_code=True)
def _(
    REP_COLORSCALE,
    active_datasets,
    active_moe,
    df_ossification,
    go,
    make_subplots,
    oss_layer_slider,
):
    """Plot 8 — Per-Layer Ossification (single selected layer)."""

    _layer = oss_layer_slider.value
    _n_moe = len(active_moe)
    _n_ds = len(active_datasets)

    if _n_moe == 0 or _n_ds == 0:
        _fig = go.Figure()
        _fig.update_layout(title="No data selected")
    else:
        _fig = make_subplots(
            rows=_n_moe, cols=_n_ds,
            subplot_titles=[f"{_ds} / {_mc}" for _mc in active_moe for _ds in active_datasets],
            vertical_spacing=0.15,
            horizontal_spacing=0.06,
        )

        for _r_idx, _mc in enumerate(active_moe):
            for _c_idx, _ds in enumerate(active_datasets):
                _sub = df_ossification[
                    (df_ossification["dataset"] == _ds)
                    & (df_ossification["moe_config"] == _mc)
                    & (df_ossification["layer"] == _layer)
                ]

                for _rep in _sub["rep_level"].cat.categories:
                    _rep_data = _sub[_sub["rep_level"] == _rep].sort_values("step")
                    if len(_rep_data) == 0:
                        continue
                    _fig.add_trace(
                        go.Scatter(
                            x=_rep_data["step"],
                            y=_rep_data["stability_rate"],
                            mode="lines+markers",
                            name=str(_rep),
                            line=dict(color=REP_COLORSCALE.get(str(_rep), "#888")),
                            legendgroup=str(_rep),
                            showlegend=(_r_idx == 0 and _c_idx == 0),
                        ),
                        row=_r_idx + 1, col=_c_idx + 1,
                    )

        _fig.update_layout(
            height=350 * _n_moe + 100,
            title_text=f"Ossification Trajectories — Layer {_layer}",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                title_text="Rep level",
            ),
        )
        for _c in range(1, _n_ds + 1):
            _fig.update_xaxes(title_text="Training step", row=_n_moe, col=_c)
        for _r in range(1, _n_moe + 1):
            _fig.update_yaxes(title_text="Stability rate", row=_r, col=1)

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Steps to Reach Stability Threshold** — For each experiment, the training step at
    which mean stability (averaged across layers) first exceeds the threshold set by the
    slider above. Computed via linear interpolation between measured checkpoints. Lower
    bars mean the model's routing decisions locked in earlier during training. Hatched
    bars indicate moe64 configurations.
    """)
    return


@app.cell(hide_code=True)
def _(
    DATASET_COLORS,
    active_datasets,
    active_moe,
    df_ossification,
    go,
    oss_threshold_slider,
    pd,
):
    """Plot 9 — Steps to Reach Stability Threshold."""

    _threshold = oss_threshold_slider.value

    def _steps_to_threshold(_group, _thresh):
        """Linearly interpolate the step at which stability first exceeds thresh."""
        _group = _group.sort_values("step")
        _steps_arr = _group["step"].values
        _rates_arr = _group["stability_rate"].values
        for _i in range(len(_rates_arr)):
            if _rates_arr[_i] >= _thresh:
                if _i == 0:
                    return float(_steps_arr[0])
                _r0, _r1 = _rates_arr[_i - 1], _rates_arr[_i]
                _s0, _s1 = _steps_arr[_i - 1], _steps_arr[_i]
                _frac = (_thresh - _r0) / (_r1 - _r0) if _r1 != _r0 else 0.0
                return _s0 + _frac * (_s1 - _s0)
        return float("nan")

    _threshold_rows = []
    for _ds in active_datasets:
        for _mc in active_moe:
            _sub = df_ossification[
                (df_ossification["dataset"] == _ds) & (df_ossification["moe_config"] == _mc)
            ]
            _mean_by_step = _sub.groupby(
                ["rep_level", "step"], observed=True
            )["stability_rate"].mean().reset_index()

            for _rep in _mean_by_step["rep_level"].cat.categories:
                _rep_data = _mean_by_step[_mean_by_step["rep_level"] == _rep]
                if len(_rep_data) == 0:
                    continue
                _s = _steps_to_threshold(_rep_data, _threshold)
                _threshold_rows.append({
                    "dataset": _ds, "moe_config": _mc,
                    "rep_level": str(_rep), "steps_to_thresh": _s,
                })

    _fig = go.Figure()

    if _threshold_rows:
        _df_thresh = pd.DataFrame(_threshold_rows)
        for _ds in active_datasets:
            for _mc in active_moe:
                _sub = _df_thresh[
                    (_df_thresh["dataset"] == _ds) & (_df_thresh["moe_config"] == _mc)
                ]
                _fig.add_trace(
                    go.Bar(
                        x=_sub["rep_level"],
                        y=_sub["steps_to_thresh"],
                        name=f"{_ds} / {_mc}",
                        marker_color=DATASET_COLORS[_ds],
                        marker_pattern_shape="/" if _mc == "moe64" else "",
                    )
                )

    _fig.update_layout(
        title=f"Steps to Reach {_threshold:.0%} Mean Stability",
        xaxis_title="Repetition level",
        yaxis_title="Training step",
        barmode="group",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section D: Co-Activation Patterns

    **Co-activation analysis** measures which pairs of experts tend to be selected together
    for the same input tokens (when top-k > 1 routing is used). The **co-activation matrix**
    is an N×N matrix where entry (i, j) is the fraction of tokens for which both expert i
    and expert j were activated. **Co-activation entropy** is the Shannon entropy of this
    distribution — higher entropy means expert pairings are more uniform (any pair is
    roughly equally likely), while lower entropy means the model prefers specific expert
    cliques.
    """)
    return


@app.cell(hide_code=True)
def _(DATASETS, MOE_CONFIGS, N_LAYERS, REP_LEVELS, mo):
    """Local controls for co-activation section."""
    ca_dataset_dd = mo.ui.dropdown(
        options=DATASETS, value=DATASETS[0], label="Dataset",
    )
    ca_moe_dd = mo.ui.dropdown(
        options=MOE_CONFIGS, value=MOE_CONFIGS[1], label="Expert count",
    )
    ca_rep_dd = mo.ui.dropdown(
        options=REP_LEVELS, value="1x", label="Rep level (matrix)",
    )
    ca_layer_slider = mo.ui.slider(
        start=0, stop=N_LAYERS - 1, step=1, value=0,
        label="Layer", show_value=True,
    )
    mo.hstack([ca_dataset_dd, ca_moe_dd, ca_rep_dd, ca_layer_slider], justify="start", gap=1)
    return ca_dataset_dd, ca_layer_slider, ca_moe_dd, ca_rep_dd


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Co-Activation Entropy vs. Repetition** — Per-layer co-activation entropy at each
    repetition level. Each line represents one layer (colored by depth). Decreasing entropy
    with repetition would indicate that repeated data causes experts to form more rigid,
    fixed pairing patterns.
    """)
    return


@app.cell(hide_code=True)
def _(active_datasets, active_moe, df_coactivation, go, make_subplots):
    """Plot 10 — Co-Activation Entropy vs. Rep Level (per layer)."""

    _LAYER_COLORS = [
        "#440154", "#46327e", "#365c8d", "#277f8e",
        "#1fa187", "#4ac16d", "#9fda3a", "#fde725",
    ]

    _n_moe = len(active_moe)
    _n_ds = len(active_datasets)

    if _n_moe == 0 or _n_ds == 0:
        _fig = go.Figure()
        _fig.update_layout(title="No data selected")
    else:
        _fig = make_subplots(
            rows=_n_moe, cols=_n_ds,
            subplot_titles=[f"{_ds} / {_mc}" for _mc in active_moe for _ds in active_datasets],
            vertical_spacing=0.12,
            horizontal_spacing=0.06,
        )

        for _r_idx, _mc in enumerate(active_moe):
            for _c_idx, _ds in enumerate(active_datasets):
                _sub = df_coactivation[
                    (df_coactivation["dataset"] == _ds)
                    & (df_coactivation["moe_config"] == _mc)
                ]

                for _layer_idx in range(8):
                    _lsub = _sub[_sub["layer"] == _layer_idx].sort_values("rep_int")
                    if len(_lsub) == 0:
                        continue
                    _fig.add_trace(
                        go.Scatter(
                            x=_lsub["rep_level"].astype(str),
                            y=_lsub["co_activation_entropy"],
                            mode="lines+markers",
                            name=f"L{_layer_idx}",
                            line=dict(color=_LAYER_COLORS[_layer_idx]),
                            legendgroup=f"L{_layer_idx}",
                            showlegend=(_r_idx == 0 and _c_idx == 0),
                        ),
                        row=_r_idx + 1, col=_c_idx + 1,
                    )

        _fig.update_layout(
            height=350 * _n_moe + 100,
            title_text="Per-Layer Co-Activation Entropy vs. Repetition",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                title_text="Layer",
            ),
        )
        for _c in range(1, _n_ds + 1):
            _fig.update_xaxes(title_text="Rep level", row=_n_moe, col=_c)
        for _r in range(1, _n_moe + 1):
            _fig.update_yaxes(title_text="Entropy (nats)", row=_r, col=1)

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Co-Activation Matrix** — Full pairwise co-activation frequency matrix for the
    selected configuration. Each cell (i, j) shows how often experts i and j are activated
    together. Block-diagonal patterns indicate expert clusters that specialize together.
    Use the dropdowns above to compare different datasets, expert counts, repetition
    levels, and layers.
    """)
    return


@app.cell(hide_code=True)
def _(
    ca_dataset_dd,
    ca_layer_slider,
    ca_moe_dd,
    ca_rep_dd,
    coactivation_matrices,
    go,
):
    """Plot 11 — Co-Activation Matrix Heatmap."""

    _key = (ca_dataset_dd.value, ca_moe_dd.value, ca_rep_dd.value, ca_layer_slider.value)

    if _key in coactivation_matrices:
        _mat = coactivation_matrices[_key]
        _n_experts = _mat.shape[0]
        _fig = go.Figure(
            data=go.Heatmap(
                z=_mat,
                x=list(range(_n_experts)),
                y=list(range(_n_experts)),
                colorscale="Viridis",
                hovertemplate="Expert %{x} × Expert %{y}<br>Co-activation: %{z:.6f}<extra></extra>",
            )
        )
        _fig.update_layout(
            title=(
                f"Co-Activation Matrix — {ca_dataset_dd.value} / {ca_moe_dd.value} / "
                f"{ca_rep_dd.value} / Layer {ca_layer_slider.value}"
            ),
            xaxis_title="Expert",
            yaxis_title="Expert",
            height=600,
            width=700,
            yaxis=dict(autorange="reversed"),
        )
    else:
        _fig = go.Figure()
        _fig.update_layout(title="No matrix available for this selection")

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Co-Activation Matrix Comparison** — Side-by-side view of the co-activation matrix
    at the 1× baseline (left) versus the selected repetition level (right), for the same
    dataset, expert count, and layer. Visual differences reveal how repetition restructures
    expert pairing patterns.
    """)
    return


@app.cell(hide_code=True)
def _(
    ca_dataset_dd,
    ca_layer_slider,
    ca_moe_dd,
    ca_rep_dd,
    coactivation_matrices,
    go,
    make_subplots,
):
    """Plot 12 — Co-Activation Matrix Comparison (rep1x vs. selected)."""

    _ds = ca_dataset_dd.value
    _mc = ca_moe_dd.value
    _layer = ca_layer_slider.value
    _rep = ca_rep_dd.value

    _key_base = (_ds, _mc, "1x", _layer)
    _key_comp = (_ds, _mc, _rep, _layer)

    if _key_base in coactivation_matrices and _key_comp in coactivation_matrices:
        _mat_base = coactivation_matrices[_key_base]
        _mat_comp = coactivation_matrices[_key_comp]

        _fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f"Baseline (1x)", f"Rep {_rep}"],
            horizontal_spacing=0.1,
        )

        for _col_i, (_mat, _label) in enumerate([(_mat_base, "1x"), (_mat_comp, _rep)]):
            _fig.add_trace(
                go.Heatmap(
                    z=_mat,
                    colorscale="Viridis",
                    showscale=(_col_i == 1),
                    hovertemplate="Expert %{x} × Expert %{y}<br>Freq: %{z:.6f}<extra></extra>",
                ),
                row=1, col=_col_i + 1,
            )
            _fig.update_yaxes(autorange="reversed", row=1, col=_col_i + 1)

        _fig.update_layout(
            title=f"Co-Activation Comparison — {_ds} / {_mc} / Layer {_layer}",
            height=500,
            width=1000,
        )
    else:
        _fig = go.Figure()
        _fig.update_layout(title="Matrix not available for this selection")

    _fig
    return


@app.cell(hide_code=True)
def _(
    ca_dataset_dd,
    ca_layer_slider,
    ca_moe_dd,
    ca_rep_dd,
    df_coactivation,
    mo,
    pd,
):
    """Plot 13 — Top Co-Activated Pairs Table."""

    _ds = ca_dataset_dd.value
    _mc = ca_moe_dd.value
    _rep = ca_rep_dd.value
    _layer = ca_layer_slider.value

    _row = df_coactivation[
        (df_coactivation["dataset"] == _ds)
        & (df_coactivation["moe_config"] == _mc)
        & (df_coactivation["rep_level"] == _rep)
        & (df_coactivation["layer"] == _layer)
    ]

    if len(_row) > 0:
        _pairs = _row.iloc[0]["top_pairs"]
        _pairs_df = pd.DataFrame(_pairs, columns=["Expert A", "Expert B", "Frequency"])
        _pairs_df["Expert A"] = _pairs_df["Expert A"].astype(int)
        _pairs_df["Expert B"] = _pairs_df["Expert B"].astype(int)
        _pairs_df["Rank"] = range(1, len(_pairs_df) + 1)
        _pairs_df = _pairs_df[["Rank", "Expert A", "Expert B", "Frequency"]]

        mo.vstack([
            mo.md(
                f"**Top Co-Activated Pairs** — {_ds} / {_mc} / {_rep} / Layer {_layer}"
            ),
            mo.ui.table(_pairs_df, selection=None),
        ])
    else:
        mo.md("No data available for this selection.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section E: Cross-Analysis

    These scatter plots test whether the three analysis dimensions (knockout, ossification,
    co-activation) are correlated with each other — for instance, whether models that
    ossify faster also develop more unequal expert importance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Ossification Speed vs. Knockout Inequality** — Each point is one experiment
    (dataset × expert count × repetition level). The x-axis is the training step at
    which mean routing stability first reaches 90%, and the y-axis is the Gini
    coefficient of knockout impacts. A positive correlation would suggest that models
    whose routing locks in earlier also develop more unequal expert utilization.
    Point labels show the repetition level; larger points indicate higher repetition.
    """)
    return


@app.cell(hide_code=True)
def _(DATASET_COLORS, df_knockout, df_ossification, go, np):
    """Plot 14 — Ossification Speed vs. Knockout Inequality."""

    _THRESHOLD = 0.90
    _SYMBOL_MAP = {"moe32": "circle", "moe64": "diamond"}

    def _gini14(values):
        _v = np.sort(np.abs(values))
        _n = len(_v)
        if _n == 0 or _v.sum() == 0:
            return 0.0
        _index = np.arange(1, _n + 1)
        return (2 * np.sum(_index * _v) - (_n + 1) * np.sum(_v)) / (_n * np.sum(_v))

    def _steps_to_thresh14(_group, _thresh):
        _group = _group.sort_values("step")
        _steps_arr = _group["step"].values
        _rates_arr = _group["stability_rate"].values
        for _i in range(len(_rates_arr)):
            if _rates_arr[_i] >= _thresh:
                if _i == 0:
                    return float(_steps_arr[0])
                _r0, _r1 = _rates_arr[_i - 1], _rates_arr[_i]
                _s0, _s1 = _steps_arr[_i - 1], _steps_arr[_i]
                _frac = (_thresh - _r0) / (_r1 - _r0) if _r1 != _r0 else 0.0
                return _s0 + _frac * (_s1 - _s0)
        return float("nan")

    _scatter_rows = []
    for (_ds, _mc, _rep), _ko_group in df_knockout.groupby(
        ["dataset", "moe_config", "rep_level"], observed=True
    ):
        _gini_val = _gini14(_ko_group["loss_increase"].values)

        _oss_sub = df_ossification[
            (df_ossification["dataset"] == _ds)
            & (df_ossification["moe_config"] == _mc)
            & (df_ossification["rep_level"] == _rep)
        ]
        _mean_by_step = _oss_sub.groupby("step")["stability_rate"].mean().reset_index()
        _s2t = _steps_to_thresh14(_mean_by_step, _THRESHOLD)
        _rep_int = _ko_group["rep_int"].iloc[0]

        _scatter_rows.append({
            "dataset": _ds, "moe_config": _mc, "rep_level": str(_rep),
            "rep_int": _rep_int, "gini": _gini_val, "steps_to_thresh": _s2t,
        })

    _fig = go.Figure()
    for _ds in DATASET_COLORS:
        for _mc in _SYMBOL_MAP:
            _sub = [_r for _r in _scatter_rows if _r["dataset"] == _ds and _r["moe_config"] == _mc]
            if not _sub:
                continue
            _fig.add_trace(
                go.Scatter(
                    x=[_r["steps_to_thresh"] for _r in _sub],
                    y=[_r["gini"] for _r in _sub],
                    mode="markers+text",
                    text=[_r["rep_level"] for _r in _sub],
                    textposition="top center",
                    textfont=dict(size=9),
                    name=f"{_ds} / {_mc}",
                    marker=dict(
                        color=DATASET_COLORS[_ds],
                        symbol=_SYMBOL_MAP[_mc],
                        size=[6 + _r["rep_int"] * 0.8 for _r in _sub],
                    ),
                )
            )

    _fig.update_layout(
        title=f"Ossification Speed vs. Knockout Inequality (threshold={_THRESHOLD:.0%})",
        xaxis_title=f"Steps to reach {_THRESHOLD:.0%} stability",
        yaxis_title="Gini coefficient of knockout impacts",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Co-Activation Entropy vs. Mean Knockout Impact** — Each point is one experiment.
    The x-axis is mean co-activation entropy (how uniform expert pairings are) and the
    y-axis is mean knockout Δloss (how critical individual experts are). A negative
    correlation would suggest that more rigid expert pairings (low entropy) go hand-in-hand
    with higher individual expert importance.
    """)
    return


@app.cell(hide_code=True)
def _(DATASET_COLORS, df_coactivation, df_knockout, go):
    """Plot 15 — Entropy vs. Knockout Impact."""

    _SYMBOL_MAP = {"moe32": "circle", "moe64": "diamond"}

    _scatter_rows = []
    for (_ds, _mc, _rep), _ko_group in df_knockout.groupby(
        ["dataset", "moe_config", "rep_level"], observed=True
    ):
        _mean_ko = _ko_group["loss_increase"].mean()
        _rep_int = _ko_group["rep_int"].iloc[0]

        _ca_sub = df_coactivation[
            (df_coactivation["dataset"] == _ds)
            & (df_coactivation["moe_config"] == _mc)
            & (df_coactivation["rep_level"] == _rep)
        ]
        _mean_entropy = _ca_sub["co_activation_entropy"].mean()

        _scatter_rows.append({
            "dataset": _ds, "moe_config": _mc, "rep_level": str(_rep),
            "rep_int": _rep_int, "mean_ko": _mean_ko, "mean_entropy": _mean_entropy,
        })

    _fig = go.Figure()
    for _ds in DATASET_COLORS:
        for _mc in _SYMBOL_MAP:
            _sub = [_r for _r in _scatter_rows if _r["dataset"] == _ds and _r["moe_config"] == _mc]
            if not _sub:
                continue
            _fig.add_trace(
                go.Scatter(
                    x=[_r["mean_entropy"] for _r in _sub],
                    y=[_r["mean_ko"] for _r in _sub],
                    mode="markers+text",
                    text=[_r["rep_level"] for _r in _sub],
                    textposition="top center",
                    textfont=dict(size=9),
                    name=f"{_ds} / {_mc}",
                    marker=dict(
                        color=DATASET_COLORS[_ds],
                        symbol=_SYMBOL_MAP[_mc],
                        size=[6 + _r["rep_int"] * 0.8 for _r in _sub],
                    ),
                )
            )

    _fig.update_layout(
        title="Co-Activation Entropy vs. Mean Knockout Impact",
        xaxis_title="Mean co-activation entropy (nats)",
        yaxis_title="Mean knockout Δloss",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
