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

    **Styling.** Line/scatter/bar plots follow the conventions of
    `regularization_figures.py`: the swept factor is encoded as **color** (viridis
    palette), the **architecture** (moe32/moe64) is encoded as **line style + shade**
    (moe32 dashed/darker, moe64 dotted/lighter), repetition axes use **power-of-2
    ticks**, and legends are single boxes with bold section headers. Each such figure
    is also exported to `ml/figures/` as a PDF. Heatmaps, the co-activation matrix
    explorer, box plots, and the pairs table stay in plotly (with the viridis palette)
    since they have no matplotlib analog in the reference notebook.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    import os
    import re

    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend: static, browser-free rendering
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import seaborn as sns
    from plotly.subplots import make_subplots

    sns.set_theme(style="whitegrid", context="notebook")
    return go, json, make_subplots, np, os, pd, plt, re


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


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
def _(DATASETS, REP_LEVELS, os, plt):
    # ------------------------------------------------------------------
    # Shared styling helpers, mirroring regularization_figures.py.
    #   - color  -> the swept factor (viridis palette)
    #   - moe32/moe64 -> line style + shade
    #   - repetition axis -> power-of-2 ticks (log2)
    #   - single-box legends with bold section headers
    #   - every matplotlib figure is exported to ml/figures/
    # ------------------------------------------------------------------
    from matplotlib.colors import to_hex
    from matplotlib.lines import Line2D

    REP_INT = {"1x": 1, "2x": 2, "4x": 4, "8x": 8, "16x": 16, "32x": 32}

    # Per-architecture lightening (moe32 darker, moe64 lighter) + line style, matching
    # the moe entries in regularization_figures.py.
    MODEL_SHADE = {"moe32": 0.15, "moe64": 0.35}
    ARCH_DASH = {"moe32": "--", "moe64": ":"}

    def shade_color(hex_color, lighten):
        """Positive `lighten` blends toward white, negative toward black; clamped."""
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        t = max(-1.0, min(1.0, lighten))
        if t >= 0:
            r, g, b = (round(c + (255 - c) * t) for c in (r, g, b))
        else:
            r, g, b = (round(c * (1 + t)) for c in (r, g, b))
        return f"#{r:02x}{g:02x}{b:02x}"

    def arch_dash(mc):
        return ARCH_DASH.get(mc, "-")

    def arch_color(base_color, mc):
        """Shade a per-factor base color by architecture."""
        return shade_color(base_color, MODEL_SHADE.get(mc, 0.0))

    def factor_palette(keys, high_end=0.8):
        """Distinct color per key (in order) across viridis; single key -> mid-palette.

        `high_end` caps how far toward yellow we go (0.8 = stop 80% of the way, so the
        bright-yellow end of viridis is never used).
        """
        keys = list(keys)
        n = len(keys)
        cmap = plt.get_cmap("viridis")
        if n == 0:
            return {}
        if n == 1:
            return {keys[0]: to_hex(cmap(0.35))}
        return {k: to_hex(cmap(i * high_end / (n - 1))) for i, k in enumerate(keys)}

    # Fixed palettes for the recurring factors.
    DATASET_COLORS = factor_palette(DATASETS)
    REP_COLORS = factor_palette(REP_LEVELS)
    MOE_COLORS = factor_palette(["moe32", "moe64"])  # for plotly box/heatmap accents

    # Truncated viridis for plotly heatmaps: 0 -> 0.8 of the colormap, so the bright
    # yellow end is never used (matches the matplotlib factor_palette cap).
    _cs_cmap = plt.get_cmap("viridis")
    VIRIDIS_CAPPED = [
        [_i / 8, to_hex(_cs_cmap(_i / 8 * 0.8))] for _i in range(9)
    ]

    def pow2_ticks(max_reps):
        if max_reps is None or max_reps < 1:
            return [1], ["1"]
        vals = []
        p = 1
        while p < max_reps:
            vals.append(p)
            p *= 2
        vals.append(p)
        return vals, [str(v) for v in vals]

    def apply_pow2_xaxis(ax, max_reps, any_data=True):
        if not any_data:
            return
        tickvals, ticktext = pow2_ticks(max_reps)
        ax.set_xscale("log", base=2)
        ax.set_xticks(tickvals)
        ax.set_xticklabels(ticktext)
        ax.minorticks_off()

    def _header():
        return Line2D([0], [0], linestyle="none", marker="", label="")

    def legend_sections(color_section=None, style_section=None, marker_section=None):
        """Build (handles, labels, titles) for a single-box multi-section legend.

        Each section is ``(title, [(label, spec), ...])``: color -> hex, style -> dash
        (drawn grey), marker -> matplotlib marker (drawn grey). Titles are returned so
        the caller can bold them after placing the legend.
        """
        handles, labels, titles = [], [], []
        if color_section:
            title, items = color_section
            titles.append(title)
            handles.append(_header())
            labels.append(title)
            for lbl, col in items:
                handles.append(Line2D([0], [0], color=col, linewidth=2))
                labels.append(str(lbl))
        if style_section:
            title, items = style_section
            titles.append(title)
            handles.append(_header())
            labels.append(title)
            for lbl, dash in items:
                handles.append(
                    Line2D([0], [0], color="#333333", linewidth=1.5, linestyle=dash)
                )
                labels.append(str(lbl))
        if marker_section:
            title, items = marker_section
            titles.append(title)
            handles.append(_header())
            labels.append(title)
            for lbl, mk in items:
                handles.append(
                    Line2D([0], [0], color="#333333", marker=mk, linestyle="none")
                )
                labels.append(str(lbl))
        return handles, labels, titles

    def place_legend(target, handles, labels, titles, on_fig=False):
        """Place a legend on an Axes (right of plot) or a Figure (center-right)."""
        if on_fig:
            leg = target.legend(
                handles, labels, fontsize=7, handlelength=2.2,
                loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0.0,
            )
        else:
            leg = target.legend(
                handles, labels, fontsize=7, handlelength=2.2,
                loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
            )
        for txt in leg.get_texts():
            if txt.get_text() in titles:
                txt.set_fontweight("bold")
        return leg

    # PDF export (matches regularization_figures.py: static, browser-free).
    def _figures_dir():
        _this_dir = (
            os.path.dirname(os.path.abspath(__file__))
            if "__file__" in dir()
            else os.getcwd()
        )
        _candidates = [
            os.path.join(_this_dir, "..", "figures"),
            os.path.join(_this_dir, "ml", "figures"),
        ]
        for d in _candidates:
            if os.path.isdir(os.path.dirname(os.path.normpath(d))):
                return os.path.normpath(d)
        return os.path.normpath(_candidates[0])

    FIGURES_DIR = _figures_dir()

    def _sanitize(name):
        return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name).strip("_")

    def save_pdf(fig, name):
        """Write `fig` to ml/figures/moe_rep__<name>.pdf (failures are non-fatal)."""
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, f"moe_rep__{_sanitize(name)}.pdf")
        try:
            fig.savefig(path, bbox_inches="tight")
        except Exception as e:  # pragma: no cover - env dependent
            print(f"NOTE: could not save PDF '{path}': {type(e).__name__}: {e}")
        return fig

    return (
        DATASET_COLORS,
        MOE_COLORS,
        REP_COLORS,
        REP_INT,
        VIRIDIS_CAPPED,
        apply_pow2_xaxis,
        arch_color,
        arch_dash,
        factor_palette,
        legend_sections,
        place_legend,
        save_pdf,
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

    Color = dataset; line style + shade = expert count (moe32 dashed, moe64 dotted).
    """)
    return


@app.cell(hide_code=True)
def _(
    DATASET_COLORS,
    REP_INT,
    active_datasets,
    active_moe,
    apply_pow2_xaxis,
    arch_color,
    arch_dash,
    df_coactivation,
    df_knockout,
    df_ossification,
    legend_sections,
    place_legend,
    plt,
    save_pdf,
):
    """Plot 1 — Key Metrics vs. Repetition (3 stacked subplots)."""

    def _xy(series_by_rep):
        pts = [(REP_INT[str(r)], v) for r, v in series_by_rep.items() if v == v]
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

    _fig1, _axes = plt.subplots(3, 1, figsize=(7.0, 10.0), sharex=True)
    _titles = [
        "Mean Knockout Impact (loss increase)",
        "Mean Co-Activation Entropy",
        "End-of-Training Routing Stability",
    ]
    _ylabels = ["Mean Δloss", "Entropy (nats)", "Stability rate"]
    _max_reps = 1
    for _ds in active_datasets:
        for _mc in active_moe:
            _col = arch_color(DATASET_COLORS[_ds], _mc)
            _dash = arch_dash(_mc)

            _ko = df_knockout[(df_knockout["dataset"] == _ds) & (df_knockout["moe_config"] == _mc)]
            _ko_agg = _ko.groupby("rep_level", observed=True)["loss_increase"].mean()

            _ca = df_coactivation[(df_coactivation["dataset"] == _ds) & (df_coactivation["moe_config"] == _mc)]
            _ca_agg = _ca.groupby("rep_level", observed=True)["co_activation_entropy"].mean()

            _oss = df_ossification[(df_ossification["dataset"] == _ds) & (df_ossification["moe_config"] == _mc)]
            _oss_final = _oss[_oss["step"] == _oss["step"].max()]
            _oss_agg = _oss_final.groupby("rep_level", observed=True)["stability_rate"].mean()

            for _ax, _agg in zip(_axes, [_ko_agg, _ca_agg, _oss_agg]):
                _xs, _ys = _xy(_agg)
                if not _xs:
                    continue
                _max_reps = max(_max_reps, max(_xs))
                _ax.plot(_xs, _ys, marker="o", markersize=4, linewidth=1.5,
                         linestyle=_dash, color=_col)

    for _ax, _t, _yl in zip(_axes, _titles, _ylabels):
        _ax.set_title(_t, fontsize=10)
        _ax.set_ylabel(_yl)
        apply_pow2_xaxis(_ax, _max_reps)
    _axes[-1].set_xlabel("repetition count")
    _fig1.suptitle("Key Metrics vs. Data Repetition", fontsize=12)

    _h, _l, _t = legend_sections(
        color_section=("Dataset", [(_ds, DATASET_COLORS[_ds]) for _ds in active_datasets]),
        style_section=("Expert count", [(_mc, arch_dash(_mc)) for _mc in active_moe]),
    )
    place_legend(_fig1, _h, _l, _t, on_fig=True)
    _fig1.tight_layout(rect=(0, 0, 0.85, 0.97))
    save_pdf(_fig1, "key_metrics_vs_rep")
    _fig1
    return


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
    REP_INT,
    active_datasets,
    active_moe,
    apply_pow2_xaxis,
    arch_color,
    arch_dash,
    df_knockout,
    legend_sections,
    place_legend,
    plt,
    save_pdf,
):
    """Plot 2 — Baseline Loss vs. Repetition."""

    _fig2, _ax = plt.subplots(figsize=(6.5, 4.5))
    _max_reps = 1
    for _ds in active_datasets:
        for _mc in active_moe:
            _sub = df_knockout[(df_knockout["dataset"] == _ds) & (df_knockout["moe_config"] == _mc)]
            _agg = _sub.groupby("rep_level", observed=True)["baseline_loss"].first()
            _pts = sorted((REP_INT[str(r)], v) for r, v in _agg.items() if v == v)
            if not _pts:
                continue
            _max_reps = max(_max_reps, _pts[-1][0])
            _ax.plot([p[0] for p in _pts], [p[1] for p in _pts], marker="o", markersize=4,
                     linewidth=1.5, linestyle=arch_dash(_mc), color=arch_color(DATASET_COLORS[_ds], _mc))

    apply_pow2_xaxis(_ax, _max_reps)
    _ax.set_xlabel("repetition count")
    _ax.set_ylabel("Baseline loss")
    _ax.set_title("Baseline Loss vs. Data Repetition", fontsize=10)
    _h, _l, _t = legend_sections(
        color_section=("Dataset", [(_ds, DATASET_COLORS[_ds]) for _ds in active_datasets]),
        style_section=("Expert count", [(_mc, arch_dash(_mc)) for _mc in active_moe]),
    )
    place_legend(_ax, _h, _l, _t)
    _fig2.tight_layout()
    save_pdf(_fig2, "baseline_loss_vs_rep")
    _fig2
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
def _(
    REP_LEVELS,
    VIRIDIS_CAPPED,
    active_datasets,
    active_moe,
    df_knockout,
    go,
    make_subplots,
):
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
                        colorscale=VIRIDIS_CAPPED,
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
    MOE_COLORS,
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
                        marker_color=MOE_COLORS.get(_mc, "#000"),
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
    Uses the first active dataset and expert count. Color = expert (viridis).
    """)
    return


@app.cell(hide_code=True)
def _(
    REP_INT,
    active_datasets,
    active_moe,
    apply_pow2_xaxis,
    df_knockout,
    factor_palette,
    ko_layer_slider,
    ko_topk_slider,
    plt,
    save_pdf,
):
    """Plot 5 — Expert Importance Ranking (Bump Chart)."""

    _layer = ko_layer_slider.value
    _topk = ko_topk_slider.value
    _ds = active_datasets[0] if active_datasets else "starcoder"
    _mc = active_moe[0] if active_moe else "moe64"

    _sub = df_knockout[
        (df_knockout["dataset"] == _ds)
        & (df_knockout["moe_config"] == _mc)
        & (df_knockout["layer"] == _layer)
    ].copy()
    _sub["rank"] = _sub.groupby("rep_level", observed=True)["loss_increase"].rank(
        ascending=False, method="min"
    )
    _baseline = _sub[_sub["rep_level"] == "1x"].nsmallest(_topk, "rank")
    _top_experts = _baseline["expert_idx"].tolist()

    _expert_color = factor_palette([str(e) for e in _top_experts])
    _fig5, _ax = plt.subplots(figsize=(6.5, 5.0))
    _max_reps = 1
    for _expert_id in _top_experts:
        _ed = _sub[_sub["expert_idx"] == _expert_id].sort_values("rep_int")
        _xs = [REP_INT[str(r)] for r in _ed["rep_level"]]
        if not _xs:
            continue
        _max_reps = max(_max_reps, max(_xs))
        _ax.plot(_xs, _ed["rank"], marker="o", markersize=5, linewidth=1.5,
                 color=_expert_color[str(_expert_id)], label=f"Expert {_expert_id}")

    apply_pow2_xaxis(_ax, _max_reps)
    _ax.set_xlabel("repetition count")
    _ax.set_ylabel("Rank (1 = most important)")
    _ax.invert_yaxis()
    _ax.set_title(f"Expert Importance Ranking — {_ds} / {_mc} / Layer {_layer} (Top {_topk} at 1x)", fontsize=10)
    _ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, title="Expert")
    _fig5.tight_layout()
    save_pdf(_fig5, f"bump_L{_layer}_{_ds}_{_mc}")
    _fig5
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
    REP_INT,
    active_datasets,
    active_moe,
    apply_pow2_xaxis,
    arch_color,
    arch_dash,
    df_knockout,
    legend_sections,
    np,
    place_legend,
    plt,
    save_pdf,
):
    """Plot 6 — Expert Importance Inequality (Gini Coefficient)."""

    def _gini(values):
        _v = np.sort(np.abs(values))
        _n = len(_v)
        if _n == 0 or _v.sum() == 0:
            return 0.0
        _index = np.arange(1, _n + 1)
        return (2 * np.sum(_index * _v) - (_n + 1) * np.sum(_v)) / (_n * np.sum(_v))

    _fig6, _ax = plt.subplots(figsize=(6.5, 4.5))
    _max_reps = 1
    for _ds in active_datasets:
        for _mc in active_moe:
            _sub = df_knockout[(df_knockout["dataset"] == _ds) & (df_knockout["moe_config"] == _mc)]
            _g = _sub.groupby("rep_level", observed=True)["loss_increase"].apply(_gini)
            _pts = sorted((REP_INT[str(r)], v) for r, v in _g.items() if v == v)
            if not _pts:
                continue
            _max_reps = max(_max_reps, _pts[-1][0])
            _ax.plot([p[0] for p in _pts], [p[1] for p in _pts], marker="o", markersize=4,
                     linewidth=1.5, linestyle=arch_dash(_mc), color=arch_color(DATASET_COLORS[_ds], _mc))

    apply_pow2_xaxis(_ax, _max_reps)
    _ax.set_xlabel("repetition count")
    _ax.set_ylabel("Gini coefficient")
    _ax.set_title("Expert Importance Inequality (Gini) vs. Repetition", fontsize=10)
    _h, _l, _t = legend_sections(
        color_section=("Dataset", [(_ds, DATASET_COLORS[_ds]) for _ds in active_datasets]),
        style_section=("Expert count", [(_mc, arch_dash(_mc)) for _mc in active_moe]),
    )
    place_legend(_ax, _h, _l, _t)
    _fig6.tight_layout()
    save_pdf(_fig6, "gini_vs_rep")
    _fig6
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
    faster or plateau higher indicate earlier or stronger ossification. Color = repetition
    level (viridis, light→dark); facets are expert count (rows) × dataset (columns).
    """)
    return


@app.cell(hide_code=True)
def _(
    REP_COLORS,
    REP_LEVELS,
    active_datasets,
    active_moe,
    df_ossification,
    legend_sections,
    mo,
    place_legend,
    plt,
    save_pdf,
):
    """Plot 7 — Ossification Trajectories (faceted by dataset × moe)."""

    _n_moe = len(active_moe)
    _n_ds = len(active_datasets)

    if _n_moe == 0 or _n_ds == 0:
        _out7 = mo.md("*No data selected.*")
    else:
        _fig7, _axes = plt.subplots(
            _n_moe, _n_ds, figsize=(4.2 * _n_ds, 3.4 * _n_moe), squeeze=False
        )
        for _r_idx, _mc in enumerate(active_moe):
            for _c_idx, _ds in enumerate(active_datasets):
                _ax = _axes[_r_idx][_c_idx]
                _sub = df_ossification[
                    (df_ossification["dataset"] == _ds) & (df_ossification["moe_config"] == _mc)
                ]
                _mean_per_step = _sub.groupby(
                    ["rep_level", "step"], observed=True
                )["stability_rate"].mean().reset_index()
                for _rep in REP_LEVELS:
                    _rd = _mean_per_step[_mean_per_step["rep_level"] == _rep].sort_values("step")
                    if len(_rd) == 0:
                        continue
                    _ax.plot(_rd["step"], _rd["stability_rate"], marker="o", markersize=3,
                             linewidth=1.2, color=REP_COLORS[_rep])
                if _r_idx == 0:
                    _ax.set_title(_ds, fontsize=10, fontweight="bold")
                if _c_idx == 0:
                    _ax.set_ylabel(f"{_mc}\nstability rate", fontsize=9)
                if _r_idx == _n_moe - 1:
                    _ax.set_xlabel("training step")

        _fig7.suptitle("Ossification Trajectories — Mean Stability Over Training", fontsize=12)
        _h, _l, _t = legend_sections(
            color_section=("Repetitions", [(_r, REP_COLORS[_r]) for _r in REP_LEVELS])
        )
        place_legend(_fig7, _h, _l, _t, on_fig=True)
        _fig7.tight_layout(rect=(0, 0, 0.9, 0.97))
        save_pdf(_fig7, "ossification_trajectories")
        _out7 = _fig7

    _out7
    return


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
    REP_COLORS,
    REP_LEVELS,
    active_datasets,
    active_moe,
    df_ossification,
    legend_sections,
    mo,
    oss_layer_slider,
    place_legend,
    plt,
    save_pdf,
):
    """Plot 8 — Per-Layer Ossification (single selected layer)."""

    _layer = oss_layer_slider.value
    _n_moe = len(active_moe)
    _n_ds = len(active_datasets)

    if _n_moe == 0 or _n_ds == 0:
        _out8 = mo.md("*No data selected.*")
    else:
        _fig8, _axes = plt.subplots(
            _n_moe, _n_ds, figsize=(4.2 * _n_ds, 3.4 * _n_moe), squeeze=False
        )
        for _r_idx, _mc in enumerate(active_moe):
            for _c_idx, _ds in enumerate(active_datasets):
                _ax = _axes[_r_idx][_c_idx]
                _sub = df_ossification[
                    (df_ossification["dataset"] == _ds)
                    & (df_ossification["moe_config"] == _mc)
                    & (df_ossification["layer"] == _layer)
                ]
                for _rep in REP_LEVELS:
                    _rd = _sub[_sub["rep_level"] == _rep].sort_values("step")
                    if len(_rd) == 0:
                        continue
                    _ax.plot(_rd["step"], _rd["stability_rate"], marker="o", markersize=3,
                             linewidth=1.2, color=REP_COLORS[_rep])
                if _r_idx == 0:
                    _ax.set_title(_ds, fontsize=10, fontweight="bold")
                if _c_idx == 0:
                    _ax.set_ylabel(f"{_mc}\nstability rate", fontsize=9)
                if _r_idx == _n_moe - 1:
                    _ax.set_xlabel("training step")

        _fig8.suptitle(f"Ossification Trajectories — Layer {_layer}", fontsize=12)
        _h, _l, _t = legend_sections(
            color_section=("Repetitions", [(_r, REP_COLORS[_r]) for _r in REP_LEVELS])
        )
        place_legend(_fig8, _h, _l, _t, on_fig=True)
        _fig8.tight_layout(rect=(0, 0, 0.9, 0.97))
        save_pdf(_fig8, f"ossification_layer_{_layer}")
        _out8 = _fig8

    _out8
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Steps to Reach Stability Threshold** — For each experiment, the training step at
    which mean stability (averaged across layers) first exceeds the threshold set by the
    slider above. Computed via linear interpolation between measured checkpoints. Lower
    bars mean the model's routing decisions locked in earlier during training. Color =
    dataset; hatched bars indicate moe64 configurations.
    """)
    return


@app.cell(hide_code=True)
def _(
    DATASET_COLORS,
    REP_LEVELS,
    active_datasets,
    active_moe,
    arch_color,
    df_ossification,
    legend_sections,
    mo,
    np,
    oss_threshold_slider,
    place_legend,
    plt,
    save_pdf,
):
    """Plot 9 — Steps to Reach Stability Threshold (grouped bars)."""

    _threshold = oss_threshold_slider.value

    def _steps_to_threshold(_group, _thresh):
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

    _groups = [(_ds, _mc) for _ds in active_datasets for _mc in active_moe]
    if not _groups:
        _out9 = mo.md("*No data selected.*")
    else:
        _fig9, _ax = plt.subplots(figsize=(7.5, 4.5))
        _x = np.arange(len(REP_LEVELS))
        _w = 0.8 / max(1, len(_groups))
        for _gi, (_ds, _mc) in enumerate(_groups):
            _sub = df_ossification[
                (df_ossification["dataset"] == _ds) & (df_ossification["moe_config"] == _mc)
            ]
            _mean_by_step = _sub.groupby(
                ["rep_level", "step"], observed=True
            )["stability_rate"].mean().reset_index()
            _vals = []
            for _rep in REP_LEVELS:
                _rd = _mean_by_step[_mean_by_step["rep_level"] == _rep]
                _vals.append(_steps_to_threshold(_rd, _threshold) if len(_rd) else float("nan"))
            _ax.bar(
                _x + (_gi - (len(_groups) - 1) / 2) * _w, _vals, _w,
                color=arch_color(DATASET_COLORS[_ds], _mc),
                hatch="//" if _mc == "moe64" else None,
                edgecolor="white", linewidth=0.4,
            )
        _ax.set_xticks(_x)
        _ax.set_xticklabels(REP_LEVELS)
        _ax.set_xlabel("repetition level")
        _ax.set_ylabel("training step")
        _ax.set_title(f"Steps to Reach {_threshold:.0%} Mean Stability", fontsize=10)
        _h, _l, _t = legend_sections(
            color_section=("Dataset", [(_ds, DATASET_COLORS[_ds]) for _ds in active_datasets]),
            style_section=("Expert count", [(_mc, "-") for _mc in active_moe]),
        )
        place_legend(_ax, _h, _l, _t)
        _fig9.tight_layout()
        save_pdf(_fig9, "steps_to_stability")
        _out9 = _fig9

    _out9
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
    repetition level. Each line represents one layer (colored by depth, viridis).
    Decreasing entropy with repetition would indicate that repeated data causes experts to
    form more rigid, fixed pairing patterns. Facets are expert count (rows) × dataset (cols).
    """)
    return


@app.cell(hide_code=True)
def _(
    REP_INT,
    active_datasets,
    active_moe,
    apply_pow2_xaxis,
    df_coactivation,
    factor_palette,
    legend_sections,
    mo,
    place_legend,
    plt,
    save_pdf,
):
    """Plot 10 — Co-Activation Entropy vs. Rep Level (per layer)."""

    _layer_color = factor_palette([f"L{_i}" for _i in range(8)])
    _n_moe = len(active_moe)
    _n_ds = len(active_datasets)

    if _n_moe == 0 or _n_ds == 0:
        _out10 = mo.md("*No data selected.*")
    else:
        _fig10, _axes = plt.subplots(
            _n_moe, _n_ds, figsize=(4.2 * _n_ds, 3.4 * _n_moe), squeeze=False
        )
        _max_reps = 1
        for _r_idx, _mc in enumerate(active_moe):
            for _c_idx, _ds in enumerate(active_datasets):
                _ax = _axes[_r_idx][_c_idx]
                _sub = df_coactivation[
                    (df_coactivation["dataset"] == _ds) & (df_coactivation["moe_config"] == _mc)
                ]
                for _layer_idx in range(8):
                    _ls = _sub[_sub["layer"] == _layer_idx].sort_values("rep_int")
                    if len(_ls) == 0:
                        continue
                    _xs = [REP_INT[str(r)] for r in _ls["rep_level"]]
                    _max_reps = max(_max_reps, max(_xs))
                    _ax.plot(_xs, _ls["co_activation_entropy"], marker="o", markersize=3,
                             linewidth=1.2, color=_layer_color[f"L{_layer_idx}"])
                apply_pow2_xaxis(_ax, _max_reps)
                if _r_idx == 0:
                    _ax.set_title(_ds, fontsize=10, fontweight="bold")
                if _c_idx == 0:
                    _ax.set_ylabel(f"{_mc}\nentropy (nats)", fontsize=9)
                if _r_idx == _n_moe - 1:
                    _ax.set_xlabel("repetition count")

        _fig10.suptitle("Per-Layer Co-Activation Entropy vs. Repetition", fontsize=12)
        _h, _l, _t = legend_sections(
            color_section=("Layer", [(f"L{_i}", _layer_color[f"L{_i}"]) for _i in range(8)])
        )
        place_legend(_fig10, _h, _l, _t, on_fig=True)
        _fig10.tight_layout(rect=(0, 0, 0.92, 0.97))
        save_pdf(_fig10, "coactivation_entropy_vs_rep")
        _out10 = _fig10

    _out10
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
    VIRIDIS_CAPPED,
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
                colorscale=VIRIDIS_CAPPED,
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
    VIRIDIS_CAPPED,
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
                    colorscale=VIRIDIS_CAPPED,
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
    Color = dataset; marker = expert count (moe32 circle, moe64 diamond); point size
    grows with repetition; text labels show the repetition level.
    """)
    return


@app.cell(hide_code=True)
def _(
    DATASET_COLORS,
    active_datasets,
    active_moe,
    arch_color,
    df_knockout,
    df_ossification,
    legend_sections,
    mo,
    np,
    place_legend,
    plt,
    save_pdf,
):
    """Plot 14 — Ossification Speed vs. Knockout Inequality."""

    _THRESHOLD = 0.90
    _MARKER = {"moe32": "o", "moe64": "D"}

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

    _rows = []
    for (_ds, _mc, _rep), _ko_group in df_knockout.groupby(
        ["dataset", "moe_config", "rep_level"], observed=True
    ):
        if _ds not in active_datasets or _mc not in active_moe:
            continue
        _oss_sub = df_ossification[
            (df_ossification["dataset"] == _ds)
            & (df_ossification["moe_config"] == _mc)
            & (df_ossification["rep_level"] == _rep)
        ]
        _mean_by_step = _oss_sub.groupby("step")["stability_rate"].mean().reset_index()
        _rows.append({
            "dataset": _ds, "moe_config": _mc, "rep_level": str(_rep),
            "rep_int": _ko_group["rep_int"].iloc[0],
            "gini": _gini14(_ko_group["loss_increase"].values),
            "steps_to_thresh": _steps_to_thresh14(_mean_by_step, _THRESHOLD),
        })

    if not _rows:
        _out14 = mo.md("*No data selected.*")
    else:
        _fig14, _ax = plt.subplots(figsize=(6.5, 5.0))
        for _ds in active_datasets:
            for _mc in active_moe:
                _sub = [r for r in _rows if r["dataset"] == _ds and r["moe_config"] == _mc]
                if not _sub:
                    continue
                _ax.scatter(
                    [_pt["steps_to_thresh"] for _pt in _sub],
                    [_pt["gini"] for _pt in _sub],
                    s=[30 + _pt["rep_int"] * 6 for _pt in _sub],
                    color=arch_color(DATASET_COLORS[_ds], _mc),
                    marker=_MARKER[_mc], edgecolor="white", linewidth=0.5, zorder=3,
                )
                for _pt in _sub:
                    _ax.annotate(_pt["rep_level"], (_pt["steps_to_thresh"], _pt["gini"]),
                                 xytext=(4, 4), textcoords="offset points", fontsize=6)
        _ax.set_xlabel(f"steps to reach {_THRESHOLD:.0%} stability")
        _ax.set_ylabel("Gini coefficient of knockout impacts")
        _ax.set_title(f"Ossification Speed vs. Knockout Inequality (threshold={_THRESHOLD:.0%})", fontsize=10)
        _h, _l, _t = legend_sections(
            color_section=("Dataset", [(_ds, DATASET_COLORS[_ds]) for _ds in active_datasets]),
            marker_section=("Expert count", [(_mc, _MARKER[_mc]) for _mc in active_moe]),
        )
        place_legend(_ax, _h, _l, _t)
        _fig14.tight_layout()
        save_pdf(_fig14, "ossification_vs_gini")
        _out14 = _fig14

    _out14
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Co-Activation Entropy vs. Mean Knockout Impact** — Each point is one experiment.
    The x-axis is mean co-activation entropy (how uniform expert pairings are) and the
    y-axis is mean knockout Δloss (how critical individual experts are). A negative
    correlation would suggest that more rigid expert pairings (low entropy) go hand-in-hand
    with higher individual expert importance. Color = dataset; marker = expert count;
    point size grows with repetition.
    """)
    return


@app.cell(hide_code=True)
def _(
    DATASET_COLORS,
    active_datasets,
    active_moe,
    arch_color,
    df_coactivation,
    df_knockout,
    legend_sections,
    mo,
    place_legend,
    plt,
    save_pdf,
):
    """Plot 15 — Entropy vs. Knockout Impact."""

    _MARKER = {"moe32": "o", "moe64": "D"}

    _rows = []
    for (_ds, _mc, _rep), _ko_group in df_knockout.groupby(
        ["dataset", "moe_config", "rep_level"], observed=True
    ):
        if _ds not in active_datasets or _mc not in active_moe:
            continue
        _ca_sub = df_coactivation[
            (df_coactivation["dataset"] == _ds)
            & (df_coactivation["moe_config"] == _mc)
            & (df_coactivation["rep_level"] == _rep)
        ]
        _rows.append({
            "dataset": _ds, "moe_config": _mc, "rep_level": str(_rep),
            "rep_int": _ko_group["rep_int"].iloc[0],
            "mean_ko": _ko_group["loss_increase"].mean(),
            "mean_entropy": _ca_sub["co_activation_entropy"].mean(),
        })

    if not _rows:
        _out15 = mo.md("*No data selected.*")
    else:
        _fig15, _ax = plt.subplots(figsize=(6.5, 5.0))
        for _ds in active_datasets:
            for _mc in active_moe:
                _sub = [r for r in _rows if r["dataset"] == _ds and r["moe_config"] == _mc]
                if not _sub:
                    continue
                _ax.scatter(
                    [_pt["mean_entropy"] for _pt in _sub],
                    [_pt["mean_ko"] for _pt in _sub],
                    s=[30 + _pt["rep_int"] * 6 for _pt in _sub],
                    color=arch_color(DATASET_COLORS[_ds], _mc),
                    marker=_MARKER[_mc], edgecolor="white", linewidth=0.5, zorder=3,
                )
                for _pt in _sub:
                    _ax.annotate(_pt["rep_level"], (_pt["mean_entropy"], _pt["mean_ko"]),
                                 xytext=(4, 4), textcoords="offset points", fontsize=6)
        _ax.set_xlabel("mean co-activation entropy (nats)")
        _ax.set_ylabel("mean knockout Δloss")
        _ax.set_title("Co-Activation Entropy vs. Mean Knockout Impact", fontsize=10)
        _h, _l, _t = legend_sections(
            color_section=("Dataset", [(_ds, DATASET_COLORS[_ds]) for _ds in active_datasets]),
            marker_section=("Expert count", [(_mc, _MARKER[_mc]) for _mc in active_moe]),
        )
        place_legend(_ax, _h, _l, _t)
        _fig15.tight_layout()
        save_pdf(_fig15, "entropy_vs_knockout")
        _out15 = _fig15

    _out15
    return


if __name__ == "__main__":
    app.run()
