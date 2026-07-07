import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regularization & Data-Repetition Figures

    Figures built from the most recent Weights & Biases CSV export in
    `ml/notebooks/data/`.

    Only runs with `State == "finished"` are plotted, and any run tagged `_HIDE`
    is dropped (see the data filtering function below).

    **Figure groups**

    1. Final train loss vs. repetition count (log X), for Dense, MoE32 & MoE64 baseline
       runs on the `olmo_mix` mixture — one figure per model scale (80M, 200M).
    2. The same, repeated for each single-source data mixture: `dclm`,
       `starcoder`, `pes2o`, `wiki`.
    3. Dense / MoE32 / MoE64 (line style) × baseline vs. regularizer value (color),
       for **dropout** and **weight decay** — final train loss vs. repetition (log X).
    4. All `famA` runs: train loss vs. `%dclm` (0/25/50/75/100 from the run name).
    """)
    return


@app.cell(hide_code=True)
def _():
    import glob
    import os
    import re

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    return glob, go, os, pd, re


@app.cell(hide_code=True)
def _(glob, os, pd):
    def find_latest_data_file() -> str:
        """Return the path to the most recently modified CSV in ml/notebooks/data/."""
        _this_dir = (
            os.path.dirname(os.path.abspath(__file__))
            if "__file__" in dir()
            else os.getcwd()
        )
        # Handle being run from the repo root or from ml/notebooks/.
        _candidates = [
            os.path.join(_this_dir, "data"),
            os.path.join(_this_dir, "ml", "notebooks", "data"),
        ]
        _data_dir = next((d for d in _candidates if os.path.isdir(d)), _candidates[0])
        _csvs = glob.glob(os.path.join(_data_dir, "*.csv"))
        if not _csvs:
            raise FileNotFoundError(f"No CSV files found in {_data_dir}")
        return max(_csvs, key=os.path.getmtime)

    DATA_FILE = find_latest_data_file()

    def load_raw_data() -> pd.DataFrame:
        return pd.read_csv(DATA_FILE)

    raw_df = load_raw_data()
    return DATA_FILE, raw_df


@app.cell(hide_code=True)
def _(DATA_FILE, mo, os, raw_df):
    mo.md(f"""
    **Loaded:** `{os.path.basename(DATA_FILE)}`  \n
    **Total runs:** {len(raw_df)}  \n
    **States present:** {", ".join(sorted(raw_df["State"].dropna().unique()))}
    """)
    return


@app.cell(hide_code=True)
def _(pd):
    # ------------------------------------------------------------------
    # Data filtering + parsing helpers.
    # ------------------------------------------------------------------

    # Column names in the wandb export.
    LOSS_COL = "train/CE loss"
    DROPOUT_COL = "model.block.dropout"
    WD_COL = "train_module.optim.weight_decay"
    REPS_COL = "data_reps"

    # Data-source tags (single-source mixes + the full olmo mix).
    DATA_TAGS = ["olmo_mix", "dclm", "starcoder", "pes2o", "wiki"]

    def filter_finished(df: pd.DataFrame) -> pd.DataFrame:
        """Keep only finished runs, dropping any tagged '_HIDE'."""
        out = df[df["State"] == "finished"].copy()
        hide_mask = out["Tags"].apply(lambda t: "_HIDE" in parse_tags(t))
        return out[~hide_mask].copy()

    def parse_tags(tag_field) -> list:
        """Split the comma-separated Tags cell into a clean list."""
        if not isinstance(tag_field, str):
            return []
        return [t.strip() for t in tag_field.split(",") if t.strip()]

    def has_tag(row, tag: str) -> bool:
        return tag in parse_tags(row.get("Tags"))

    def get_scale(row) -> str:
        """Model scale is encoded in the run name, e.g. 'olmo2_ml_80M'."""
        name = str(row.get("Name", ""))
        if "200M" in name:
            return "200M"
        if "80M" in name:
            return "80M"
        return "other"

    def get_model_type(row) -> str:
        """Classify a run as 'dense', 'moe64', 'moe32', or 'other' from its tags."""
        tags = parse_tags(row.get("Tags"))
        if "dense" in tags:
            return "dense"
        if "MoE64" in tags:
            return "moe64"
        if "MoE32" in tags:
            return "moe32"
        return "other"

    def get_reps(row) -> float:
        """Repetition count. Prefer the numeric data_reps column."""
        val = row.get(REPS_COL)
        try:
            if pd.notna(val):
                return float(val)
        except (TypeError, ValueError):
            pass
        return float("nan")

    def get_data_tag(row):
        """Return the single data-source tag on this run (or None)."""
        tags = parse_tags(row.get("Tags"))
        for dt in DATA_TAGS:
            if dt in tags:
                return dt
        return None

    return (
        DROPOUT_COL,
        WD_COL,
        filter_finished,
        get_model_type,
        get_reps,
        get_scale,
        has_tag,
    )


@app.cell
def _(filter_finished, raw_df):
    # Single filtered frame reused by every figure below.
    finished_df = filter_finished(raw_df)
    return (finished_df,)


@app.cell(hide_code=True)
def _(mo, raw_df):
    # ------------------------------------------------------------------
    # Metric selector: train loss + all eval CE-loss metrics.
    # ------------------------------------------------------------------
    def build_metric_options(df):
        """Collect the plottable metric columns present in the data.

        Includes the train CE loss, every ``eval/lm/*/CE loss`` metric, and every
        ``eval/downstream/* (CE loss)`` metric (the non-``v2`` variant).
        """
        cols = list(df.columns)
        options = []
        if "train/CE loss" in cols:
            options.append("train/CE loss")
        for c in cols:
            if c.startswith("eval/lm/") and c.endswith("/CE loss"):
                options.append(c)
        for c in cols:
            if c.startswith("eval/downstream/") and c.endswith("(CE loss)"):
                options.append(c)
        return options

    METRIC_OPTIONS = build_metric_options(raw_df)

    metric_dropdown = mo.ui.dropdown(
        options=METRIC_OPTIONS,
        value="train/CE loss" if "train/CE loss" in METRIC_OPTIONS else METRIC_OPTIONS[0],
        label="Metric to plot",
    )
    metric_dropdown
    return (metric_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group 1 & 2 — Final loss vs. repetition (baseline runs)

    For each data mixture we plot Dense, MoE32 and MoE64 **baseline** runs (tag
    `baseline`, i.e. no dropout/EOM/FOM/jitter). Final train loss on Y, repetition
    count on a log X axis. If more than one baseline run exists at the same
    repetition count, a warning is printed and the **lowest** train loss is kept.
    """)
    return


@app.cell(hide_code=True)
def _(get_model_type, get_reps, get_scale, go, has_tag, pd):
    def collect_baseline_points(
        df: pd.DataFrame, data_tag: str, scale: str, model_type: str, metric: str
    ):
        """Return sorted (reps, value) points for baseline runs of one series.

        Prints a warning when multiple baseline runs share a repetition count, and
        keeps the run with the lowest metric value.
        """
        rows = []
        for _, row in df.iterrows():
            if get_scale(row) != scale:
                continue
            if get_model_type(row) != model_type:
                continue
            if not has_tag(row, "baseline"):
                continue
            if not has_tag(row, data_tag):
                continue
            reps = get_reps(row)
            loss = row.get(metric)
            if pd.isna(reps) or pd.isna(loss):
                continue
            rows.append((reps, float(loss), str(row.get("Name", ""))))

        # Collapse duplicates at the same rep count -> keep min value, warn.
        by_reps: dict = {}
        for reps, loss, name in rows:
            by_reps.setdefault(reps, []).append((loss, name))

        points = []
        for reps in sorted(by_reps):
            entries = by_reps[reps]
            if len(entries) > 1:
                print(
                    f"WARNING: {len(entries)} baseline runs for "
                    f"{data_tag}/{scale}/{model_type} at rep={reps:g}; "
                    f"keeping lowest value. Runs: {[n for _, n in entries]}"
                )
            best_loss = min(loss for loss, _ in entries)
            points.append((reps, best_loss))
        return points

    def make_repetition_figure(df: pd.DataFrame, data_tag: str, scale: str, metric: str):
        """One figure: Dense + MoE32 + MoE64 baseline, metric vs reps (log X)."""
        fig = go.Figure()
        colors = {"dense": "#1f77b4", "moe32": "#2ca02c", "moe64": "#d62728"}
        any_data = False
        for model_type in ["dense", "moe32", "moe64"]:
            points = collect_baseline_points(df, data_tag, scale, model_type, metric)
            if not points:
                continue
            any_data = True
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=model_type,
                    line=dict(color=colors[model_type], width=2),
                    marker=dict(size=8),
                )
            )
        title = f"{data_tag} — {scale}: {metric} vs. repetition (baseline)"
        if not any_data:
            title += "  [no finished baseline runs]"
        fig.update_layout(
            title=title,
            xaxis_title="repetition count",
            yaxis_title=metric,
            xaxis_type="log",
            template="plotly_white",
            width=700,
            height=450,
        )
        return fig

    return (make_repetition_figure,)


@app.cell
def _(finished_df, make_repetition_figure, metric_dropdown):
    # Group 1: olmo_mix, 80M then 200M.
    fig_olmo_80m = make_repetition_figure(
        finished_df, "olmo_mix", "80M", metric_dropdown.value
    )
    fig_olmo_80m
    return


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, metric_dropdown):
    fig_olmo_200m = make_repetition_figure(
        finished_df, "olmo_mix", "200M", metric_dropdown.value
    )
    fig_olmo_200m
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Group 2 — Single-source mixtures (`dclm`, `starcoder`, `pes2o`, `wiki`)
    """)
    return


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, metric_dropdown):
    fig_dclm_80m = make_repetition_figure(finished_df, "dclm", "80M", metric_dropdown.value)
    fig_dclm_80m
    return


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, metric_dropdown):
    fig_dclm_200m = make_repetition_figure(
        finished_df, "dclm", "200M", metric_dropdown.value
    )
    fig_dclm_200m
    return


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, metric_dropdown):
    fig_starcoder_80m = make_repetition_figure(
        finished_df, "starcoder", "80M", metric_dropdown.value
    )
    fig_starcoder_80m
    return


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, metric_dropdown):
    fig_starcoder_200m = make_repetition_figure(
        finished_df, "starcoder", "200M", metric_dropdown.value
    )
    fig_starcoder_200m
    return


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, metric_dropdown):
    fig_pes2o_80m = make_repetition_figure(
        finished_df, "pes2o", "80M", metric_dropdown.value
    )
    fig_pes2o_80m
    return


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, metric_dropdown):
    fig_pes2o_200m = make_repetition_figure(
        finished_df, "pes2o", "200M", metric_dropdown.value
    )
    fig_pes2o_200m
    return


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, metric_dropdown):
    fig_wiki_80m = make_repetition_figure(finished_df, "wiki", "80M", metric_dropdown.value)
    fig_wiki_80m
    return


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, metric_dropdown):
    fig_wiki_200m = make_repetition_figure(
        finished_df, "wiki", "200M", metric_dropdown.value
    )
    fig_wiki_200m
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group 3 — Regularizer sweeps: dropout & weight decay

    Model type is encoded as **line style** (solid = dense, dotted = moe32,
    dashed = moe64). Baseline vs. each regularizer value is encoded as **color**.
    X axis is the repetition count (log); Y is final train CE loss. Runs are
    restricted to the `olmo_mix` mixture at 80M scale (where the regularizer sweeps
    live). Note the dropout/weight-decay sweeps currently only cover dense and moe64.
    """)
    return


@app.cell(hide_code=True)
def _(get_model_type, get_reps, get_scale, go, has_tag, pd):
    def _factor_value(row, col):
        """Read a regularizer column as a float; blank/NaN means 'baseline'."""
        val = row.get(col)
        try:
            if pd.notna(val) and str(val).strip() != "":
                return float(val)
        except (TypeError, ValueError):
            return None
        return None

    def make_factor_figure(df, factor_col, factor_label, baseline_value, metric):
        """dense/moe32/moe64 -> line dash; baseline vs each factor value -> color.

        `baseline_value` is the value that represents 'no regularization' for this
        column (e.g. dropout baseline = None/blank, weight-decay baseline = 0.1).
        """
        # Gather points keyed by (model_type, factor_value) -> {reps: min_value}.
        series: dict = {}
        factor_values = set()
        for _, row in df.iterrows():
            if get_scale(row) != "80M":
                continue
            if not has_tag(row, "olmo_mix"):
                continue
            model_type = get_model_type(row)
            if model_type not in ("dense", "moe32", "moe64"):
                continue
            reps = get_reps(row)
            loss = row.get(metric)
            if pd.isna(reps) or pd.isna(loss):
                continue

            fval = _factor_value(row, factor_col)
            # Normalize the "baseline" bucket.
            if fval is None or (baseline_value is not None and fval == baseline_value):
                fval_key = "baseline"
            else:
                fval_key = fval
                factor_values.add(fval)

            key = (model_type, fval_key)
            series.setdefault(key, {})
            prev = series[key].get(reps)
            series[key][reps] = min(loss, prev) if prev is not None else float(loss)

        # Color per factor bucket; dash per model type.
        ordered_buckets = ["baseline"] + sorted(factor_values)
        palette = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd", "#ff7f0e", "#8c564b"]
        color_map = {b: palette[i % len(palette)] for i, b in enumerate(ordered_buckets)}
        dash_map = {"dense": "solid", "moe32": "dot", "moe64": "dash"}

        fig = go.Figure()
        for (model_type, fval_key), pts in sorted(
            series.items(), key=lambda kv: (str(kv[0][1]), kv[0][0])
        ):
            reps_sorted = sorted(pts)
            xs = reps_sorted
            ys = [pts[r] for r in reps_sorted]
            label_val = (
                "baseline" if fval_key == "baseline" else f"{factor_label}={fval_key:g}"
            )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=f"{model_type}, {label_val}",
                    line=dict(color=color_map[fval_key], dash=dash_map[model_type], width=2),
                    marker=dict(size=7),
                )
            )
        fig.update_layout(
            title=f"Dense (solid) / MoE32 (dotted) / MoE64 (dashed) — "
            f"baseline vs {factor_label} (olmo_mix, 80M)",
            xaxis_title="repetition count",
            yaxis_title=metric,
            xaxis_type="log",
            template="plotly_white",
            width=800,
            height=500,
        )
        return fig

    return (make_factor_figure,)


@app.cell
def _(DROPOUT_COL, finished_df, make_factor_figure, metric_dropdown):
    # Dropout: baseline = no dropout set (blank).
    fig_dropout = make_factor_figure(
        finished_df, DROPOUT_COL, "dropout", None, metric_dropdown.value
    )
    fig_dropout
    return


@app.cell(hide_code=True)
def _(WD_COL, finished_df, make_factor_figure, metric_dropdown):
    # Weight decay: baseline = the default 0.1; swept values are 0.2 / 0.4.
    fig_weight_decay = make_factor_figure(
        finished_df, WD_COL, "weight_decay", 0.1, metric_dropdown.value
    )
    fig_weight_decay
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group 4 — `famA` runs: train loss vs. %dclm

    Every finished run whose name contains `famA`. The X axis is the DCLM
    percentage parsed from `dclm{N}` in the run name (0, 25, 50, 75, 100);
    Y is final train CE loss. Points are colored by model type.
    """)
    return


@app.cell(hide_code=True)
def _(finished_df, get_model_type, go, metric_dropdown, pd, re):
    def parse_dclm_pct(name: str):
        """Extract the DCLM percentage from a 'dclm{N}' token in the run name."""
        m = re.search(r"dclm(\d+)", str(name))
        return int(m.group(1)) if m else None

    def make_famA_figure(df, metric):
        fig = go.Figure()
        colors = {"dense": "#1f77b4", "moe64": "#d62728", "moe32": "#2ca02c"}
        # Collect points per model type.
        per_type: dict = {}
        for _, row in df.iterrows():
            name = str(row.get("Name", ""))
            if "famA" not in name:
                continue
            pct = parse_dclm_pct(name)
            loss = row.get(metric)
            if pct is None or pd.isna(loss):
                continue
            mt = get_model_type(row)
            per_type.setdefault(mt, []).append((pct, float(loss)))

        for mt in sorted(per_type):
            pts = sorted(per_type[mt])
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in pts],
                    y=[p[1] for p in pts],
                    mode="lines+markers",
                    name=mt,
                    line=dict(color=colors.get(mt, "#7f7f7f"), width=2),
                    marker=dict(size=9),
                )
            )
        fig.update_layout(
            title=f"famA runs: {metric} vs. %dclm",
            xaxis_title="% dclm",
            yaxis_title=metric,
            xaxis=dict(tickmode="array", tickvals=[0, 25, 50, 75, 100]),
            template="plotly_white",
            width=700,
            height=450,
        )
        return fig

    fig_famA = make_famA_figure(finished_df, metric_dropdown.value)
    fig_famA
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
