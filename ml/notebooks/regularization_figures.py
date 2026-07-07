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

    **Per-plot metric selection.** Every plot has its own multi-select dropdown of
    metrics (train CE loss + all `eval/lm/*/CE loss` and `eval/downstream/* (CE loss)`
    metrics). Selecting multiple metrics renders one figure per metric, side by side.

    **Figure groups**

    1. Final loss vs. repetition count (log X), for Dense, MoE32 & MoE64 baseline
       runs on the `olmo_mix` mixture — one figure per model scale (80M, 200M).
    2. The same, repeated for each single-source data mixture: `dclm`,
       `starcoder`, `pes2o`, `wiki`.
    3. Dense / MoE32 / MoE64 (line style) × baseline vs. regularizer value (color),
       for **dropout** and **weight decay** — loss vs. repetition (log X).
    4. All `famA` runs: loss vs. `%dclm` (0/25/50/75/100 from the run name).
    5. `famB` runs: single-source mixes into `olmo_mix` vs. repetition.
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

    def pow2_ticks(max_reps):
        """Power-of-2 tick values from 1 up to the smallest power of 2 >= max_reps.

        Returns (tickvals, ticktext) for a log-scaled repetition axis. If there is
        no valid max (e.g. no data), returns ([1], ["1"]).
        """
        if max_reps is None or max_reps < 1:
            return [1], ["1"]
        vals = []
        p = 1
        while p < max_reps:
            vals.append(p)
            p *= 2
        vals.append(p)  # smallest power of 2 >= max_reps
        return vals, [str(v) for v in vals]

    # Per-model-type lightening: dense = darkest (0.0), moe64 = lightest.
    MODEL_SHADE = {"dense": 0.0, "moe32": 0.35, "moe64": 0.65}

    def shade_color(hex_color: str, lighten: float) -> str:
        """Lighten a hex color toward white by `lighten` in [0, 1] (0 = unchanged)."""
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        r = round(r + (255 - r) * lighten)
        g = round(g + (255 - g) * lighten)
        b = round(b + (255 - b) * lighten)
        return f"#{r:02x}{g:02x}{b:02x}"

    def model_color(base_hex: str, model_type: str) -> str:
        """Shade of `base_hex` for a model type (dense darkest, moe64 lightest)."""
        return shade_color(base_hex, MODEL_SHADE.get(model_type, 0.0))

    return (
        DROPOUT_COL,
        WD_COL,
        filter_finished,
        get_model_type,
        get_reps,
        get_scale,
        has_tag,
        model_color,
        pow2_ticks,
    )


@app.cell
def _(filter_finished, raw_df):
    # Single filtered frame reused by every figure below.
    finished_df = filter_finished(raw_df)
    return (finished_df,)


@app.cell(hide_code=True)
def _(mo, raw_df):
    # ------------------------------------------------------------------
    # Per-plot metric selection utilities.
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

    _DEFAULT_METRIC = (
        ["train/CE loss"] if "train/CE loss" in METRIC_OPTIONS else METRIC_OPTIONS[:1]
    )

    def metric_multiselect(label, chosen_values=[]):
        """A per-plot multi-select over the available metrics."""
        return mo.ui.multiselect(
            options=METRIC_OPTIONS, value=(chosen_values if chosen_values else _DEFAULT_METRIC), label=label,
            # full_width=True,
        )

    def render_side_by_side(metrics, build_fig):
        """Render one figure per selected metric, laid out side by side.

        `build_fig(metric)` should return a plotly figure for a single metric.
        """
        if not metrics:
            return mo.md("*Select at least one metric.*")
        return mo.hstack(
            [build_fig(m) for m in metrics], widths="equal", gap=1, wrap=True
        )

    return metric_multiselect, render_side_by_side


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group 1 & 2 — Final loss vs. repetition (baseline runs)

    For each data mixture we plot Dense, MoE32 and MoE64 **baseline** runs (tag
    `baseline`, i.e. no dropout/EOM/FOM/jitter). Metric on Y, repetition count on a
    log X axis. If more than one baseline run exists at the same repetition count, a
    warning is printed and the **lowest** value is kept.
    """)
    return


@app.cell(hide_code=True)
def _(
    get_model_type,
    get_reps,
    get_scale,
    go,
    has_tag,
    model_color,
    pd,
    pow2_ticks,
):
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
        """One figure: Dense + MoE32 + MoE64 baseline, metric vs reps (log X).

        Model type is distinguished by both line style and shade of a single base
        color (dense darkest, moe64 lightest).
        """
        fig = go.Figure()
        base_color = "#1f77b4"
        dash_map = {"dense": "solid", "moe32": "dot", "moe64": "dash"}
        any_data = False
        max_reps = 0
        for model_type in ["dense", "moe32", "moe64"]:
            points = collect_baseline_points(df, data_tag, scale, model_type, metric)
            if not points:
                continue
            any_data = True
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            max_reps = max(max_reps, max(xs))
            line_color = model_color(base_color, model_type)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=model_type,
                    line=dict(color=line_color, dash=dash_map[model_type], width=2),
                    marker=dict(size=8, color=line_color),
                )
            )
        title = f"{data_tag} — {scale}: {metric} vs. repetition (baseline)"
        if not any_data:
            title += "  [no finished baseline runs]"
        tickvals, ticktext = pow2_ticks(max_reps if any_data else None)
        fig.update_layout(
            title=title,
            xaxis_title="repetition count",
            yaxis_title=metric,
            xaxis_type="log",
            xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticktext),
            template="plotly_white",
            width=450,
            height=450,
        )
        return fig

    return collect_baseline_points, make_repetition_figure


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_olmo_80m = metric_multiselect("olmo_mix 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/pile-validation/CE loss"])
    sel_olmo_80m
    return (sel_olmo_80m,)


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, render_side_by_side, sel_olmo_80m):
    _out = render_side_by_side(
        sel_olmo_80m.value,
        lambda m: make_repetition_figure(finished_df, "olmo_mix", "80M", m),
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_olmo_200m = metric_multiselect("olmo_mix 200M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/pile-validation/CE loss"])
    sel_olmo_200m
    return (sel_olmo_200m,)


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, render_side_by_side, sel_olmo_200m):
    _out = render_side_by_side(
        sel_olmo_200m.value,
        lambda m: make_repetition_figure(finished_df, "olmo_mix", "200M", m),
    )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Group 2 — Single-source mixtures (`dclm`, `starcoder`, `pes2o`, `wiki`)
    """)
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_dclm_80m = metric_multiselect("dclm 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/pile-validation/CE loss"])
    sel_dclm_80m
    return (sel_dclm_80m,)


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, render_side_by_side, sel_dclm_80m):
    _out = render_side_by_side(
        sel_dclm_80m.value,
        lambda m: make_repetition_figure(finished_df, "dclm", "80M", m),
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_starcoder_80m = metric_multiselect("starcoder 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_stack-validation/CE loss"])
    sel_starcoder_80m
    return (sel_starcoder_80m,)


@app.cell(hide_code=True)
def _(
    finished_df,
    make_repetition_figure,
    render_side_by_side,
    sel_starcoder_80m,
):
    _out = render_side_by_side(
        sel_starcoder_80m.value,
        lambda m: make_repetition_figure(finished_df, "starcoder", "80M", m),
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_pes2o_80m = metric_multiselect("pes2o 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_pes2o-validation/CE loss"])
    sel_pes2o_80m
    return (sel_pes2o_80m,)


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, render_side_by_side, sel_pes2o_80m):
    _out = render_side_by_side(
        sel_pes2o_80m.value,
        lambda m: make_repetition_figure(finished_df, "pes2o", "80M", m),
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_wiki_80m = metric_multiselect("wiki 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_wiki-validation/CE loss"])
    sel_wiki_80m
    return (sel_wiki_80m,)


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, render_side_by_side, sel_wiki_80m):
    _out = render_side_by_side(
        sel_wiki_80m.value,
        lambda m: make_repetition_figure(finished_df, "wiki", "80M", m),
    )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group 3 — Regularizer sweeps: dropout & weight decay

    Model type is encoded as **line style** (solid = dense, dotted = moe32,
    dashed = moe64). Baseline vs. each regularizer value is encoded as **color**.
    X axis is the repetition count (log); Y is the chosen metric. Runs are restricted
    to the `olmo_mix` mixture at 80M scale (where the regularizer sweeps live). Note
    the dropout/weight-decay sweeps currently only cover dense and moe64.
    """)
    return


@app.cell(hide_code=True)
def _(
    get_model_type,
    get_reps,
    get_scale,
    go,
    has_tag,
    model_color,
    pd,
    pow2_ticks,
):
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
        """model type -> line dash + shade; baseline vs each factor value -> base color.

        Within a factor-value color, model type is distinguished by both dash style
        and shade (dense darkest, moe64 lightest).

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

        # Base color per factor bucket; model type -> dash + shade of that color.
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
            line_color = model_color(color_map[fval_key], model_type)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=f"{model_type}, {label_val}",
                    line=dict(color=line_color, dash=dash_map[model_type], width=2),
                    marker=dict(size=7, color=line_color),
                )
            )
        # Power-of-2 ticks based on the largest repetition count present.
        _all_reps = [r for pts in series.values() for r in pts]
        _max_reps = max(_all_reps) if _all_reps else None
        _tickvals, _ticktext = pow2_ticks(_max_reps)
        fig.update_layout(
            title=f"{factor_label} — {metric} (olmo_mix, 80M)\n"
            f"solid=dense / dotted=moe32 / dashed=moe64",
            xaxis_title="repetition count",
            yaxis_title=metric,
            xaxis_type="log",
            xaxis=dict(tickmode="array", tickvals=_tickvals, ticktext=_ticktext),
            template="plotly_white",
            width=700,
            height=450,
        )
        return fig

    return (make_factor_figure,)


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_dropout = metric_multiselect("dropout — metrics", chosen_values=["train/CE loss", "eval/lm/pile-validation/CE loss"])
    sel_dropout
    return (sel_dropout,)


@app.cell(hide_code=True)
def _(
    DROPOUT_COL,
    finished_df,
    make_factor_figure,
    render_side_by_side,
    sel_dropout,
):
    # Dropout: baseline = no dropout set (blank).
    _out = render_side_by_side(
        sel_dropout.value,
        lambda m: make_factor_figure(finished_df, DROPOUT_COL, "dropout", None, m),
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_weight_decay = metric_multiselect("weight decay — metrics", chosen_values=["train/CE loss", "eval/lm/pile-validation/CE loss"])
    sel_weight_decay
    return (sel_weight_decay,)


@app.cell(hide_code=True)
def _(
    WD_COL,
    finished_df,
    make_factor_figure,
    render_side_by_side,
    sel_weight_decay,
):
    # Weight decay: baseline = the default 0.1; swept values are 0.2 / 0.4.
    _out = render_side_by_side(
        sel_weight_decay.value,
        lambda m: make_factor_figure(finished_df, WD_COL, "weight_decay", 0.1, m),
    )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group 4 — `famA` runs: metric vs. %dclm

    Every finished run whose name contains `famA`. The X axis is the DCLM
    percentage parsed from `dclm{N}` in the run name (0, 25, 50, 75, 100);
    Y is the chosen metric. Lines are colored by model type.
    """)
    return


@app.cell(hide_code=True)
def _(get_model_type, go, pd, re):
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
            width=600,
            height=450,
        )
        return fig

    return (make_famA_figure,)


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_famA = metric_multiselect("famA — metrics", chosen_values=["train/CE loss", "eval/lm/pile-validation/CE loss"])
    sel_famA
    return (sel_famA,)


@app.cell(hide_code=True)
def _(finished_df, make_famA_figure, render_side_by_side, sel_famA):
    _out = render_side_by_side(
        sel_famA.value, lambda m: make_famA_figure(finished_df, m)
    )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group 5 — `famB` runs: metric vs. repetition

    `famB` runs mix a single source into `olmo_mix` at a fixed fraction, encoded in
    the run name (`sc50`/`sc90` = 50%/90% starcoder, `p2o50`/`p2o90` = 50%/90% pes2o).

    Each figure plots the chosen metric vs. repetition count (log X, power-of-2 ticks)
    with:

    - **line color** = data source (`olmo_mix` baseline, single-source baseline, and the
      two famB mixes), and
    - **line style** = model type (solid = dense, dotted = moe32, dashed = moe64).

    The `olmo_mix` and single-source lines are **baseline** runs (tag `baseline`);
    duplicate baseline runs at a rep count keep the lowest value (see Group 1 & 2).
    """)
    return


@app.cell(hide_code=True)
def _(
    collect_baseline_points,
    get_model_type,
    get_reps,
    go,
    model_color,
    pd,
    pow2_ticks,
):
    def collect_famB_points(df, name_substr, model_type, metric):
        """Sorted (reps, value) points for famB runs matching a name substring."""
        by_reps: dict = {}
        for _, row in df.iterrows():
            name = str(row.get("Name", ""))
            if "famB" not in name or name_substr not in name:
                continue
            if get_model_type(row) != model_type:
                continue
            reps = get_reps(row)
            loss = row.get(metric)
            if pd.isna(reps) or pd.isna(loss):
                continue
            prev = by_reps.get(reps)
            by_reps[reps] = min(float(loss), prev) if prev is not None else float(loss)
        return [(r, by_reps[r]) for r in sorted(by_reps)]

    def make_famB_figure(df, metric, title, sources):
        """One famB figure.

        `sources` is an ordered list of (label, color, kind, key) tuples where kind is
        either "baseline" (key = data tag, uses baseline runs at 80M) or "famB"
        (key = run-name substring).
        """
        dash_map = {"dense": "solid", "moe32": "dot", "moe64": "dash"}
        fig = go.Figure()
        max_reps = 0
        any_data = False
        for label, color, kind, key in sources:
            for model_type in ["dense", "moe32", "moe64"]:
                if kind == "baseline":
                    points = collect_baseline_points(df, key, "80M", model_type, metric)
                else:
                    points = collect_famB_points(df, key, model_type, metric)
                if not points:
                    continue
                any_data = True
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                max_reps = max(max_reps, max(xs))
                line_color = model_color(color, model_type)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        name=f"{label}, {model_type}",
                        legendgroup=label,
                        line=dict(color=line_color, dash=dash_map[model_type], width=2),
                        marker=dict(size=7, color=line_color),
                    )
                )
        tickvals, ticktext = pow2_ticks(max_reps if any_data else None)
        fig.update_layout(
            title=title,
            xaxis_title="repetition count",
            yaxis_title=metric,
            xaxis_type="log",
            xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticktext),
            template="plotly_white",
            width=700,
            height=550,
        )
        return fig

    return (make_famB_figure,)


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_famB_sc = metric_multiselect("famB starcoder — metrics", chosen_values=["eval/lm/pile-validation/CE loss", "eval/lm/dolma_stack-validation/CE loss"])
    sel_famB_sc
    return (sel_famB_sc,)


@app.cell(hide_code=True)
def _(finished_df, make_famB_figure, render_side_by_side, sel_famB_sc):
    # Plot 1: starcoder family. Color = source; dash = model type.
    _sources = [
        ("olmo_mix", "#1f77b4", "baseline", "olmo_mix"),
        ("starcoder", "#2ca02c", "baseline", "starcoder"),
        ("sc50", "#ff7f0e", "famB", "sc50"),
        ("sc90", "#d62728", "famB", "sc90"),
    ]
    _out = render_side_by_side(
        sel_famB_sc.value,
        lambda m: make_famB_figure(
            finished_df, m, f"famB (starcoder): {m} vs. repetition", _sources
        ),
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_famB_p2o = metric_multiselect("famB pes2o — metrics", chosen_values=["eval/lm/pile-validation/CE loss", "eval/lm/dolma_pes2o-validation/CE loss"])
    sel_famB_p2o
    return (sel_famB_p2o,)


@app.cell(hide_code=True)
def _(finished_df, make_famB_figure, render_side_by_side, sel_famB_p2o):
    # Plot 2: pes2o family. Color = source; dash = model type.
    _sources = [
        ("olmo_mix", "#1f77b4", "baseline", "olmo_mix"),
        ("pes2o", "#2ca02c", "baseline", "pes2o"),
        ("p2o50", "#ff7f0e", "famB", "p2o50"),
        ("p2o90", "#d62728", "famB", "p2o90"),
    ]
    _out = render_side_by_side(
        sel_famB_p2o.value,
        lambda m: make_famB_figure(
            finished_df, m, f"famB (pes2o): {m} vs. repetition", _sources
        ),
    )
    _out
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
