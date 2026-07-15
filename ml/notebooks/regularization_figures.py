import marimo

__generated_with = "0.23.9"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    # Regularization & Data-Repetition Figures

    Figures built from the most recent Weights & Biases CSV export. The notebook
    reads the automated export in `ml/runs_data/` (written by
    `ml/wandb_download/download.py`) if present, falling back to the
    hand-downloaded CSVs in `ml/notebooks/data/` otherwise.

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
    3. Regularizer sweeps (**dropout**, **weight decay**, **max grad norm**) —
       color = model type, line style = regularizer value; loss vs. repetition (log X).
    3b. MoE-only regularizer sweeps (**jitter_eps**, **EOM**, **FOM**) — same encoding,
       Dense excluded.
    4. All `famA` runs: loss vs. `%dclm` (0/25/50/75/100 from the run name).
    5. `famB` runs: single-source mixes into `olmo_mix` vs. repetition.
    """)
    return


@app.cell(hide_code=True)
def _():
    import glob
    import json
    import os
    import re

    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend: no GUI/browser needed for PDF export
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    return glob, json, os, pd, plt, re


@app.cell(hide_code=True)
def _(glob, os, pd):
    def find_latest_data_file() -> str:
        """Return the path to the most recently modified CSV to plot.

        Prefers the automated export written by ``ml/wandb_download/download.py``
        into ``ml/runs_data/``; falls back to the hand-downloaded W&B CSVs in
        ``ml/notebooks/data/`` only when ``ml/runs_data/`` has no CSV yet.
        """
        _this_dir = (
            os.path.dirname(os.path.abspath(__file__))
            if "__file__" in dir()
            else os.getcwd()
        )
        # Handle being run from the repo root or from ml/notebooks/. The primary
        # location (ml/runs_data) is tried first; the legacy location is the
        # fallback. First directory that actually contains a CSV wins.
        _dir_candidates = [
            # ml/runs_data relative to ml/notebooks/ (this file's dir).
            os.path.join(_this_dir, "..", "runs_data"),
            # ml/runs_data relative to the repo root.
            os.path.join(_this_dir, "ml", "runs_data"),
            # Legacy hand-download location(s).
            os.path.join(_this_dir, "data"),
            os.path.join(_this_dir, "ml", "notebooks", "data"),
        ]
        _searched = []
        for _d in _dir_candidates:
            _d = os.path.normpath(_d)
            _searched.append(_d)
            if not os.path.isdir(_d):
                continue
            _csvs = glob.glob(os.path.join(_d, "*.csv"))
            if _csvs:
                return max(_csvs, key=os.path.getmtime)
        raise FileNotFoundError(
            "No CSV files found in any data directory: " + ", ".join(_searched)
        )

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
def _(json, pd):
    # ------------------------------------------------------------------
    # Data filtering + parsing helpers.
    # ------------------------------------------------------------------

    # Column names in the wandb export.
    LOSS_COL = "train/CE loss"
    DROPOUT_COL = "model.block.dropout"
    WD_COL = "train_module.optim.weight_decay"
    MGN_COL = "train_module.max_grad_norm"
    FOM_COL = "model.block.feed_forward_moe.fom_prob"
    ROUTERS_COL = "model.block.feed_forward_moe.routers_list"
    REPS_COL = "data_reps"

    def router_field(row, field):
        """Read a field (e.g. 'jitter_eps', 'eom_prob') from the first router in the
        JSON-encoded ``routers_list`` column. Returns None if absent/unparseable."""
        rl = row.get(ROUTERS_COL)
        if not isinstance(rl, str):
            return None
        try:
            routers = json.loads(rl)
        except (ValueError, TypeError):
            return None
        if not routers:
            return None
        return routers[0].get(field)

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

    MAX_DUR_COL = "trainer.max_duration.value"
    REQ_TOKENS_COL = "dataset.source_mixture_config.requested_tokens"

    def get_reps(row) -> float:
        """Repetition count = total training tokens / unique tokens, i.e.
        ``trainer.max_duration.value / dataset.source_mixture_config.requested_tokens``.

        An early (pre-2026-04-10) bug wrote wrong repetition counts into run *ids*
        and the ``data_reps`` column, so neither of those can be trusted. The wide
        CSV's ``Name`` column, however, is the *corrected* run name, and its
        ``requested_tokens`` / ``max_duration`` columns are correct — so we compute
        the ratio straight from them. A run with no source-mixture requested-token
        budget is single-pass, i.e. rep 1.
        """
        md = row.get(MAX_DUR_COL)
        rt = row.get(REQ_TOKENS_COL)
        try:
            if pd.notna(rt) and float(rt) > 0 and pd.notna(md):
                return float(md) / float(rt)
        except (TypeError, ValueError):
            pass
        # No source-mixture requested-token budget -> single pass over the data.
        return 1.0

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
        """Shade a hex color. Positive `lighten` blends toward white, negative blends
        toward black; 0 leaves it unchanged. Values are clamped to [-1, 1]."""
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        t = max(-1.0, min(1.0, lighten))
        if t >= 0:
            r, g, b = (round(c + (255 - c) * t) for c in (r, g, b))
        else:
            r, g, b = (round(c * (1 + t)) for c in (r, g, b))
        return f"#{r:02x}{g:02x}{b:02x}"

    def model_color(base_hex: str, model_type: str) -> str:
        """Shade of `base_hex` for a model type (dense darkest, moe64 lightest)."""
        return shade_color(base_hex, MODEL_SHADE.get(model_type, 0.0))

    def keep_arch(model_type, include_archs=None, exclude_archs=None) -> bool:
        """Whether a model type passes the arch include/exclude filters.

        `include_archs=None` includes all; if a list is given, only those archs are
        kept. `exclude_archs` removes archs. Both may be combined.
        """
        if include_archs is not None and model_type not in include_archs:
            return False
        if exclude_archs and model_type in exclude_archs:
            return False
        return True

    def filter_rep_points(points, include_reps=None, exclude_reps=None):
        """Filter a list of (reps, value) points by rep include/exclude sets.

        `include_reps=None` includes all reps; if a list is given, only those rep
        counts are kept. `exclude_reps` removes rep counts. Values are compared as
        floats so e.g. 8 and 8.0 match.
        """
        inc = None if include_reps is None else {float(r) for r in include_reps}
        exc = set() if not exclude_reps else {float(r) for r in exclude_reps}
        out = []
        for reps, val in points:
            if inc is not None and float(reps) not in inc:
                continue
            if float(reps) in exc:
                continue
            out.append((reps, val))
        return out

    def _arch_of(row) -> str:
        """Architecture (dense/moe32/moe64) from the run-name suffix, falling back to
        tags (matches the famA handling; famAr runs are only tagged 'MoE')."""
        import re as _re

        m = _re.search(r"_(dense|moe32|moe64)$", str(row.get("Name", "")))
        if m:
            return m.group(1)
        return get_model_type(row)

    def _norm(v):
        """Normalize a scalar for hashing: NaN/blank -> None so equal configs match
        (NaN != NaN in Python would otherwise break dedup)."""
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    def _sources_key(row):
        """Canonical, order-stable key for the source-mixture list. Compares each
        source dict field by field, ignoring the derived 'unique_tokens' field that
        is present on only some runs."""
        s = row.get("dataset.source_mixture_config.source_list.sources")
        if not isinstance(s, str):
            return None
        try:
            sources = json.loads(s)
        except (ValueError, TypeError):
            return None
        canon = []
        for d in sources:
            items = []
            for k in sorted(d):
                if k == "unique_tokens":
                    continue
                val = d[k]
                items.append((k, tuple(val) if isinstance(val, list) else val))
            canon.append(tuple(items))
        return tuple(canon)

    def _namemix_key(row):
        """Fallback data-mix signature, used only when the run has no
        ``source_mixture_config`` (e.g. pure single-source mixes and plain-datamix
        runs record no sources). Combines the data-source tag (olmo_mix / dclm /
        starcoder / pes2o / wiki) with any ``dclm{N}`` percentage from the name, so
        e.g. wiki vs olmo_mix, and dclm0 vs dclm100, aren't merged just because they
        all lack a sources config."""
        if _sources_key(row) is not None:
            return None
        import re as _re

        m = _re.search(r"dclm(\d+)", str(row.get("Name", "")))
        return (get_data_tag(row), int(m.group(1)) if m else None)

    def _dedup_key(row):
        """Identity key for de-duplicating runs that are the same experiment."""
        return (
            _sources_key(row),
            _namemix_key(row),
            _norm(row.get("dataset.source_mixture_config.requested_tokens")),
            _norm(row.get("model.d_model")),
            _norm(row.get(DROPOUT_COL)),
            _norm(row.get(WD_COL)),
            _norm(row.get(MGN_COL)),
            _norm(row.get("train_module.optim.lr")),
            _norm(router_field(row, "jitter_eps")),
            _norm(router_field(row, "eom_prob")),
            _norm(row.get(FOM_COL)),
            _arch_of(row),
        )

    def dedupe_runs(df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate runs that share the same experiment identity (source
        mixture + requested tokens + model size + dropout + weight decay + MoE
        jitter/EOM/FOM + architecture). Keeps the first occurrence and warns."""
        seen: dict = {}
        keep_idx = []
        for idx, row in df.iterrows():
            key = _dedup_key(row)
            if key in seen:
                print(
                    f"DEDUP: dropping '{row.get('Name')}' — duplicate of "
                    f"'{seen[key]}'"
                )
                continue
            seen[key] = row.get("Name")
            keep_idx.append(idx)
        return df.loc[keep_idx].copy()

    return (
        DROPOUT_COL,
        FOM_COL,
        MGN_COL,
        WD_COL,
        dedupe_runs,
        filter_finished,
        filter_rep_points,
        get_model_type,
        get_reps,
        get_scale,
        has_tag,
        keep_arch,
        pow2_ticks,
        router_field,
        shade_color,
    )


@app.cell
def _(dedupe_runs, filter_finished, raw_df):
    # Single filtered frame reused by every figure below. Duplicate runs (same
    # experiment identity) are dropped up front so no plot double-counts them.
    finished_df = dedupe_runs(filter_finished(raw_df))
    return (finished_df,)


@app.cell(hide_code=True)
def _(json, os):
    # ------------------------------------------------------------------
    # Per-step training history (for the training-curve column).
    #
    # ml/wandb_download/download.py --include-history writes one JSONL per run to
    # ml/runs_data/history/<run Name>.jsonl, each line a logged step with a "_step"
    # field plus whatever metrics were logged that step (train/CE loss is dense;
    # eval metrics are sparse). We load a run's (step, value) series on demand.
    # ------------------------------------------------------------------
    def _history_dir():
        _this_dir = (
            os.path.dirname(os.path.abspath(__file__))
            if "__file__" in dir()
            else os.getcwd()
        )
        _candidates = [
            os.path.join(_this_dir, "..", "runs_data", "history"),
            os.path.join(_this_dir, "ml", "runs_data", "history"),
        ]
        for d in _candidates:
            if os.path.isdir(os.path.normpath(d)):
                return os.path.normpath(d)
        return os.path.normpath(_candidates[0])

    HISTORY_DIR = _history_dir()

    _history_cache: dict = {}

    def load_history_series(run_name: str, metric: str):
        """Return sorted [(step, value), ...] for `metric` in run `run_name`'s history.

        Reads ml/runs_data/history/<run_name>.jsonl, keeping only rows where both
        ``_step`` and the metric are present. Results are cached per (run, metric).
        Returns [] if the history file is missing.
        """
        cache_key = (run_name, metric)
        if cache_key in _history_cache:
            return _history_cache[cache_key]
        path = os.path.join(HISTORY_DIR, f"{run_name}.jsonl")
        series = []
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    step = row.get("_step")
                    val = row.get(metric)
                    if step is None or val is None:
                        continue
                    try:
                        series.append((float(step), float(val)))
                    except (TypeError, ValueError):
                        continue
            series.sort()
        _history_cache[cache_key] = series
        return series

    # Metric that gets smoothed for the column-1 (final-value) plots, and how the
    # smoothed axis is labeled.
    SMOOTHED_METRIC = "train/CE loss"
    SMOOTH_WINDOW = 100
    SMOOTHED_LABEL = f"{SMOOTHED_METRIC} (last-{SMOOTH_WINDOW}-step avg)"

    def smoothed_train_loss(run_name: str, window: int = SMOOTH_WINDOW):
        """Mean of ``train/CE loss`` over the last `window` logged steps of a run.

        train/CE loss is logged every step, so the last `window` logged points are
        the last `window` steps. Returns None if the run has no usable history.
        """
        series = load_history_series(run_name, SMOOTHED_METRIC)
        if not series:
            return None
        tail = series[-window:]
        if not tail:
            return None
        return sum(v for _, v in tail) / len(tail)

    def metric_value(row, metric):
        """Value to plot for `metric` in the column-1 (final-value) figures.

        For ``train/CE loss`` this is the mean over the last SMOOTH_WINDOW steps from
        the run's history; every other metric uses its raw final value from the CSV.
        Falls back to the raw CSV value if a run has no history.
        """
        if metric == SMOOTHED_METRIC:
            sm = smoothed_train_loss(str(row.get("Name", "")))
            if sm is not None:
                return sm
        return row.get(metric)

    def axis_label(metric):
        """Y-axis / legend label for a metric (annotates the smoothed train loss)."""
        return SMOOTHED_LABEL if metric == SMOOTHED_METRIC else metric

    return axis_label, load_history_series, metric_value


@app.cell(hide_code=True)
def _(os):
    # ------------------------------------------------------------------
    # PDF export: every figure is written to ml/figures/ when generated.
    # ------------------------------------------------------------------
    def _figures_dir():
        _this_dir = (
            os.path.dirname(os.path.abspath(__file__))
            if "__file__" in dir()
            else os.getcwd()
        )
        _candidates = [
            os.path.join(_this_dir, "..", "figures"),  # from ml/notebooks/
            os.path.join(_this_dir, "ml", "figures"),  # from repo root
        ]
        for d in _candidates:
            parent = os.path.dirname(os.path.normpath(d))
            if os.path.isdir(parent):
                return os.path.normpath(d)
        return os.path.normpath(_candidates[0])

    FIGURES_DIR = _figures_dir()

    def _sanitize(name: str) -> str:
        keep = []
        for ch in name:
            keep.append(ch if (ch.isalnum() or ch in "-_.") else "_")
        return "".join(keep).strip("_")

    def save_pdf(fig, plot_key: str, metric: str):
        """Write matplotlib `fig` to ml/figures/<plot_key>__<metric>.pdf.

        matplotlib's PDF backend is pure-Python (no browser/kaleido needed), so this
        is reliable. Failures are caught so the notebook never crashes on export.
        """
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fname = f"{_sanitize(plot_key)}__{_sanitize(metric)}.pdf"
        path = os.path.join(FIGURES_DIR, fname)
        try:
            fig.savefig(path, bbox_inches="tight")
        except Exception as e:  # pragma: no cover - env dependent
            print(f"NOTE: could not save PDF '{fname}': {type(e).__name__}: {e}")
        return fig

    return (save_pdf,)


@app.cell(hide_code=True)
def _(mo, raw_df, save_pdf):
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

    def render_side_by_side(metrics, build_fig, plot_key):
        """Render one figure per selected metric, laid out side by side.

        `build_fig(metric)` should return a matplotlib figure for a single metric.
        Each figure is also saved to ml/figures/ as `<plot_key>__<metric>.pdf`.
        """
        if not metrics:
            return mo.md("*Select at least one metric.*")
        figs = []
        for m in metrics:
            fig = build_fig(m)
            save_pdf(fig, plot_key, m)
            figs.append(fig)
        return mo.hstack(figs, widths="equal", gap=1, wrap=True)

    return metric_multiselect, render_side_by_side


@app.cell(column=1, hide_code=True)
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
    axis_label,
    filter_rep_points,
    get_model_type,
    get_reps,
    get_scale,
    has_tag,
    keep_arch,
    metric_value,
    pd,
    plt,
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
            loss = metric_value(row, metric)
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

    def apply_pow2_xaxis(ax, max_reps, any_data):
        """Log2 x-axis with power-of-2 tick labels."""
        tickvals, ticktext = pow2_ticks(max_reps if any_data else None)
        ax.set_xscale("log", base=2)
        ax.set_xticks(tickvals)
        ax.set_xticklabels(ticktext)
        ax.minorticks_off()

    def make_repetition_figure(
        df: pd.DataFrame,
        data_tag: str,
        scale: str,
        metric: str,
        include_archs=None,
        exclude_archs=None,
        include_reps=None,
        exclude_reps=None,
    ):
        """One figure: Dense + MoE32 + MoE64 baseline, metric vs reps (log X).

        Model type is distinguished by both line style (dense solid, moe32 dotted,
        moe64 dashed) and a distinct color.

        Filters (all default to no-op):

        - ``include_archs`` / ``exclude_archs``: keep/drop architectures
          (``"dense"``, ``"moe32"``, ``"moe64"``).
        - ``include_reps`` / ``exclude_reps``: keep/drop repetition counts.
        """
        fig, ax = plt.subplots(figsize=(5.0, 4.5))
        color_map = {"dense": "#1f77b4", "moe32": "#2ca02c", "moe64": "#d62728"}
        dash_map = {"dense": "-", "moe32": ":", "moe64": "--"}
        any_data = False
        max_reps = 0
        for model_type in ["dense", "moe32", "moe64"]:
            if not keep_arch(model_type, include_archs, exclude_archs):
                continue
            points = collect_baseline_points(df, data_tag, scale, model_type, metric)
            points = filter_rep_points(points, include_reps, exclude_reps)
            if not points:
                continue
            any_data = True
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            max_reps = max(max_reps, max(xs))
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=4,
                linewidth=1.2,
                linestyle=dash_map[model_type],
                color=color_map[model_type],
                label=model_type,
            )
        title = f"{data_tag} — {scale}\n{metric} vs. repetition (baseline)"
        if not any_data:
            title += "  [no finished baseline runs]"
        apply_pow2_xaxis(ax, max_reps, any_data)
        ax.set_xlabel("repetition count")
        ax.set_ylabel(axis_label(metric))
        ax.set_title(title, fontsize=10)
        if any_data:
            # plt.rcParams['legend.numpoints'] = 1
            # ax.legend(fontsize=8)
            ax.legend(fontsize=8, markerscale=0.5)
        fig.tight_layout()
        return fig

    return apply_pow2_xaxis, collect_baseline_points, make_repetition_figure


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_olmo_80m = metric_multiselect("olmo_mix 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_olmo_80m
    return (sel_olmo_80m,)


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, render_side_by_side, sel_olmo_80m):
    _out = render_side_by_side(
        sel_olmo_80m.value,
        lambda m: make_repetition_figure(finished_df, "olmo_mix", "80M", m),
        "olmo_mix_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_olmo_200m = metric_multiselect("olmo_mix 200M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_olmo_200m
    return (sel_olmo_200m,)


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, render_side_by_side, sel_olmo_200m):
    _out = render_side_by_side(
        sel_olmo_200m.value,
        lambda m: make_repetition_figure(finished_df, "olmo_mix", "200M", m),
        "olmo_mix_200M",
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
    sel_dclm_80m = metric_multiselect("dclm 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_dclm_80m
    return (sel_dclm_80m,)


@app.cell(hide_code=True)
def _(finished_df, make_repetition_figure, render_side_by_side, sel_dclm_80m):
    _out = render_side_by_side(
        sel_dclm_80m.value,
        lambda m: make_repetition_figure(finished_df, "dclm", "80M", m),
        "dclm_80M",
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
        "starcoder_80M",
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
        "pes2o_80M",
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
        "wiki_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group 3 — Regularizer sweeps: dropout, weight decay & max grad norm

    Model type is encoded as **color** (blue = dense, green = moe32, red = moe64).
    Baseline vs. each regularizer value is encoded as **line style**.
    X axis is the repetition count (log); Y is the chosen metric. Runs are restricted
    to the `olmo_mix` mixture at 80M scale (where the regularizer sweeps live). Note
    the sweeps currently only cover dense and moe64. For max grad norm, the baseline
    is the default `1.0`; a `null` (no clipping) setting is shown as its own series.
    """)
    return


@app.cell(hide_code=True)
def _(
    apply_pow2_xaxis,
    axis_label,
    filter_rep_points,
    get_model_type,
    get_reps,
    get_scale,
    has_tag,
    keep_arch,
    metric_value,
    pd,
    plt,
    shade_color,
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

    def make_factor_figure(
        df,
        factor_col,
        factor_label,
        baseline_value,
        metric,
        include_archs=None,
        exclude_archs=None,
        include_reps=None,
        exclude_reps=None,
        nan_label=None,
        value_getter=None,
    ):
        """color -> model type; line style -> baseline vs each factor value.

        `baseline_value` is the value that represents 'no regularization' for this
        column (e.g. dropout baseline = None/blank, weight-decay baseline = 0.1).

        `nan_label`: when set, runs whose factor value is NaN/blank are bucketed under
        this label (a real setting, e.g. max_grad_norm=null -> "no clip") instead of
        being merged into "baseline". When None (the default), NaN counts as baseline.

        `value_getter`: optional ``callable(row) -> raw value`` used to read the factor
        value from a row. Defaults to ``row.get(factor_col)``. Use this for values
        nested inside JSON columns (e.g. MoE ``jitter_eps`` / ``eom_prob`` inside
        ``routers_list``).

        Filters (all default to no-op):

        - ``include_archs`` / ``exclude_archs``: keep/drop architectures.
        - ``include_reps`` / ``exclude_reps``: keep/drop repetition counts.
        """
        if value_getter is None:
            value_getter = lambda row: row.get(factor_col)

        # Gather points keyed by (model_type, factor_value) -> {reps: min_value}.
        series: dict = {}
        factor_values = set()
        extra_buckets: list = []  # non-numeric buckets (e.g. nan_label), in first-seen order
        for _, row in df.iterrows():
            if get_scale(row) != "80M":
                continue
            if not has_tag(row, "olmo_mix"):
                continue
            model_type = get_model_type(row)
            if model_type not in ("dense", "moe32", "moe64"):
                continue
            if not keep_arch(model_type, include_archs, exclude_archs):
                continue
            reps = get_reps(row)
            loss = metric_value(row, metric)
            if pd.isna(reps) or pd.isna(loss):
                continue

            raw_val = value_getter(row)
            is_blank = pd.isna(raw_val) or str(raw_val).strip() == ""
            try:
                fval = float(raw_val) if not is_blank else None
            except (TypeError, ValueError):
                fval = None
            # Bucket the run.
            if is_blank and nan_label is not None:
                fval_key = nan_label
                if nan_label not in extra_buckets:
                    extra_buckets.append(nan_label)
            elif fval is None or (baseline_value is not None and fval == baseline_value):
                fval_key = "baseline"
            else:
                fval_key = fval
                factor_values.add(fval)

            key = (model_type, fval_key)
            series.setdefault(key, {})
            prev = series[key].get(reps)
            series[key][reps] = min(loss, prev) if prev is not None else float(loss)

        # Color encodes the model type; line style encodes the factor value.
        # Within a model-type color, the factor value also sets the shade: the first
        # bucket is the DARKEST + solid line, later buckets are lighter with other
        # dashes. `extra_buckets` (e.g. max_grad_norm "no clip") come first so they
        # get the solid line.
        ordered_buckets = extra_buckets + ["baseline"] + sorted(factor_values)
        color_map = {"dense": "#1f77b4", "moe32": "#2ca02c", "moe64": "#d62728"}
        dash_cycle = ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]
        factor_dash = {
            b: dash_cycle[i % len(dash_cycle)] for i, b in enumerate(ordered_buckets)
        }
        # Shade level per bucket: baseline is the DARKEST (blended toward black),
        # higher factor values get progressively lighter up to LIGHTEST. Spread
        # evenly; with a single bucket everything sits at DARKEST.
        _DARKEST = -0.45
        _LIGHTEST = 0.4
        _n = len(ordered_buckets)
        bucket_shade = {
            b: (_DARKEST + (_LIGHTEST - _DARKEST) * i / (_n - 1) if _n > 1 else _DARKEST)
            for i, b in enumerate(ordered_buckets)
        }

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        any_data = False
        max_reps = 0
        # The factor value the baseline actually takes (dropout/jitter/eom/fom -> 0.0,
        # weight_decay -> 0.1, max_grad_norm -> 1.0). Shown as its numeric value.
        baseline_display = baseline_value if baseline_value is not None else 0.0

        def _bucket_label(fval_key):
            if fval_key == "baseline":
                lbl = f"{baseline_display:.10g}"
                return lbl if "." in lbl else f"{baseline_display:.1f}"
            if isinstance(fval_key, str):
                return fval_key
            return f"{fval_key:g}"

        shown_models = []  # model types actually plotted, in canonical order
        shown_buckets = []  # factor buckets actually plotted, in ordered_buckets order
        for (model_type, fval_key), pts in sorted(
            series.items(), key=lambda kv: (str(kv[0][1]), kv[0][0])
        ):
            points = filter_rep_points(sorted(pts.items()), include_reps, exclude_reps)
            if not points:
                continue
            any_data = True
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            max_reps = max(max_reps, max(xs))
            if model_type not in shown_models:
                shown_models.append(model_type)
            if fval_key not in shown_buckets:
                shown_buckets.append(fval_key)
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=3,
                linewidth=1.2,
                linestyle=factor_dash[fval_key],
                color=shade_color(color_map[model_type], bucket_shade[fval_key]),
            )
        apply_pow2_xaxis(ax, max_reps, any_data)
        ax.set_xlabel("repetition count")
        ax.set_ylabel(axis_label(metric))
        ax.set_title(f"{factor_label} — {metric} (olmo_mix, 80M)", fontsize=10)
        if any_data:
            from matplotlib.lines import Line2D

            def _header():
                # Invisible handle used to render a section-title row in the legend.
                return Line2D([0], [0], linestyle="none", marker="", label="")

            # Section 1: color -> model type (solid lines in each model's color).
            model_order = [m for m in ["dense", "moe32", "moe64"] if m in shown_models]
            model_handles = [
                Line2D([0], [0], color=color_map[m], linewidth=2, linestyle="-")
                for m in model_order
            ]
            # Section 2: line style -> factor value. Shade the key from black (first
            # bucket, solid) to light grey (last bucket), mirroring the data lines.
            bucket_order = [b for b in ordered_buckets if b in shown_buckets]
            _nb = len(bucket_order)
            _grey_top = 0.7  # lightest grey level for the last bucket
            bucket_handles = [
                Line2D(
                    [0],
                    [0],
                    color=shade_color("#000000", _grey_top * i / (_nb - 1) if _nb > 1 else 0.0),
                    linewidth=1.5,
                    linestyle=factor_dash[b],
                )
                for i, b in enumerate(bucket_order)
            ]
            # Combine both sections into one legend box with bold section headers.
            handles = (
                [_header()]
                + model_handles
                + [_header()]
                + bucket_handles
            )
            labels = (
                ["Model type"]
                + model_order
                + [factor_label]
                + [_bucket_label(b) for b in bucket_order]
            )
            legend = ax.legend(
                handles,
                labels,
                fontsize=7,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                handlelength=2.2,
            )
            # Bold + left-align the section-header rows.
            for text in legend.get_texts():
                if text.get_text() in ("Model type", factor_label):
                    text.set_fontweight("bold")
        fig.tight_layout()
        return fig

    return (make_factor_figure,)


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_dropout = metric_multiselect("dropout — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
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
        lambda m: make_factor_figure(finished_df, DROPOUT_COL, "dropout", None, m, exclude_reps=[128,256]),
        "dropout",
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_weight_decay = metric_multiselect("weight decay — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
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
        lambda m: make_factor_figure(finished_df, WD_COL, "weight_decay", 0.1, m, exclude_reps=[128,256]),
        "weight_decay",
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_max_grad_norm = metric_multiselect("max grad norm — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_max_grad_norm
    return (sel_max_grad_norm,)


@app.cell(hide_code=True)
def _(
    MGN_COL,
    finished_df,
    make_factor_figure,
    render_side_by_side,
    sel_max_grad_norm,
):
    # Max grad norm: baseline = the default 1.0; swept value is 0.2; a `null` (no
    # clipping) setting is bucketed separately via nan_label.
    _out = render_side_by_side(
        sel_max_grad_norm.value,
        lambda m: make_factor_figure(
            finished_df, MGN_COL, "max_grad_norm", 1.0, m, nan_label="no clip", exclude_reps=[128,256]
        ),
        "max_grad_norm",
    )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group 3b — MoE-specific regularizer sweeps: jitter, EOM & FOM

    Like Group 3, but for the **MoE-only** regularizers — router jitter
    (`jitter_eps`), Expert Output Masking (`eom_prob`), and Final Output Masking
    (`fom_prob`). Dense runs are excluded (these knobs only apply to MoE).

    Color encodes model type; line style encodes the regularizer value (baseline =
    no regularizer set). `olmo_mix`, 80M, log X with power-of-2 ticks.
    """)
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_moe_jitter = metric_multiselect("MoE jitter_eps — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_moe_jitter
    return (sel_moe_jitter,)


@app.cell(hide_code=True)
def _(
    finished_df,
    make_factor_figure,
    render_side_by_side,
    router_field,
    sel_moe_jitter,
):
    # Router jitter: baseline = not set (blank); MoE only.
    _out = render_side_by_side(
        sel_moe_jitter.value,
        lambda m: make_factor_figure(
            finished_df,
            None,
            "jitter_eps",
            None,
            m,
            exclude_archs=["dense"],
            value_getter=lambda row: router_field(row, "jitter_eps"),
        ),
        "moe_jitter_eps",
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_moe_eom = metric_multiselect("MoE eom_prob — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_moe_eom
    return (sel_moe_eom,)


@app.cell(hide_code=True)
def _(
    finished_df,
    make_factor_figure,
    render_side_by_side,
    router_field,
    sel_moe_eom,
):
    # Expert Output Masking: baseline = not set (blank); MoE only.
    _out = render_side_by_side(
        sel_moe_eom.value,
        lambda m: make_factor_figure(
            finished_df,
            None,
            "eom_prob",
            None,
            m,
            exclude_archs=["dense"],
            value_getter=lambda row: router_field(row, "eom_prob"),
        ),
        "moe_eom_prob",
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_moe_fom = metric_multiselect("MoE fom_prob — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_moe_fom
    return (sel_moe_fom,)


@app.cell(hide_code=True)
def _(
    FOM_COL,
    finished_df,
    make_factor_figure,
    render_side_by_side,
    sel_moe_fom,
):
    # Final Output Masking: baseline = not set (blank); MoE only.
    _out = render_side_by_side(
        sel_moe_fom.value,
        lambda m: make_factor_figure(
            finished_df, FOM_COL, "fom_prob", None, m, exclude_archs=["dense"]
        ),
        "moe_fom_prob",
    )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group 4 — `famA` runs

    Every finished run whose name contains `famA`. Color = model type (parsed from
    the run-name suffix, since the newer `famAr` rep-sweep runs are only tagged
    `MoE`). Two plots:

    1. **metric vs. %dclm** (0/25/50/75/100 from `dclm{N}` in the name) — line style
       + shade encode the repetition count.
    2. **metric vs. repetition count** (log X; `rep{N}x` in the name, original famA
       runs are rep 1) — line style + shade encode %dclm.
    """)
    return


@app.cell(hide_code=True)
def _(
    apply_pow2_xaxis,
    axis_label,
    get_reps,
    keep_arch,
    metric_value,
    pd,
    plt,
    re,
    shade_color,
):
    def parse_dclm_pct(name: str):
        """Extract the DCLM percentage from a 'dclm{N}' token in the run name."""
        m = re.search(r"dclm(\d+)", str(name))
        return int(m.group(1)) if m else None

    def famA_model_type(name: str):
        """Model type from the run-name suffix (famAr runs aren't tagged MoE32/64)."""
        m = re.search(r"_(dense|moe32|moe64)$", str(name))
        return m.group(1) if m else None

    _DASH_CYCLE = ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]
    _COLORS = {"dense": "#1f77b4", "moe32": "#2ca02c", "moe64": "#d62728"}

    def _style_maps(keys):
        """Given ordered numeric keys, return {key: dash} and {key: shade level}
        (first darkest -> last lightest)."""
        _DARKEST, _LIGHTEST = -0.45, 0.4
        n = len(keys)
        dash = {k: _DASH_CYCLE[i % len(_DASH_CYCLE)] for i, k in enumerate(keys)}
        shade = {
            k: (_DARKEST + (_LIGHTEST - _DARKEST) * i / (n - 1) if n > 1 else _DARKEST)
            for i, k in enumerate(keys)
        }
        return dash, shade

    def _famA_legend(ax, shown_models, style_title, style_keys, style_dash, key_fmt):
        """Single-box legend: color -> model type, line style -> the other factor."""
        from matplotlib.lines import Line2D

        def header():
            return Line2D([0], [0], linestyle="none", marker="", label="")

        model_order = [m for m in ["dense", "moe32", "moe64"] if m in shown_models]
        model_handles = [
            Line2D([0], [0], color=_COLORS[m], linewidth=2, linestyle="-")
            for m in model_order
        ]
        ng = len(style_keys)
        grey_top = 0.7
        style_handles = [
            Line2D(
                [0],
                [0],
                color=shade_color("#000000", grey_top * i / (ng - 1) if ng > 1 else 0.0),
                linewidth=1.5,
                linestyle=style_dash[k],
            )
            for i, k in enumerate(style_keys)
        ]
        handles = [header()] + model_handles + [header()] + style_handles
        labels = (
            ["Model type"]
            + model_order
            + [style_title]
            + [key_fmt(k) for k in style_keys]
        )
        legend = ax.legend(
            handles,
            labels,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            handlelength=2.2,
        )
        for text in legend.get_texts():
            if text.get_text() in ("Model type", style_title):
                text.set_fontweight("bold")

    def make_famA_figure(df, metric, include_archs=None, exclude_archs=None):
        """famA: metric vs. %dclm. Color = model type; line style + shade = rep count."""
        # Collect: {(model_type, reps): [(pct, value), ...]}
        series: dict = {}
        rep_set = set()
        for _, row in df.iterrows():
            name = str(row.get("Name", ""))
            if "famA" not in name:
                continue
            pct = parse_dclm_pct(name)
            loss = metric_value(row, metric)
            if pct is None or pd.isna(loss):
                continue
            mt = famA_model_type(name)
            if mt is None or not keep_arch(mt, include_archs, exclude_archs):
                continue
            reps = get_reps(row)
            series.setdefault((mt, reps), []).append((pct, float(loss)))
            rep_set.add(reps)

        rep_order = sorted(rep_set)
        rep_dash, rep_shade = _style_maps(rep_order)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        shown_models = []
        for (mt, reps), pts in sorted(series.items()):
            pts = sorted(pts)
            if mt not in shown_models:
                shown_models.append(mt)
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                marker="o",
                markersize=5,
                linewidth=1.5,
                linestyle=rep_dash[reps],
                color=shade_color(_COLORS[mt], rep_shade[reps]),
            )
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("% dclm")
        ax.set_ylabel(axis_label(metric))
        ax.set_title(f"famA runs: {metric} vs. %dclm", fontsize=10)
        if series:
            _famA_legend(
                ax, shown_models, "Repetitions", rep_order, rep_dash, lambda r: f"{r:g}x"
            )
        fig.tight_layout()
        return fig

    def make_famA_rep_figure(df, metric, include_archs=None, exclude_archs=None):
        """famA: metric vs. repetition count (log X). Color = model type; line style +
        shade = %dclm."""
        # Collect: {(model_type, pct): [(reps, value), ...]}
        series: dict = {}
        pct_set = set()
        for _, row in df.iterrows():
            name = str(row.get("Name", ""))
            if "famA" not in name:
                continue
            pct = parse_dclm_pct(name)
            loss = metric_value(row, metric)
            if pct is None or pd.isna(loss):
                continue
            mt = famA_model_type(name)
            if mt is None or not keep_arch(mt, include_archs, exclude_archs):
                continue
            reps = get_reps(row)
            series.setdefault((mt, pct), []).append((reps, float(loss)))
            pct_set.add(pct)

        pct_order = sorted(pct_set)
        pct_dash, pct_shade = _style_maps(pct_order)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        shown_models = []
        max_reps = 0
        any_data = False
        for (mt, pct), pts in sorted(series.items()):
            # Collapse duplicate rep counts (keep min).
            by_reps: dict = {}
            for reps, val in pts:
                by_reps[reps] = min(val, by_reps.get(reps, val))
            xy = sorted(by_reps.items())
            if not xy:
                continue
            any_data = True
            if mt not in shown_models:
                shown_models.append(mt)
            max_reps = max(max_reps, max(r for r, _ in xy))
            ax.plot(
                [r for r, _ in xy],
                [v for _, v in xy],
                marker="o",
                markersize=5,
                linewidth=1.5,
                linestyle=pct_dash[pct],
                color=shade_color(_COLORS[mt], pct_shade[pct]),
            )
        apply_pow2_xaxis(ax, max_reps, any_data)
        ax.set_xlabel("repetition count")
        ax.set_ylabel(axis_label(metric))
        ax.set_title(f"famA runs: {metric} vs. repetition", fontsize=10)
        if any_data:
            _famA_legend(
                ax, shown_models, "% dclm", pct_order, pct_dash, lambda p: f"{p:g}%"
            )
        fig.tight_layout()
        return fig

    return make_famA_figure, make_famA_rep_figure


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_famA = metric_multiselect("famA — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_famA
    return (sel_famA,)


@app.cell(hide_code=True)
def _(finished_df, make_famA_figure, render_side_by_side, sel_famA):
    _out = render_side_by_side(
        sel_famA.value, lambda m: make_famA_figure(finished_df, m), "famA"
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_famA_rep = metric_multiselect(
        "famA (vs reps) — metrics",
        chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"],
    )
    sel_famA_rep
    return (sel_famA_rep,)


@app.cell(hide_code=True)
def _(finished_df, make_famA_rep_figure, render_side_by_side, sel_famA_rep):
    _out = render_side_by_side(
        sel_famA_rep.value,
        lambda m: make_famA_rep_figure(finished_df, m),
        "famA_rep",
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

    - **line color** = model type (blue = dense, green = moe32, red = moe64), and
    - **line style** = data source (`olmo_mix` baseline, single-source baseline, and
      the two famB mixes).

    The `olmo_mix` and single-source lines are **baseline** runs (tag `baseline`);
    duplicate baseline runs at a rep count keep the lowest value (see Group 1 & 2).
    """)
    return


@app.cell(hide_code=True)
def _(
    apply_pow2_xaxis,
    axis_label,
    collect_baseline_points,
    filter_rep_points,
    get_model_type,
    get_reps,
    keep_arch,
    metric_value,
    pd,
    plt,
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
            loss = metric_value(row, metric)
            if pd.isna(reps) or pd.isna(loss):
                continue
            prev = by_reps.get(reps)
            by_reps[reps] = min(float(loss), prev) if prev is not None else float(loss)
        return [(r, by_reps[r]) for r in sorted(by_reps)]

    def make_famB_figure(
        df,
        metric,
        title,
        sources,
        include_archs=None,
        exclude_archs=None,
        include_reps=None,
        exclude_reps=None,
    ):
        """One famB figure.

        `sources` is an ordered list of (label, color, kind, key) tuples where kind is
        either "baseline" (key = data tag, uses baseline runs at 80M) or "famB"
        (key = run-name substring). The per-source color is ignored: color now encodes
        the model type and line style encodes the source.

        Filters (all default to no-op):

        - ``include_archs`` / ``exclude_archs``: keep/drop architectures.
        - ``include_reps`` / ``exclude_reps``: keep/drop repetition counts.
        """
        color_map = {"dense": "#1f77b4", "moe32": "#2ca02c", "moe64": "#d62728"}
        dash_cycle = ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]
        source_dash = {
            src[0]: dash_cycle[i % len(dash_cycle)] for i, src in enumerate(sources)
        }
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        max_reps = 0
        any_data = False
        shown_models = []
        shown_sources = []
        for label, color, kind, key in sources:
            for model_type in ["dense", "moe32", "moe64"]:
                if not keep_arch(model_type, include_archs, exclude_archs):
                    continue
                if kind == "baseline":
                    points = collect_baseline_points(df, key, "80M", model_type, metric)
                else:
                    points = collect_famB_points(df, key, model_type, metric)
                points = filter_rep_points(points, include_reps, exclude_reps)
                if not points:
                    continue
                any_data = True
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                max_reps = max(max_reps, max(xs))
                if model_type not in shown_models:
                    shown_models.append(model_type)
                if label not in shown_sources:
                    shown_sources.append(label)
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    linestyle=source_dash[label],
                    color=color_map[model_type],
                )
        apply_pow2_xaxis(ax, max_reps, any_data)
        ax.set_xlabel("repetition count")
        ax.set_ylabel(axis_label(metric))
        ax.set_title(title, fontsize=10)
        if any_data:
            from matplotlib.lines import Line2D

            def _header():
                return Line2D([0], [0], linestyle="none", marker="", label="")

            # Section 1: color -> model type.
            model_order = [m for m in ["dense", "moe32", "moe64"] if m in shown_models]
            model_handles = [
                Line2D([0], [0], color=color_map[m], linewidth=2, linestyle="-")
                for m in model_order
            ]
            # Section 2: line style -> data source (black/grey).
            source_order = [s[0] for s in sources if s[0] in shown_sources]
            source_handles = [
                Line2D([0], [0], color="#333333", linewidth=1.5, linestyle=source_dash[s])
                for s in source_order
            ]
            # Combine both sections into one legend box with bold section headers.
            handles = [_header()] + model_handles + [_header()] + source_handles
            labels = (
                ["Model type"]
                + model_order
                + ["Data source"]
                + source_order
            )
            legend = ax.legend(
                handles,
                labels,
                fontsize=7,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                handlelength=2.2,
            )
            for text in legend.get_texts():
                if text.get_text() in ("Model type", "Data source"):
                    text.set_fontweight("bold")
        fig.tight_layout()
        return fig

    return (make_famB_figure,)


@app.cell(hide_code=True)
def _(
    get_model_type,
    get_reps,
    get_scale,
    has_tag,
    keep_arch,
    load_history_series,
    pd,
    plt,
    shade_color,
):
    def make_training_curve_figure(
        df,
        data_tag,
        scale,
        metric,
        include_archs=None,
        exclude_archs=None,
        include_reps=None,
        exclude_reps=None,
    ):
        """Training curves (metric vs. train step) for baseline runs of one
        data mixture + model scale.

        One line per run. Color encodes architecture; repetition count sets both
        the shade (baseline/lowest-rep darkest -> highest-rep lightest) and the
        line style. Reads per-step values from the downloaded W&B history.
        """
        # Collect baseline runs: {(model_type, reps): run_name} (keep last seen).
        runs: dict = {}
        rep_set = set()
        for _, row in df.iterrows():
            if get_scale(row) != scale:
                continue
            if not has_tag(row, "baseline"):
                continue
            if not has_tag(row, data_tag):
                continue
            model_type = get_model_type(row)
            if model_type not in ("dense", "moe32", "moe64"):
                continue
            if not keep_arch(model_type, include_archs, exclude_archs):
                continue
            reps = get_reps(row)
            if pd.isna(reps):
                continue
            reps = float(reps)
            if include_reps is not None and reps not in {float(r) for r in include_reps}:
                continue
            if exclude_reps and reps in {float(r) for r in exclude_reps}:
                continue
            runs[(model_type, reps)] = str(row.get("Name", ""))
            rep_set.add(reps)

        color_map = {"dense": "#1f77b4", "moe32": "#2ca02c", "moe64": "#d62728"}
        dash_cycle = ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]
        rep_order = sorted(rep_set)
        # Shade + dash per repetition (lowest rep darkest, highest lightest).
        _DARKEST, _LIGHTEST = -0.45, 0.4
        _nr = len(rep_order)
        rep_shade = {
            r: (_DARKEST + (_LIGHTEST - _DARKEST) * i / (_nr - 1) if _nr > 1 else _DARKEST)
            for i, r in enumerate(rep_order)
        }
        rep_dash = {r: dash_cycle[i % len(dash_cycle)] for i, r in enumerate(rep_order)}

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        any_data = False
        shown_models, shown_reps = [], []
        for (model_type, reps), run_name in sorted(
            runs.items(), key=lambda kv: (kv[0][0], kv[0][1])
        ):
            series = load_history_series(run_name, metric)
            if not series:
                continue
            any_data = True
            xs = [p[0] for p in series]
            ys = [p[1] for p in series]
            if model_type not in shown_models:
                shown_models.append(model_type)
            if reps not in shown_reps:
                shown_reps.append(reps)
            ax.plot(
                xs,
                ys,
                linewidth=1.0,
                linestyle=rep_dash[reps],
                color=shade_color(color_map[model_type], rep_shade[reps]),
            )
        ax.set_xlabel("train step")
        ax.set_ylabel(metric)
        title = f"{data_tag} — {scale}: {metric} vs. train step (baseline)"
        if not any_data:
            title += "  [no history found]"
        ax.set_title(title, fontsize=10)

        if any_data:
            from matplotlib.lines import Line2D

            def _header():
                return Line2D([0], [0], linestyle="none", marker="", label="")

            model_order = [m for m in ["dense", "moe32", "moe64"] if m in shown_models]
            model_handles = [
                Line2D([0], [0], color=color_map[m], linewidth=2, linestyle="-")
                for m in model_order
            ]
            rep_shown = [r for r in rep_order if r in shown_reps]
            _ng = len(rep_shown)
            _grey_top = 0.7
            rep_handles = [
                Line2D(
                    [0],
                    [0],
                    color=shade_color("#000000", _grey_top * i / (_ng - 1) if _ng > 1 else 0.0),
                    linewidth=1.5,
                    linestyle=rep_dash[r],
                )
                for i, r in enumerate(rep_shown)
            ]
            handles = [_header()] + model_handles + [_header()] + rep_handles
            labels = (
                ["Model type"]
                + model_order
                + ["Repetitions"]
                + [f"{r:g}x" for r in rep_shown]
            )
            legend = ax.legend(
                handles,
                labels,
                fontsize=7,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                handlelength=2.2,
            )
            for text in legend.get_texts():
                if text.get_text() in ("Model type", "Repetitions"):
                    text.set_fontweight("bold")
        fig.tight_layout()
        return fig

    return (make_training_curve_figure,)


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_famB_sc = metric_multiselect("famB starcoder — metrics", chosen_values=["eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_stack-validation/CE loss"])
    sel_famB_sc
    return (sel_famB_sc,)


@app.cell(hide_code=True)
def _(finished_df, make_famB_figure, render_side_by_side, sel_famB_sc):
    # Plot 1: starcoder family. Color = model type; line style = source.
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
        "famB_starcoder",
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_famB_p2o = metric_multiselect("famB pes2o — metrics", chosen_values=["eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_pes2o-validation/CE loss"])
    sel_famB_p2o
    return (sel_famB_p2o,)


@app.cell(hide_code=True)
def _(finished_df, make_famB_figure, render_side_by_side, sel_famB_p2o):
    # Plot 2: pes2o family. Color = model type; line style = source.
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
        "famB_pes2o",
    )
    _out
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training curves — metric vs. train step (baseline runs)

    Per-step training curves from the downloaded W&B history
    (`ml/runs_data/history/`). Baseline `olmo_mix` runs (and the single-source
    mixtures) at 80M / 200M. Color = architecture; repetition count sets the shade
    (lowest rep darkest → highest lightest) and the line style.
    """)
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_curve_olmo_80m = metric_multiselect(
        "curves: olmo_mix 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_curve_olmo_80m
    return (sel_curve_olmo_80m,)


@app.cell(hide_code=True)
def _(
    finished_df,
    make_training_curve_figure,
    render_side_by_side,
    sel_curve_olmo_80m,
):
    _out = render_side_by_side(
        sel_curve_olmo_80m.value,
        lambda m: make_training_curve_figure(finished_df, "olmo_mix", "80M", m),
        "curve_olmo_mix_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_curve_olmo_200m = metric_multiselect(
        "curves: olmo_mix 200M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_curve_olmo_200m
    return (sel_curve_olmo_200m,)


@app.cell(hide_code=True)
def _(
    finished_df,
    make_training_curve_figure,
    render_side_by_side,
    sel_curve_olmo_200m,
):
    _out = render_side_by_side(
        sel_curve_olmo_200m.value,
        lambda m: make_training_curve_figure(finished_df, "olmo_mix", "200M", m),
        "curve_olmo_mix_200M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Single-source mixtures (80M): `dclm`, `starcoder`, `pes2o`, `wiki`
    """)
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_curve_dclm_80m = metric_multiselect(
        "curves: dclm 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_curve_dclm_80m
    return (sel_curve_dclm_80m,)


@app.cell(hide_code=True)
def _(
    finished_df,
    make_training_curve_figure,
    render_side_by_side,
    sel_curve_dclm_80m,
):
    _out = render_side_by_side(
        sel_curve_dclm_80m.value,
        lambda m: make_training_curve_figure(finished_df, "dclm", "80M", m),
        "curve_dclm_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_curve_starcoder_80m = metric_multiselect(
        "curves: starcoder 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_stack-validation/CE loss"])
    sel_curve_starcoder_80m
    return (sel_curve_starcoder_80m,)


@app.cell(hide_code=True)
def _(
    finished_df,
    make_training_curve_figure,
    render_side_by_side,
    sel_curve_starcoder_80m,
):
    _out = render_side_by_side(
        sel_curve_starcoder_80m.value,
        lambda m: make_training_curve_figure(finished_df, "starcoder", "80M", m),
        "curve_starcoder_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_curve_pes2o_80m = metric_multiselect(
        "curves: pes2o 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_pes2o-validation/CE loss"])
    sel_curve_pes2o_80m
    return (sel_curve_pes2o_80m,)


@app.cell(hide_code=True)
def _(
    finished_df,
    make_training_curve_figure,
    render_side_by_side,
    sel_curve_pes2o_80m,
):
    _out = render_side_by_side(
        sel_curve_pes2o_80m.value,
        lambda m: make_training_curve_figure(finished_df, "pes2o", "80M", m),
        "curve_pes2o_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(metric_multiselect):
    sel_curve_wiki_80m = metric_multiselect(
        "curves: wiki 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_wiki-validation/CE loss"]
    )
    sel_curve_wiki_80m
    return (sel_curve_wiki_80m,)


@app.cell(hide_code=True)
def _(
    finished_df,
    make_training_curve_figure,
    render_side_by_side,
    sel_curve_wiki_80m,
):
    _out = render_side_by_side(
        sel_curve_wiki_80m.value,
        lambda m: make_training_curve_figure(finished_df, "wiki", "80M", m),
        "curve_wiki_80M",
    )
    _out
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
