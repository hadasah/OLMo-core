import marimo

__generated_with = "0.24.0"
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
       runs on the `olmo_mix` mixture — one figure per model scale (10M, 80M, 200M,
       1B). The 1B figure is wired up but empty until those runs finish *and* are
       tagged `baseline`.
    1b. Off-axis run families, each in its own frame and excluded from every other
       group: the **4C** token-budget runs (`data_rep_AC_x4`, tagged `Data=4C`) at
       80M, the **small-batch** 10M runs (global batch 131072), and the
       **high-LR** 10M runs (`lr=1e-2`).
    2. The same, repeated for each single-source data mixture: `dclm`,
       `starcoder`, `pes2o`, `wiki`.
    3. Regularizer sweeps (**dropout**, **weight decay**, **max grad norm**) —
       color = model type, line style = regularizer value; loss vs. repetition (log X).
    3b. MoE-only regularizer sweeps (**jitter_eps**, **EOM**, **FOM**,
       **expert_dropout**) — same encoding, Dense excluded.
    4. All `famA` runs: loss vs. `%dclm` (0/25/50/75/100 from the run name).
    5. `famB` runs: single-source mixes into `olmo_mix` vs. repetition.
    """)
    return


@app.cell(hide_code=True)
def _():
    import glob
    import json
    import math
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
    return glob, json, math, mticker, os, pd, plt, re


@app.cell
def _():
    import marimo as mo

    return (mo,)


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
    EXPERT_DROPOUT_COL = "model.block.expert_dropout"
    WD_COL = "train_module.optim.weight_decay"
    MGN_COL = "train_module.max_grad_norm"
    FOM_COL = "model.block.feed_forward_moe.fom_prob"
    # Dense runs record FOM on the block itself rather than under the MoE config,
    # so the knob lives in one of two columns depending on architecture.
    DENSE_FOM_COL = "model.block.fom_prob"
    ROUTERS_COL = "model.block.feed_forward_moe.routers_list"
    REPS_COL = "data_reps"

    def fom_prob_value(row):
        """FOM probability for a run, from whichever column its architecture uses.

        MoE runs set ``model.block.feed_forward_moe.fom_prob``; the dense runs set
        ``model.block.fom_prob``. Reading only the MoE column would silently file
        every dense FOM run into the baseline bucket.
        """
        for _c in (FOM_COL, DENSE_FOM_COL):
            v = row.get(_c)
            if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v).strip() != "":
                return v
        return None

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
        """Model scale is encoded in the run name, e.g. 'olmo2_ml_80M'.

        The 10M check is underscore-anchored on purpose: a bare ``"10M" in name``
        would also match ``olmo2_ml_110M``.
        """
        name = str(row.get("Name", ""))
        if "1B" in name:
            return "1B"
        if "200M" in name:
            return "200M"
        if "80M" in name:
            return "80M"
        if "_10M" in name:
            return "10M"
        return "other"

    def get_model_type(row) -> str:
        """Classify a run's architecture from its tags.

        The expert-size variants (``MoE64s8`` = 64 experts at 1/8 the dense FFN,
        ``MoE64s32`` = 1/2) must be checked *before* plain ``MoE64``: some of those
        runs carry both tags, and the narrower one is the truthful label.
        """
        tags = parse_tags(row.get("Tags"))
        if "dense" in tags:
            return "dense"
        if "MoE64s8" in tags:
            return "moe64s8"
        if "MoE64s32" in tags:
            return "moe64s32"
        if "MoE128" in tags:
            return "moe128"
        if "MoE64" in tags:
            return "moe64"
        if "MoE32" in tags:
            return "moe32"
        if "MoE16" in tags:
            return "moe16"
        return "other"

    MIX_COL = "dataset.mix"
    MAX_DUR_COL = "trainer.max_duration.value"
    LR_COL = "train_module.optim.lr"
    BATCH_COL = "data_loader.global_batch_size"
    MIX_BATCH_COL = "dataset.source_mixture_config.global_batch_size"
    SEQ_LEN_COL = "dataset.sequence_length"
    REQ_TOKENS_COL = "dataset.source_mixture_config.requested_tokens"
    SOURCES_COL = "dataset.source_mixture_config.source_list.sources"

    def max_repetition_ratio(row) -> float:
        """Largest per-source ``max_repetition_ratio`` in the mixture (1.0 if none).

        famB runs keep a fixed token budget and instead encode repetition of the
        minority source via its ``max_repetition_ratio`` (e.g. rep8x -> 8), so the
        ``max_duration / requested_tokens`` ratio stays 1 for them. Reading the max
        ratio across sources recovers the intended repetition count.
        """
        raw = row.get(SOURCES_COL)
        if not isinstance(raw, str):
            return 1.0
        try:
            sources = json.loads(raw)
        except (ValueError, TypeError):
            return 1.0
        best = 1.0
        for s in sources or []:
            try:
                r = float(s.get("max_repetition_ratio"))
            except (TypeError, ValueError):
                continue
            if r > best:
                best = r
        return best

    def get_reps(row) -> float:
        """Repetition count for a run.

        Two encodings exist and we take whichever implies more repetition:

        1. **Shrunken token budget** (baseline data_rep runs): total training tokens
           over unique tokens, i.e. ``trainer.max_duration.value /
           dataset.source_mixture_config.requested_tokens``.
        2. **Per-source repetition ratio** (famB runs): the mixture keeps a fixed
           token budget (so ratio #1 is 1) and repeats the minority source via its
           ``max_repetition_ratio``.

        An early (pre-2026-04-10) bug wrote wrong repetition counts into run *ids*
        and the ``data_reps`` column, so neither of those can be trusted. The wide
        CSV's ``Name`` column, however, is the *corrected* run name, and its
        ``requested_tokens`` / ``max_duration`` / ``source_list`` columns are
        correct — so we compute the count straight from them. A run with neither
        signal is single-pass, i.e. rep 1.
        """
        token_reps = 1.0
        md = row.get(MAX_DUR_COL)
        rt = row.get(REQ_TOKENS_COL)
        try:
            if pd.notna(rt) and float(rt) > 0 and pd.notna(md):
                token_reps = float(md) / float(rt)
        except (TypeError, ValueError):
            pass
        # No source-mixture budget signal -> fall back to per-source rep ratio.
        return max(token_reps, max_repetition_ratio(row))

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

    # Per-model-type lightening: dense = darkest, moe64 = lightest. Positive values
    # blend toward white; negative toward black. moe64 now sits at the old moe32
    # level and dense goes slightly below pure base color.
    MODEL_SHADE = {
        "dense": -0.2,
        "moe16": -0.15,
        "moe32": -0.1,
        "moe64s8": -0.05,
        "moe64": 0.05,
        "moe128": 0.08,
        "moe64s32": 0.1,
    }

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

    def arch_of_run(row) -> str:
        """Architecture (dense/moe32/moe64) from the run-name suffix, falling back to
        tags.

        Some sweeps (the ``famAr`` rep runs, the ``famB`` ``p2o10`` runs) are only
        tagged ``MoE`` without the ``MoE32``/``MoE64`` tag, so tag-only classification
        drops them as "other". The trailing ``_dense``/``_moe32``/``_moe64`` in the run
        name is reliable, so prefer it and fall back to tags when it is absent.
        """
        import re as _re

        m = _re.search(r"_(dense|moe32|moe64)$", str(row.get("Name", "")))
        if m:
            return m.group(1)
        return get_model_type(row)

    _arch_of = arch_of_run  # backwards-compatible alias used by the dedup key

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
        starcoder / pes2o / wiki), the ``dataset.mix`` config field, and any
        ``dclm{N}`` percentage from the name, so e.g. wiki vs olmo_mix, and dclm0 vs
        dclm100, aren't merged just because they all lack a sources config.

        ``dataset.mix`` is what makes this reliable for the older (pre-2026-05)
        single-source runs: they are tagged ``ml, dropless, 1.0gen, ...`` with no
        data-source tag at all, so the tag alone reads as ``None`` for every one of
        them and a pes2o rep1x run collides with a starcoder rep1x run that matches
        on every other field."""
        if _sources_key(row) is not None:
            return None
        import re as _re

        m = _re.search(r"dclm(\d+)", str(row.get("Name", "")))
        return (
            get_data_tag(row),
            _norm(row.get(MIX_COL)),
            int(m.group(1)) if m else None,
        )

    def _dedup_key(row):
        """Identity key for de-duplicating runs that are the same experiment.

        ``max_duration`` (the total training token budget) is part of the identity:
        the same unique data at the same ``requested_tokens`` trained for a different
        number of total tokens is a *different* experiment. Without it the 4C
        (``data_rep_AC_x4``) runs collide with the 1C runs — e.g. 4C/rep8x and
        1C/rep2x both read 800M unique tokens — and get dropped as false duplicates.

        Batch size and sequence length are likewise part of the identity: the
        ``new_bs_256plus`` sweep re-runs earlier configs at a larger batch, and
        without them those runs collide with the originals they were meant to be
        compared against.

        NOTE: this key is an explicit allowlist, so any *new* config knob is
        invisible to it and its runs will silently vanish as false duplicates. Add
        the field here whenever you add a swept hyperparameter.
        """
        return (
            _sources_key(row),
            _namemix_key(row),
            _norm(row.get("dataset.source_mixture_config.requested_tokens")),
            _norm(row.get(MAX_DUR_COL)),
            _norm(row.get(BATCH_COL)),
            _norm(row.get(MIX_BATCH_COL)),
            _norm(row.get(SEQ_LEN_COL)),
            _norm(row.get("model.d_model")),
            _norm(row.get(DROPOUT_COL)),
            _norm(row.get(EXPERT_DROPOUT_COL)),
            _norm(row.get(WD_COL)),
            _norm(row.get(MGN_COL)),
            _norm(row.get(LR_COL)),
            _norm(router_field(row, "jitter_eps")),
            _norm(router_field(row, "eom_prob")),
            # Coalesced: dense runs record FOM on the block, MoE runs under the MoE
            # config. Reading only one column makes every dense FOM run look
            # identical to the plain dense baseline at the same rep, and they get
            # dropped here as false duplicates before any plot sees them.
            _norm(fom_prob_value(row)),
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

    # The 4C ("x4" token budget) repetition family. These runs share the mixture and
    # tags of the 1C runs but train for 4x the tokens, so they must not be mixed into
    # any of the 1C plots — they are split out of `finished_df` into `x4_df` and
    # plotted on their own in Group 1b.
    X4_NAME_MARKER = "data_rep_AC_x4"
    X4_TAG = "Data=4C"

    def is_x4_run(row) -> bool:
        """True for the 4C repetition family (name marker, tag as a fallback)."""
        return X4_NAME_MARKER in str(row.get("Name", "")) or has_tag(row, X4_TAG)

    def split_x4(df: pd.DataFrame):
        """Split a frame into (everything else, the 4C family)."""
        mask = pd.Series(
            [is_x4_run(row) for _, row in df.iterrows()], index=df.index, dtype=bool
        )
        return df[~mask].copy(), df[mask].copy()

    # The small-batch ("bs 256") family: the `new_bs_256plus` 10M sweep re-ran the
    # extreme repetition counts (256x/512x/1024x) at a global batch of 131072 tokens
    # instead of the usual 1048576. Same treatment as the 4C family — a different
    # batch size is a different experiment, so they get their own frame rather than
    # sharing an axis with the regular-batch runs.
    #
    # NOTE: only `data_loader.global_batch_size` is used. The same sweep also varies
    # `source_mixture_config.global_batch_size` across several values at the regular
    # 1048576 data-loader batch; those runs stay in `finished_df`.
    SMALL_BS_VALUE = 131072

    def is_small_bs_run(row) -> bool:
        """True for runs at the reduced 131072-token global batch."""
        try:
            return float(row.get(BATCH_COL)) == float(SMALL_BS_VALUE)
        except (TypeError, ValueError):
            return False

    def split_small_bs(df: pd.DataFrame):
        """Split a frame into (everything else, the small-batch family)."""
        mask = pd.Series(
            [is_small_bs_run(row) for _, row in df.iterrows()], index=df.index, dtype=bool
        )
        return df[~mask].copy(), df[mask].copy()

    # The high-LR family: a 10M sweep at lr=1e-2 (`data_rep_AC_lr1e-2` in the run
    # name) against the 4e-4 used everywhere else. At 10M that reaches ~6.5 train CE
    # vs ~9.3, so it dwarfs every other effect and cannot share an axis. Splitting on
    # the LR value rather than the name marker: the two select exactly the same 9
    # runs, and every other run in the project is at 4e-4.
    OFF_AXIS_LR = 0.01

    def is_high_lr_run(row) -> bool:
        """True for runs at the off-axis 1e-2 learning rate."""
        try:
            return float(row.get(LR_COL)) == float(OFF_AXIS_LR)
        except (TypeError, ValueError):
            return False

    def split_high_lr(df: pd.DataFrame):
        """Split a frame into (everything else, the lr=1e-2 family)."""
        mask = pd.Series(
            [is_high_lr_run(row) for _, row in df.iterrows()], index=df.index, dtype=bool
        )
        return df[~mask].copy(), df[mask].copy()

    def expert_size_frame(df: pd.DataFrame):
        """80M baselines for the expert-size comparison.

        A *view*, not a split: the olmo_mix dense/moe64 baselines stay in
        `finished_df` for Group 1 as well. The MoE64s8 / MoE64s32 runs carry no
        mixture tag, so the figure has to be drawn with the mixture filter off
        (`data_tag=None`); restricting the frame here is what stops the dense and
        moe64 lines from picking up dclm/starcoder/pes2o/wiki runs as well.
        """
        keep = [
            idx
            for idx, row in df.iterrows()
            if get_scale(row) == "80M"
            and has_tag(row, "baseline")
            and (
                has_tag(row, "olmo_mix")
                or get_model_type(row) in ("moe64s8", "moe64s32")
            )
        ]
        return df.loc[keep].copy()

    return (
        DROPOUT_COL,
        EXPERT_DROPOUT_COL,
        MGN_COL,
        MODEL_SHADE,
        WD_COL,
        X4_TAG,
        arch_of_run,
        dedupe_runs,
        expert_size_frame,
        filter_finished,
        filter_rep_points,
        fom_prob_value,
        get_model_type,
        get_reps,
        get_scale,
        has_tag,
        keep_arch,
        pow2_ticks,
        router_field,
        shade_color,
        split_high_lr,
        split_small_bs,
        split_x4,
    )


@app.cell
def _(
    dedupe_runs,
    expert_size_frame,
    filter_finished,
    raw_df,
    split_high_lr,
    split_small_bs,
    split_x4,
):
    # Single filtered frame reused by every figure below. Duplicate runs (same
    # experiment identity) are dropped up front so no plot double-counts them.
    #
    # The 4C repetition family is then split off into its own frame: `finished_df`
    # (used by every group except 1b) never sees it, so those runs cannot leak into
    # the 1C plots — notably the Group 3/3b baseline bucket, which selects on
    # 80M + olmo_mix + knobs-at-default rather than on a tag.
    #
    # The small-batch (131072-token) and high-LR (1e-2) 10M runs are split off the
    # same way, so the regular 10M plot in Group 1 and the two off-axis ones in
    # Group 1b never share an axis.
    finished_df, x4_df = split_x4(dedupe_runs(filter_finished(raw_df)))
    finished_df, small_bs_df = split_small_bs(finished_df)
    finished_df, high_lr_df = split_high_lr(finished_df)
    # Derived view (not a split) for the Group 1b expert-size comparison.
    expert_size_df = expert_size_frame(finished_df)
    return expert_size_df, finished_df, high_lr_df, small_bs_df, x4_df


@app.cell(hide_code=True)
def _(
    ARCH_ORDER_DEFAULT,
    MODEL_DISPLAY_NAME,
    MODEL_SHADE,
    math,
    mticker,
    plt,
    shade_color,
):
    # ------------------------------------------------------------------
    # Shared visual encoding used by EVERY figure:
    #   - architecture -> line style + shade (dense solid/darkest,
    #     moe32 dashed, moe64 dotted/lightest)
    #   - color -> the plot's factor (regularizer value, data source,
    #     rep count, or %dclm), assigned from a perceptual colormap.
    # ------------------------------------------------------------------
    from matplotlib.colors import to_hex
    from matplotlib.lines import Line2D

    ARCH_DASH = {
        "dense": "-",
        "moe16": (0, (3, 1, 1, 1)),
        "moe32": "--",
        "moe64": ":",
        "moe128": (0, (1, 1)),
        "moe64s8": "-.",
        "moe64s32": (0, (5, 1)),
    }
    # Per-architecture base color, used only by plots where architecture is the sole
    # variable (Groups 1 & 2). These are the viridis endpoints (purple / greenish-blue
    # / yellow) so they match the factor palette used everywhere else. Elsewhere color
    # encodes the factor and architecture is carried by line style + shade.
    _viridis = plt.get_cmap("viridis")
    ARCH_COLOR = {
        # Spread over [0, 0.8] -- the yellow end stays unused. Ordered so both
        # Group 1b comparisons read monotonically: expert count
        # (dense < 16 < 32 < 64 < 128) and expert size (dense < s8 < 64 < s32).
        "dense": to_hex(_viridis(0.0)),
        "moe16": to_hex(_viridis(0.16)),
        "moe32": to_hex(_viridis(0.32)),
        "moe64s8": to_hex(_viridis(0.32)),
        "moe64": to_hex(_viridis(0.60)),
        "moe128": to_hex(_viridis(0.80)),
        "moe64s32": to_hex(_viridis(0.80)),
    }

    def arch_dash(model_type):
        return ARCH_DASH.get(model_type, "-")

    def arch_shade(model_type):
        return MODEL_SHADE.get(model_type, 0.0)

    def arch_color(base_color, model_type):
        """Shade a per-factor base color by architecture."""
        return shade_color(base_color, arch_shade(model_type))

    def factor_palette(keys, high_end=0.8):
        """Distinct color per factor key (in the given order), spread across viridis.

        A single key gets a fixed mid-palette color so lone-series plots stay
        readable rather than collapsing to an extreme of the colormap.
        """
        keys = list(keys)
        n = len(keys)
        cmap = plt.get_cmap("viridis")
        if n == 1:
            return {keys[0]: to_hex(cmap(0.35))}
        return {k: to_hex(cmap(i * high_end / (n - 1))) for i, k in enumerate(keys)}

    def arch_factor_legend(
        ax,
        shown_archs,
        factor_title=None,
        factor_keys=(),
        factor_color=None,
        key_fmt=str,
        arch_colored=False,
        arch_order=None,
    ):
        """Single-box legend. Section 1: line style + shade -> architecture. When
        ``arch_colored`` is set, the architecture swatches also use each arch's base
        color (for Groups 1 & 2, where color encodes architecture); otherwise they are
        drawn in grey so only style/shade read. Section 2 (optional): color -> factor."""

        def _header():
            return Line2D([0], [0], linestyle="none", marker="", label="")

        arch_order = [
            a for a in (arch_order or ARCH_ORDER_DEFAULT) if a in shown_archs
        ]
        arch_handles = [
            Line2D(
                [0],
                [0],
                color=(
                    ARCH_COLOR[a]
                    if arch_colored
                    else shade_color("#000000", arch_shade(a) + 0.2)
                ),
                linewidth=1.5,
                linestyle=arch_dash(a),
            )
            for a in arch_order
        ]
        handles = [_header()] + arch_handles
        # Show the human-readable architecture names (e.g. "MoE (64 x 1/4)"), falling
        # back to the raw key for any arch not listed in MODEL_DISPLAY_NAME.
        labels = ["Model type"] + [MODEL_DISPLAY_NAME.get(a, a) for a in arch_order]
        factor_keys = list(factor_keys)
        if factor_keys and factor_color is not None:
            factor_handles = [
                Line2D([0], [0], color=factor_color[k], linewidth=2, linestyle="-")
                for k in factor_keys
            ]
            handles += [_header()] + factor_handles
            labels += [factor_title] + [key_fmt(k) for k in factor_keys]
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
            if text.get_text() in ("Model type", factor_title):
                text.set_fontweight("bold")

    def _plain_num(y, _pos=None):
        """Tick label as a plain number: 5, 0.9, 0.002 — never 10^0 or 6.000."""
        if y <= 0:
            return ""
        s = f"{y:g}"
        if "e" in s or "E" in s:  # keep extreme values out of scientific notation
            s = f"{y:.10f}".rstrip("0").rstrip(".")
        return s

    def apply_log_yaxis(ax):
        """Log-scale the y axis but keep plain-number tick labels.

        Matplotlib's default log formatter writes powers of ten (``10^0``, ``10^1``),
        which is unreadable for loss values. We label with plain numbers instead.

        Two things adapt to how many decades the data spans, tracked separately:

        - ``grid_subs`` — which sub-decade ticks *exist*, and therefore where the
          horizontal rules are drawn. Decade-only gridlines are far too sparse (a
          half-decade range gets a single rule), so there is always something
          between the decades.
        - ``label_subs`` — which of those get a text label. Labelling 2..9 inside
          every decade only reads well over a narrow span; across ~4 decades (a
          train loss collapsing from 5 to 0.002) it becomes a wall of overlapping
          text, so wide ranges keep the rules but drop the sub-decade labels.
        """
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        decades = (
            math.log10(ymax) - math.log10(ymin) if ymin > 0 and ymax > ymin else 1.0
        )
        if decades <= 1.5:
            grid_subs = label_subs = tuple(range(2, 10))
        elif decades <= 3.0:
            grid_subs = label_subs = (2, 3, 5)
        else:
            grid_subs, label_subs = (2, 5), ()

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_plain_num))
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=grid_subs, numticks=15)
        )
        if label_subs:
            ax.yaxis.set_minor_formatter(mticker.FuncFormatter(_plain_num))
        else:
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        # Sub-decade labels are ordinary tick labels, not annotations: keep them the
        # same size as the decade ones so the axis doesn't read as two tiers.
        ax.tick_params(axis="y", which="both", labelsize=plt.rcParams["ytick.labelsize"])

        # Horizontal rules at every tick, with the sub-decade ones lighter so the
        # decade boundaries still read as the primary reference.
        ax.grid(True, which="major", axis="y", linewidth=0.8, alpha=0.7)
        ax.grid(True, which="minor", axis="y", linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)

    return (
        ARCH_COLOR,
        apply_log_yaxis,
        arch_color,
        arch_dash,
        arch_factor_legend,
        factor_palette,
    )


@app.cell(hide_code=True)
def _(METRIC_DISPLAY_NAMES, json, os):
    # ------------------------------------------------------------------
    # Per-step training history (for the training-curve column, and for the
    # smoothed train loss in the column-1 plots).
    #
    # ml/wandb_download/download.py --include-history writes one JSONL per run to
    # ml/runs_data/history/<run_id>.jsonl, each line a logged step with a "_step"
    # field plus whatever metrics were logged that step (train/CE loss is dense;
    # eval metrics are sparse).
    #
    # IMPORTANT: history files are named by *run_id*, but everywhere else the
    # notebook identifies a run by the CSV ``Name`` column, which is the *corrected*
    # run ``name``. For the early (pre-2026-04-10) runs the run_id carries a wrong
    # rep label while the name is corrected, so name != run_id. We therefore
    # translate name -> run_id (via runs.jsonl) before opening a history file, or
    # the smoothed train loss / training curves would be cross-wired (e.g. the
    # rep-16 run would load the rep-128 run's history).
    # ------------------------------------------------------------------
    def _runs_data_dir():
        _this_dir = (
            os.path.dirname(os.path.abspath(__file__))
            if "__file__" in dir()
            else os.getcwd()
        )
        _candidates = [
            os.path.join(_this_dir, "..", "runs_data"),
            os.path.join(_this_dir, "ml", "runs_data"),
        ]
        for d in _candidates:
            if os.path.isdir(os.path.normpath(d)):
                return os.path.normpath(d)
        return os.path.normpath(_candidates[0])

    RUNS_DATA_DIR = _runs_data_dir()
    HISTORY_DIR = os.path.join(RUNS_DATA_DIR, "history")

    def _load_name_to_run_id():
        """Map corrected run ``name`` -> ``run_id`` from runs.jsonl. History files
        are named by run_id, but the CSV/plots key on name."""
        path = os.path.join(RUNS_DATA_DIR, "runs.jsonl")
        mapping: dict = {}
        if not os.path.isfile(path):
            return mapping
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except (ValueError, TypeError):
                    continue
                name = d.get("name")
                rid = d.get("run_id")
                if name is not None and rid is not None:
                    mapping[name] = rid
        return mapping

    NAME_TO_RUN_ID = _load_name_to_run_id()

    _history_cache: dict = {}

    def load_history_series(run_name: str, metric: str):
        """Return sorted [(step, value), ...] for `metric` in a run's history.

        `run_name` is the CSV ``Name`` (the corrected run name). We resolve it to the
        run_id (history files are named by run_id) before reading. Keeping only rows
        where both ``_step`` and the metric are present. Cached per (run, metric).
        Returns [] if the history file is missing.
        """
        cache_key = (run_name, metric)
        if cache_key in _history_cache:
            return _history_cache[cache_key]
        # History files are keyed by run_id; translate name -> run_id (falling back
        # to the name itself for runs where they coincide / aren't in runs.jsonl).
        run_id = NAME_TO_RUN_ID.get(run_name, run_name)
        path = os.path.join(HISTORY_DIR, f"{run_id}.jsonl")
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
    # smoothed axis is labeled. Every ``train/*`` metric is averaged over the last
    # SMOOTH_WINDOW logged steps (they are logged every step), matching train loss.
    SMOOTHED_METRIC = "train/CE loss"
    SMOOTH_WINDOW = 100
    SMOOTHED_LABEL = f"{SMOOTHED_METRIC} (last-{SMOOTH_WINDOW}-step avg)"

    def is_smoothed_metric(metric) -> bool:
        """Whether `metric` gets last-window averaging (all ``train/*`` metrics)."""
        return isinstance(metric, str) and metric.startswith("train/")

    def smoothed_metric_value(run_name: str, metric: str, window: int = SMOOTH_WINDOW):
        """Mean of `metric` over the last `window` logged steps of a run.

        ``train/*`` metrics are logged every step, so the last `window` logged points
        are the last `window` steps. Returns None if the run has no usable history.
        """
        series = load_history_series(run_name, metric)
        if not series:
            return None
        tail = series[-window:]
        if not tail:
            return None
        return sum(v for _, v in tail) / len(tail)

    def smoothed_train_loss(run_name: str, window: int = SMOOTH_WINDOW):
        """Mean of ``train/CE loss`` over the last `window` logged steps of a run."""
        return smoothed_metric_value(run_name, SMOOTHED_METRIC, window)

    def metric_value(row, metric):
        """Value to plot for `metric` in the column-1 (final-value) figures.

        For any ``train/*`` metric this is the mean over the last SMOOTH_WINDOW steps
        from the run's history; every other metric uses its raw final value from the
        CSV. Falls back to the raw CSV value if a run has no history.
        """
        if is_smoothed_metric(metric):
            sm = smoothed_metric_value(str(row.get("Name", "")), metric)
            if sm is not None:
                return sm
        return row.get(metric)

    def axis_label(raw_metric):
        """Y-axis / legend label for a metric (annotates the smoothed train metrics)."""
        metric = METRIC_DISPLAY_NAMES.get(raw_metric, raw_metric)
        if is_smoothed_metric(raw_metric):
            return f"{metric} (last-{SMOOTH_WINDOW}-step avg)"
        return metric

    return axis_label, load_history_series, metric_value


@app.cell(hide_code=True)
def _(METRIC_SHORTNAMES, mo, os, plt):
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

    # ------------------------------------------------------------------
    # Organized export used by the per-column "Export images" buttons.
    # Files land in ml/figures/<subfolder>/ and are named by their
    # distinguishing features joined with "__", in the order:
    #   data source __ model scale __ architectures __ varied feature
    # with the metric appended last for the training-history columns.
    # ------------------------------------------------------------------
    # Canonical architecture ordering, shared by the legend, the per-figure arch
    # loops and the export filenames. Individual plots may override the order.
    ARCH_ORDER_DEFAULT = [
        "dense", "moe16", "moe32", "moe64s8", "moe64", "moe64s32", "moe128",
    ]

    def arch_tag(archs):
        """Canonical architecture part of a filename, e.g. ``dense_moe32_moe64``."""
        sel = [a for a in ARCH_ORDER_DEFAULT if a in (archs or [])]
        return "_".join(sel) if sel else "none"

    def metric_tag(metric):
        """Short, filename-friendly name for a metric.

        Uses ``METRIC_SHORTNAMES`` when the metric is listed there (e.g.
        ``train/CE loss`` -> ``train_loss``); otherwise the raw metric name is used
        and ``export_pdf`` sanitizes it.
        """
        return METRIC_SHORTNAMES.get(metric, metric)

    def export_pdf(fig, subfolder, *parts, close=True):
        """Save `fig` to ml/figures/<subfolder>/<parts joined by '__'>.pdf.

        Empty/None parts are dropped. The figure is closed after saving so a bulk
        export does not accumulate open figures.
        """
        out_dir = os.path.join(FIGURES_DIR, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        name = "__".join(_sanitize(str(p)) for p in parts if p not in (None, ""))
        path = os.path.join(out_dir, f"{name}.pdf")
        try:
            fig.savefig(path, bbox_inches="tight")
        except Exception as e:  # pragma: no cover - env dependent
            print(f"NOTE: could not save PDF '{path}': {type(e).__name__}: {e}")
        finally:
            if close:
                plt.close(fig)
        return os.path.join(subfolder, f"{name}.pdf")

    def export_report(paths):
        """Markdown summary listing what a column export wrote."""
        if not paths:
            return mo.md("*Nothing to export — select at least one metric.*")
        listing = "\n".join(f"- `{p}`" for p in paths)
        return mo.md(f"**Exported {len(paths)} figure(s) to `ml/figures/`:**\n\n{listing}")

    return ARCH_ORDER_DEFAULT, arch_tag, export_pdf, export_report, metric_tag


@app.cell(hide_code=True)
def _(METRIC_SHORTNAMES, mo, raw_df):
    # ------------------------------------------------------------------
    # Per-plot metric selection utilities.
    # ------------------------------------------------------------------
    def build_metric_options(df):
        """Collect the plottable loss metrics present in the data.

        The train CE loss, every ``eval/lm/*/CE loss`` metric, and every
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

    def build_accuracy_options(df):
        """Accuracy metrics, taken from ``METRIC_SHORTNAMES``.

        These are not matched by the CE-loss patterns above, so they are pulled from
        the shortname table (which is the list of accuracy metrics we care about) and
        kept in its order. Only columns actually present in the data are offered.
        """
        cols = set(df.columns)
        return [
            c
            for c in METRIC_SHORTNAMES
            if "accuracy" in c.lower() and c in cols
        ]

    # Loss metrics only — used by the training-curve (history) dropdowns.
    HISTORY_METRIC_OPTIONS = build_metric_options(raw_df)
    # Loss + accuracy — used by the Group 1-5 dropdowns.
    METRIC_OPTIONS = HISTORY_METRIC_OPTIONS + [
        c for c in build_accuracy_options(raw_df) if c not in HISTORY_METRIC_OPTIONS
    ]

    def build_load_metric_options(df):
        """Collect the MoE router load metrics present in the data.

        Every column whose name ends in ``load imbalance`` or ``load balancing loss
        unscaled`` (the aggregate ``train/router 0 ...`` plus the per-block
        ``train/block NN/router 0 ...`` variants). Aggregate metrics are listed first.
        """
        cols = list(df.columns)
        agg, per_block = [], []
        for c in cols:
            cl = c.lower()
            if cl.endswith("load imbalance") or cl.endswith("load balancing loss unscaled"):
                (per_block if "/block " in c else agg).append(c)
        return agg + per_block

    LOAD_METRIC_OPTIONS = build_load_metric_options(raw_df)

    _DEFAULT_METRIC = (
        ["train/CE loss"] if "train/CE loss" in METRIC_OPTIONS else METRIC_OPTIONS[:1]
    )
    _DEFAULT_LOAD_METRIC = (
        ["train/router 0 load imbalance"]
        if "train/router 0 load imbalance" in LOAD_METRIC_OPTIONS
        else LOAD_METRIC_OPTIONS[:1]
    )

    def metric_multiselect(label, chosen_values=[]):
        """A per-plot multi-select over the loss + accuracy metrics.

        Used by the Group 1-5 columns.
        """
        return mo.ui.multiselect(
            options=METRIC_OPTIONS, value=(chosen_values if chosen_values else _DEFAULT_METRIC), label=label,
            # full_width=True,
        )

    def history_metric_multiselect(label, chosen_values=[]):
        """A per-plot multi-select for the training-curve (history) column.

        Loss metrics only — the accuracy metrics are offered on the Group 1-5
        dropdowns instead.
        """
        return mo.ui.multiselect(
            options=HISTORY_METRIC_OPTIONS,
            value=(chosen_values if chosen_values else _DEFAULT_METRIC),
            label=label,
        )

    def load_metric_multiselect(label, chosen_values=None):
        """A per-plot multi-select over the MoE router load metrics (load imbalance /
        load balancing loss unscaled)."""
        return mo.ui.multiselect(
            options=LOAD_METRIC_OPTIONS,
            value=chosen_values if chosen_values else _DEFAULT_LOAD_METRIC,
            label=label,
        )

    _ARCH_OPTIONS = ["dense", "moe32", "moe64"]

    def arch_multiselect(label, chosen_values=None, options=None):
        """A per-plot multi-select over the architectures (dense/moe32/moe64).

        Its ``.value`` is a list suitable to pass as ``include_archs``. Defaults to
        all architectures selected. Pass ``options`` to restrict the choices (e.g.
        MoE-only plots offer just ``["moe32", "moe64"]``).
        """
        opts = list(options) if options is not None else list(_ARCH_OPTIONS)
        return mo.ui.multiselect(
            options=opts,
            value=chosen_values if chosen_values is not None else list(opts),
            label=label,
        )

    # Model scales, smallest first. Used by the paper figures, where scale is the
    # facet dimension rather than a per-plot constant.
    SCALE_OPTIONS = ["10M", "80M", "200M", "1B"]

    def metric_dropdown(label, value=None):
        """Single-select over the metrics, for plots where each panel picks one."""
        return mo.ui.dropdown(
            options=METRIC_OPTIONS,
            value=value if value in METRIC_OPTIONS else METRIC_OPTIONS[0],
            label=label,
        )

    def scale_multiselect(label, chosen_values=None, options=None):
        """A multi-select over model scales, ordered smallest to largest."""
        opts = list(options) if options is not None else list(SCALE_OPTIONS)
        return mo.ui.multiselect(
            options=opts,
            value=chosen_values if chosen_values is not None else list(opts),
            label=label,
        )

    def render_side_by_side(metrics, build_fig, plot_key=None):
        """Render one figure per selected metric, laid out side by side.

        `build_fig(metric)` should return a matplotlib figure for a single metric.
        PDFs are *not* written here — use a column's "Export images" button, which
        saves into the organized ``ml/figures/<subfolder>/`` layout.
        """
        if not metrics:
            return mo.md("*Select at least one metric.*")
        figs = [build_fig(m) for m in metrics]
        return mo.hstack(figs, widths="equal", gap=1, wrap=True)

    def render_faceted(metrics, build_fig, plot_key=None):
        """Render a single faceted figure covering all selected metrics at once.

        `build_fig(metrics)` should return one matplotlib figure whose facets already
        span the metric list (e.g. metrics as rows, architectures as columns). PDFs
        are written by the column's "Export images" button, not here.
        """
        if not metrics:
            return mo.md("*Select at least one metric.*")
        return build_fig(metrics)

    return (
        arch_multiselect,
        history_metric_multiselect,
        load_metric_multiselect,
        metric_dropdown,
        metric_multiselect,
        render_faceted,
        render_side_by_side,
        scale_multiselect,
    )


@app.cell
def _():
    METRIC_SHORTNAMES = {
        "train/CE loss": "train_loss",
        "eval/lm/c4_en-validation/CE loss": "eval_c4_loss",
        "eval/lm/dolma_books-validation/CE loss": "eval_dolma_books_loss",
        "eval/lm/dolma_common-crawl-validation/CE loss": "eval_dolma_cc_loss",
        "eval/lm/dolma_pes2o-validation/CE loss": "eval_dolma_pes2o_loss", 
        "eval/lm/dolma_reddit-validation/CE loss": "eval_dolma_reddit_loss",
        "eval/lm/dolma_stack-validation/CE loss": "eval_dolma_stack_loss",
        "eval/lm/dolma_wiki-validation/CE loss": "eval_dolma_wiki_loss", 
        "eval/lm/ice-validation/CE loss": "eval_ice_loss",
        "eval/lm/m2d2_s2orc-validation/CE loss": "eval_s2orc_loss",
        "eval/lm/pile-validation/CE loss": "eval_pile_loss",
        "eval/lm/wikitext_103-validation/CE loss": "eval_wikitext_loss",
        "eval/downstream/hellaswag (CE loss)": "eval_hellaswag_loss",
        "eval/downstream/boolq (CE loss)": "eval_boolq_loss",
        "eval/downstream/mmlu_humanities_mc_5shot_test (CE loss)": "eval_mmlu_hum_loss",
        "eval/downstream/mmlu_other_mc_5shot_test (CE loss)": "eval_mmlu_oth_loss",
        "eval/downstream/mmlu_social_sciences_mc_5shot_test (CE loss)": "eval_mmlu_ss_loss",
        "eval/downstream/mmlu_stem_mc_5shot_test (CE loss)": "eval_mmlu_stem_loss",
        "eval/downstream/boolq (accuracy)": "eval_boolq_acc",
        "eval/downstream/hellaswag (length-normalized accuracy)": "eval_hellaswag_acc",
        "eval/downstream/mmlu_humanities_mc_5shot_test (length-normalized accuracy)": "eval_mmlu_hum_acc",
        "eval/downstream/mmlu_other_mc_5shot_test (length-normalized accuracy)": "eval_mmlu_oth_acc",
        "eval/downstream/mmlu_social_sciences_mc_5shot_test (length-normalized accuracy)": "eval_mmlu_ss_acc",
        "eval/downstream/mmlu_stem_mc_5shot_test (length-normalized accuracy)": "eval_mmlu_stem_acc",
        "train/router 0 load imbalance": "train_load_imbalance",
    }

    # Human-readable names for the data mixtures, used in figure titles.
    # Keyed by the identifiers the plotting code already passes to
    # `make_repetition_figure`: the `DATA_TAGS` values for Groups 1 & 2, and the
    # `data_label` strings for the Group 1b off-axis families.
    DATA_DISPLAY_NAMES = {
        # Data-source tags (DATA_TAGS).
        "olmo_mix": "OLMoE mix",
        "dclm": "DCLM",
        "starcoder": "StarCoder",
        "pes2o": "peS2o",
        "wiki": "Wikipedia",
        # Group 1b off-axis families — all the OLMoE mix, so the name carries the
        # axis that makes them non-comparable to the Group 1 runs.
        "olmo_mix (bs 131072)": "OLMoE mix (batch 131k)",
        "olmo_mix (lr 1e-2)": "OLMoE mix (lr 1e-2)",
        "olmo_mix (expert size)": "OLMoE mix (expert size)",
        "olmo_mix (expert count)": "OLMoE mix (expert count)",
    }

    # Short, human-readable metric names for axis and legend labels. Suffix
    # convention: "CE" for cross-entropy losses, "acc" for accuracies, so the two
    # downstream variants of the same task stay distinguishable.
    #
    # The per-block router metrics (`train/block NN/router 0 ...`, 30 of them and
    # layer-count dependent) are deliberately absent — they fall back to their raw
    # names rather than being enumerated here.
    METRIC_DISPLAY_NAMES = {
        "train/CE loss": "Train CE",
        # Held-out LM validation sets.
        "eval/lm/c4_en-validation/CE loss": "C4 CE",
        "eval/lm/dolma_books-validation/CE loss": "Books CE",
        "eval/lm/dolma_common-crawl-validation/CE loss": "Common Crawl CE",
        "eval/lm/dolma_pes2o-validation/CE loss": "peS2o CE",
        "eval/lm/dolma_reddit-validation/CE loss": "Reddit CE",
        "eval/lm/dolma_stack-validation/CE loss": "Stack CE",
        "eval/lm/dolma_wiki-validation/CE loss": "Wikipedia CE",
        "eval/lm/ice-validation/CE loss": "ICE CE",
        "eval/lm/m2d2_s2orc-validation/CE loss": "S2ORC CE",
        "eval/lm/pile-validation/CE loss": "Pile CE",
        "eval/lm/wikitext_103-validation/CE loss": "WikiText-103 CE",
        # Downstream tasks, CE loss.
        "eval/downstream/boolq (CE loss)": "BoolQ CE",
        "eval/downstream/hellaswag (CE loss)": "HellaSwag CE",
        "eval/downstream/mmlu_humanities_mc_5shot_test (CE loss)": "MMLU humanities CE",
        "eval/downstream/mmlu_other_mc_5shot_test (CE loss)": "MMLU other CE",
        "eval/downstream/mmlu_social_sciences_mc_5shot_test (CE loss)": "MMLU social sci CE",
        "eval/downstream/mmlu_stem_mc_5shot_test (CE loss)": "MMLU STEM CE",
        # Downstream tasks, accuracy.
        "eval/downstream/boolq (accuracy)": "BoolQ acc",
        "eval/downstream/hellaswag (length-normalized accuracy)": "HellaSwag acc",
        "eval/downstream/mmlu_humanities_mc_5shot_test (length-normalized accuracy)": "MMLU humanities acc",
        "eval/downstream/mmlu_other_mc_5shot_test (length-normalized accuracy)": "MMLU other acc",
        "eval/downstream/mmlu_social_sciences_mc_5shot_test (length-normalized accuracy)": "MMLU social sci acc",
        "eval/downstream/mmlu_stem_mc_5shot_test (length-normalized accuracy)": "MMLU STEM acc",
        # MoE router load metrics (aggregate).
        "train/router 0 load imbalance": "Router load imbalance",
        "train/router 0 load balancing loss unscaled": "Router LB loss",
    }

    # Short names for the swept regularizers, used as the Group 3/3b figure title
    # and as the legend's factor-section header — both tight on space, so these stay
    # brief. Keys are the `factor_label` values passed to `make_factor_figure`.
    FACTOR_DISPLAY_NAMES = {
        "dropout": "Dropout",
        "expert_dropout": "Expert Dropout",
        "weight_decay": "Weight Decay",
        "max_grad_norm": "Max Grad Norm",
        "jitter_eps": "Router Jitter",
        "eom_prob": "EOM",
        "fom_prob": "FOM",
    }

    # Human-readable architecture names, used for the "Model type" legend section.
    # Format is (num experts x expert size relative to the dense FFN).
    # famB source lists, shared by the Group 5 plots and the paper figures so the
    # two cannot drift apart. Tuple is (legend label, legacy color, kind, run key);
    # the color is ignored -- `factor_palette` assigns by position.
    FAMB_SC_SOURCES = [
        ("dclm", "#1f77b4", "hline", "dclm"),
        ("10% StarCoder", "#d62728", "famB", "sc90"),
        ("50% StarCoder", "#ff7f0e", "famB", "sc50"),
        ("100% StarCoder", "#2ca02c", "baseline", "starcoder"),
    ]
    FAMB_P2O_SOURCES = [
        ("dclm", "#1f77b4", "hline", "dclm"),
        ("10% peS2o", "#9467bd", "famB", "p2o10"),
        ("50% peS2o", "#ff7f0e", "famB", "p2o50"),
        ("100% peS2o", "#2ca02c", "baseline", "pes2o"),
    ]
    FAMB_SC_TITLE = "% StarCoder"
    FAMB_P2O_TITLE = "% peS2o"

    MODEL_DISPLAY_NAME = {
        "dense": "Dense (1 x 1)",
        "moe16": "MoE (16 x 1/4)",
        "moe32": "MoE (32 x 1/4)",
        "moe64": "MoE (64 x 1/4)",
        "moe128": "MoE (128 x 1/4)",
        "moe64s8": "MoE (64 x 1/8)",
        "moe64s32": "MoE (64 x 1/2)",
    }
    return (
        DATA_DISPLAY_NAMES,
        FACTOR_DISPLAY_NAMES,
        FAMB_P2O_SOURCES,
        FAMB_P2O_TITLE,
        FAMB_SC_SOURCES,
        FAMB_SC_TITLE,
        METRIC_DISPLAY_NAMES,
        METRIC_SHORTNAMES,
        MODEL_DISPLAY_NAME,
    )


@app.cell(column=1, hide_code=True)
def _(mo):
    export_btn_g12 = mo.ui.run_button(label="Export images")
    log_y_g12 = mo.ui.checkbox(label="Log y axis")
    mo.vstack([
        mo.md(r"""
    ## Group 1 & 2 — Final loss vs. repetition (baseline runs)

    For each data mixture we plot Dense, MoE32 and MoE64 **baseline** runs (tag
    `baseline`, i.e. no dropout/EOM/FOM/jitter). Metric on Y, repetition count on a
    log X axis. If more than one baseline run exists at the same repetition count, a
    warning is printed and the **lowest** value is kept.

    Group 1 covers the `olmo_mix` mixture at 10M, 80M, 200M and 1B.

    The 10M figure excludes the small-batch (131072-token) runs, which are plotted
    separately in Group 1b.

    > **10M caveat.** The separate `lr=1e-2` 10M sweep (`data_rep_AC_lr1e-2`) is
    > also tagged `baseline`, and at 10M it reaches ~6.5 train CE against ~9.3 for
    > the `lr=4e-4` runs plotted here. It is split out into `high_lr_df` and shown
    > in Group 1b, so it cannot reach this figure. One visible consequence: the 10M
    > dense series has **no rep=1 point**, because the only 10M dense rep1x run
    > belongs to that sweep.

    > **1B is wired but empty.** The sweep is still in flight, and separately the 1B
    > runs launched so far carry tags like `1B, Data=1C, MoE64, olmo_mix, rep1x` with
    > **no `baseline` tag**, which is what `collect_baseline_points` selects on. Tag
    > them `baseline` at launch (as the 80M/200M runs are) or the figure stays blank
    > even after they finish.

    **Export images** writes Group 1 to `ml/figures/olmoe_mix/` and Group 2 to
    `ml/figures/single_domain/`.
    """),
        mo.hstack([export_btn_g12, log_y_g12], justify="start", gap=1),
    ])
    return export_btn_g12, log_y_g12


@app.cell(hide_code=True)
def _(
    arch_tag,
    export_btn_g12,
    export_pdf,
    export_report,
    finished_df,
    log_y_g12,
    make_repetition_figure,
    metric_tag,
    mo,
    sel_dclm_80m,
    sel_dclm_80m_arch,
    sel_olmo_10m,
    sel_olmo_10m_arch,
    sel_olmo_1b,
    sel_olmo_1b_arch,
    sel_olmo_200m,
    sel_olmo_200m_arch,
    sel_olmo_80m,
    sel_olmo_80m_arch,
    sel_pes2o_80m,
    sel_pes2o_80m_arch,
    sel_starcoder_80m,
    sel_starcoder_80m_arch,
    sel_wiki_80m,
    sel_wiki_80m_arch,
):
    # Export every Group 1 & 2 plot. Group 1 (the full olmo mix) goes to
    # olmoe_mix/, the single-source mixtures of Group 2 to single_domain/.
    # Name: <data source>__<scale>__<architectures>__<metric>.
    if not export_btn_g12.value:
        _out = mo.md("*Click **Export images** to write this column's PDFs.*")
    else:
        _specs = [
            ("olmoe_mix", "olmo_mix", "10M", sel_olmo_10m, sel_olmo_10m_arch),
            ("olmoe_mix", "olmo_mix", "80M", sel_olmo_80m, sel_olmo_80m_arch),
            ("olmoe_mix", "olmo_mix", "200M", sel_olmo_200m, sel_olmo_200m_arch),
            ("olmoe_mix", "olmo_mix", "1B", sel_olmo_1b, sel_olmo_1b_arch),
            ("single_domain", "dclm", "80M", sel_dclm_80m, sel_dclm_80m_arch),
            ("single_domain", "starcoder", "80M", sel_starcoder_80m, sel_starcoder_80m_arch),
            ("single_domain", "pes2o", "80M", sel_pes2o_80m, sel_pes2o_80m_arch),
            ("single_domain", "wiki", "80M", sel_wiki_80m, sel_wiki_80m_arch),
        ]
        _paths = []
        for _sub, _tag, _scale, _sel, _arch in _specs:
            # One separate image per selected metric, always suffixed with the
            # short metric name.
            for _m in _sel.value:
                _fig = make_repetition_figure(
                    finished_df, _tag, _scale, _m, include_archs=_arch.value,
                    log_y=log_y_g12.value,
                )
                _paths.append(
                    export_pdf(
                        _fig, _sub, _tag, _scale, arch_tag(_arch.value),
                        metric_tag(_m),
                    )
                )
        _out = export_report(_paths)
    _out
    return


@app.cell(hide_code=True)
def _(
    ARCH_COLOR,
    ARCH_ORDER_DEFAULT,
    DATA_DISPLAY_NAMES,
    apply_log_yaxis,
    arch_dash,
    arch_factor_legend,
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
        df: pd.DataFrame,
        data_tag: str,
        scale: str,
        model_type: str,
        metric: str,
        series_tag: str = "baseline",
    ):
        """Return sorted (reps, value) points for one series.

        ``series_tag`` is the tag that selects the run family: ``"baseline"`` for
        Groups 1 & 2, ``"Data=4C"`` for the Group 1b (4C token budget) family.

        ``data_tag`` of ``None`` skips the data-mixture filter entirely. That is
        needed for the 10M sweeps: they are all ``olmo_mix`` in substance, but most
        of them (and *all* of the small-batch ones) were launched before the
        ``olmo_mix`` tag was applied, so filtering on it would drop them. No 10M run
        carries any other data-source tag, so nothing foreign can slip in.

        Prints a warning when multiple runs share a repetition count, and keeps the
        run with the lowest metric value.
        """
        rows = []
        for _, row in df.iterrows():
            if get_scale(row) != scale:
                continue
            if get_model_type(row) != model_type:
                continue
            if not has_tag(row, series_tag):
                continue
            if data_tag is not None and not has_tag(row, data_tag):
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
                    f"WARNING: {len(entries)} {series_tag} runs for "
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
        series_tag="baseline",
        series_label=None,
        data_label=None,
        log_y=False,
        arch_order=None,
    ):
        """One figure: Dense + MoE32 + MoE64 baseline, metric vs reps (log X).

        Architecture is the only variable here, so it drives color (dense = purple,
        moe32 = greenish-blue, moe64 = yellow, i.e. the viridis palette used by the
        factor plots) as well as line style (solid/dashed/dotted).

        Set ``log_y=True`` for a log-scaled y axis (X is always log/power-of-2).

        Filters (all default to no-op):

        - ``include_archs`` / ``exclude_archs``: keep/drop architectures
          (``"dense"``, ``"moe32"``, ``"moe64"``).
        - ``include_reps`` / ``exclude_reps``: keep/drop repetition counts.

        ``series_tag`` selects which run family to plot (``"baseline"`` for Groups
        1 & 2, ``"Data=4C"`` for Group 1b); ``series_label`` overrides how that
        family is named in the title. ``data_tag=None`` disables the mixture filter
        (see ``collect_baseline_points``); ``data_label`` then supplies the title
        text in its place.
        """
        label = series_label if series_label is not None else series_tag
        mix_label = DATA_DISPLAY_NAMES.get(
            data_label, DATA_DISPLAY_NAMES.get(data_tag, data_tag or "all data")
        )
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        any_data = False
        max_reps = 0
        shown_archs = []
        for model_type in arch_order or ARCH_ORDER_DEFAULT:
            if not keep_arch(model_type, include_archs, exclude_archs):
                continue
            points = collect_baseline_points(
                df, data_tag, scale, model_type, metric, series_tag=series_tag
            )
            points = filter_rep_points(points, include_reps, exclude_reps)
            if not points:
                continue
            any_data = True
            shown_archs.append(model_type)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            max_reps = max(max_reps, max(xs))
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=4,
                linewidth=1.5,
                linestyle=arch_dash(model_type),
                color=ARCH_COLOR[model_type],
            )
        # title = f"{mix_label} — {scale}"
        # title = f"{mix_label} — {scale}"
        # title = f"{mix_label} — {scale}\n{metric} vs. repetition ({label})"
        # if not any_data:
        #     title += f"  [no finished {label} runs]"
        apply_pow2_xaxis(ax, max_reps, any_data)
        if log_y:
            apply_log_yaxis(ax)
        ax.set_xlabel("Repetition Rate")
        ax.set_ylabel(axis_label(metric))
        # ax.set_title(title, fontsize=10)
        if any_data:
            arch_factor_legend(
                ax, shown_archs, arch_colored=True, arch_order=arch_order
            )
        fig.tight_layout()
        return fig

    return apply_pow2_xaxis, collect_baseline_points, make_repetition_figure


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_olmo_10m = metric_multiselect("olmo_mix 10M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_olmo_10m_arch = arch_multiselect("olmo_mix 10M — architectures")
    mo.vstack([sel_olmo_10m, sel_olmo_10m_arch])
    return sel_olmo_10m, sel_olmo_10m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_g12,
    make_repetition_figure,
    render_side_by_side,
    sel_olmo_10m,
    sel_olmo_10m_arch,
):
    # 10M, regular batch (1048576). Same call shape as the other Group 1 plots: the
    # `olmo_mix` tag filter is what keeps the separate `lr1e-2` 10M sweep (also
    # tagged `baseline`, ~2.8 nats better) out of this series. The small-batch
    # (131072) runs are split out into Group 1b.
    _out = render_side_by_side(
        sel_olmo_10m.value,
        lambda m: make_repetition_figure(
            finished_df, "olmo_mix", "10M", m, include_archs=sel_olmo_10m_arch.value
        , log_y=log_y_g12.value
        ),
        "olmo_mix_10M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_olmo_80m = metric_multiselect("olmo_mix 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_olmo_80m_arch = arch_multiselect("olmo_mix 80M — architectures")
    mo.vstack([sel_olmo_80m, sel_olmo_80m_arch])
    return sel_olmo_80m, sel_olmo_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_g12,
    make_repetition_figure,
    render_side_by_side,
    sel_olmo_80m,
    sel_olmo_80m_arch,
):
    _out = render_side_by_side(
        sel_olmo_80m.value,
        lambda m: make_repetition_figure(
            finished_df, "olmo_mix", "80M", m, include_archs=sel_olmo_80m_arch.value
        , log_y=log_y_g12.value
        ),
        "olmo_mix_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_olmo_200m = metric_multiselect("olmo_mix 200M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_olmo_200m_arch = arch_multiselect("olmo_mix 200M — architectures")
    mo.vstack([sel_olmo_200m, sel_olmo_200m_arch])
    return sel_olmo_200m, sel_olmo_200m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_g12,
    make_repetition_figure,
    render_side_by_side,
    sel_olmo_200m,
    sel_olmo_200m_arch,
):
    _out = render_side_by_side(
        sel_olmo_200m.value,
        lambda m: make_repetition_figure(
            finished_df, "olmo_mix", "200M", m, include_archs=sel_olmo_200m_arch.value
        , log_y=log_y_g12.value
        ),
        "olmo_mix_200M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_olmo_1b = metric_multiselect("olmo_mix 1B — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_olmo_1b_arch = arch_multiselect("olmo_mix 1B — architectures")
    mo.vstack([sel_olmo_1b, sel_olmo_1b_arch])
    return sel_olmo_1b, sel_olmo_1b_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_g12,
    make_repetition_figure,
    render_side_by_side,
    sel_olmo_1b,
    sel_olmo_1b_arch,
):
    # 1B sweep. Stays empty until the runs finish AND carry the `baseline` tag —
    # see the note in the Group 1 & 2 header.
    _out = render_side_by_side(
        sel_olmo_1b.value,
        lambda m: make_repetition_figure(
            finished_df, "olmo_mix", "1B", m, include_archs=sel_olmo_1b_arch.value
        , log_y=log_y_g12.value
        ),
        "olmo_mix_1B",
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
def _(arch_multiselect, metric_multiselect, mo):
    sel_dclm_80m = metric_multiselect("dclm 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_dclm_80m_arch = arch_multiselect("dclm 80M — architectures")
    mo.vstack([sel_dclm_80m, sel_dclm_80m_arch])
    return sel_dclm_80m, sel_dclm_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_g12,
    make_repetition_figure,
    render_side_by_side,
    sel_dclm_80m,
    sel_dclm_80m_arch,
):
    _out = render_side_by_side(
        sel_dclm_80m.value,
        lambda m: make_repetition_figure(
            finished_df, "dclm", "80M", m, include_archs=sel_dclm_80m_arch.value
        , log_y=log_y_g12.value
        ),
        "dclm_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_starcoder_80m = metric_multiselect("starcoder 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_stack-validation/CE loss"])
    sel_starcoder_80m_arch = arch_multiselect("starcoder 80M — architectures")
    mo.vstack([sel_starcoder_80m, sel_starcoder_80m_arch])
    return sel_starcoder_80m, sel_starcoder_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_g12,
    make_repetition_figure,
    render_side_by_side,
    sel_starcoder_80m,
    sel_starcoder_80m_arch,
):
    _out = render_side_by_side(
        sel_starcoder_80m.value,
        lambda m: make_repetition_figure(
            finished_df, "starcoder", "80M", m, include_archs=sel_starcoder_80m_arch.value
        , log_y=log_y_g12.value
        ),
        "starcoder_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_pes2o_80m = metric_multiselect("pes2o 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_pes2o-validation/CE loss"])
    sel_pes2o_80m_arch = arch_multiselect("pes2o 80M — architectures")
    mo.vstack([sel_pes2o_80m, sel_pes2o_80m_arch])
    return sel_pes2o_80m, sel_pes2o_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_g12,
    make_repetition_figure,
    render_side_by_side,
    sel_pes2o_80m,
    sel_pes2o_80m_arch,
):
    _out = render_side_by_side(
        sel_pes2o_80m.value,
        lambda m: make_repetition_figure(
            finished_df, "pes2o", "80M", m, include_archs=sel_pes2o_80m_arch.value
        , log_y=log_y_g12.value
        ),
        "pes2o_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_wiki_80m = metric_multiselect("wiki 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_wiki-validation/CE loss", "eval/lm/wikitext_103-validation/CE loss"])
    sel_wiki_80m_arch = arch_multiselect("wiki 80M — architectures")
    mo.vstack([sel_wiki_80m, sel_wiki_80m_arch])
    return sel_wiki_80m, sel_wiki_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_g12,
    make_repetition_figure,
    render_side_by_side,
    sel_wiki_80m,
    sel_wiki_80m_arch,
):
    _out = render_side_by_side(
        sel_wiki_80m.value,
        lambda m: make_repetition_figure(
            finished_df, "wiki", "80M", m, include_archs=sel_wiki_80m_arch.value
        , log_y=log_y_g12.value
        ),
        "wiki_80M",
    )
    _out
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    export_btn_g1b = mo.ui.run_button(label="Export images")
    log_y_g1b = mo.ui.checkbox(label="Log y axis")
    mo.vstack([
        mo.md(r"""
    ## Group 1b — Off-axis run families

    Runs that share Group 1's axes but differ in a way that would make them
    misleading if drawn on the same plot. Each is held in its own frame and
    excluded from every other group.

    **4C token budget** — `data_rep_AC_x4` in the run name, tagged `Data=4C`
    (frame: `x4_df`). Same `olmo_mix` data as the 80M baselines but 4x the token
    budget (6.4B vs. 1.6B), so it cannot share the baseline axis. Only dense and
    MoE64 were run. Uses `series_tag="Data=4C"` in place of `baseline`.

    **Small batch (bs 131072)** — the `new_bs_256plus` 10M sweep re-ran the extreme
    repetition counts (256x / 512x / 1024x) at a 131072-token global batch instead
    of the usual 1048576 (frame: `small_bs_df`). A different batch size is a
    different experiment, so these are kept off the regular-batch 10M plot in
    Group 1. This plot does not filter on the `olmo_mix` tag — none of these runs
    carry it, and the frame is homogeneous, so nothing foreign can slip in.

    **High LR (lr 1e-2)** — a 10M sweep at `lr=1e-2` (`data_rep_AC_lr1e-2`) against
    the `4e-4` used by every other run in the project (frame: `high_lr_df`). Reps
    1x / 8x / 32x, all three architectures. At 10M this reaches ~6.5 train CE vs
    ~9.3 for `4e-4`, so plotting it alongside would flatten every other curve.
    Same `data_tag=None` treatment and the same reasoning.

    **Export images** writes this column to `ml/figures/olmoe_mix_4C/`,
    `ml/figures/olmoe_mix_bs131072/` and `ml/figures/olmoe_mix_lr1e-2/`.
    """),
        mo.hstack([export_btn_g1b, log_y_g1b], justify="start", gap=1),
    ])
    return export_btn_g1b, log_y_g1b


@app.cell(hide_code=True)
def _(
    X4_TAG,
    arch_tag,
    expert_size_df,
    export_btn_g1b,
    export_pdf,
    export_report,
    finished_df,
    high_lr_df,
    log_y_g1b,
    make_repetition_figure,
    metric_tag,
    mo,
    sel_expcount_80m,
    sel_expcount_80m_arch,
    sel_expsize_80m,
    sel_expsize_80m_arch,
    sel_highlr_10m,
    sel_highlr_10m_arch,
    sel_smallbs_10m,
    sel_smallbs_10m_arch,
    sel_x4_80m,
    sel_x4_80m_arch,
    small_bs_df,
    x4_df,
):
    # Export the Group 1b plots: the 4C family to olmoe_mix_4C/ and the small-batch
    # 10M family to olmoe_mix_bs131072/.
    # Name: <data source>__<scale>__<architectures>__<metric>.
    if not export_btn_g1b.value:
        _out = mo.md("*Click **Export images** to write this column's PDFs.*")
    else:
        _paths = []
        for _m in sel_x4_80m.value:
            _fig = make_repetition_figure(
                x4_df,
                "olmo_mix",
                "80M",
                _m,
                include_archs=sel_x4_80m_arch.value,
                series_tag=X4_TAG,
             log_y=log_y_g1b.value
            )
            _paths.append(
                export_pdf(
                    _fig, "olmoe_mix_4C", "olmo_mix_4C", "80M",
                    arch_tag(sel_x4_80m_arch.value), metric_tag(_m),
                )
            )
        for _m in sel_smallbs_10m.value:
            _fig = make_repetition_figure(
                small_bs_df,
                None,
                "10M",
                _m,
                include_archs=sel_smallbs_10m_arch.value,
                data_label="olmo_mix (bs 131072)",
             log_y=log_y_g1b.value
            )
            _paths.append(
                export_pdf(
                    _fig, "olmoe_mix_bs131072", "olmo_mix_bs131072", "10M",
                    arch_tag(sel_smallbs_10m_arch.value), metric_tag(_m),
                )
            )
        for _m in sel_highlr_10m.value:
            _fig = make_repetition_figure(
                high_lr_df,
                None,
                "10M",
                _m,
                include_archs=sel_highlr_10m_arch.value,
                data_label="olmo_mix (lr 1e-2)",
             log_y=log_y_g1b.value
            )
            _paths.append(
                export_pdf(
                    _fig, "olmoe_mix_lr1e-2", "olmo_mix_lr1e-2", "10M",
                    arch_tag(sel_highlr_10m_arch.value), metric_tag(_m),
                )
            )
        for _m in sel_expsize_80m.value:
            _fig = make_repetition_figure(
                expert_size_df,
                None,
                "80M",
                _m,
                include_archs=sel_expsize_80m_arch.value,
                include_reps=[1, 2, 4, 8, 16, 32],
                data_label="olmo_mix (expert size)",
                arch_order=["dense", "moe64s8", "moe64", "moe64s32"],
                log_y=log_y_g1b.value,
            )
            _paths.append(
                export_pdf(
                    _fig, "olmoe_mix_expert_size", "olmo_mix_expert_size", "80M",
                    arch_tag(sel_expsize_80m_arch.value), metric_tag(_m),
                )
            )
        for _m in sel_expcount_80m.value:
            _fig = make_repetition_figure(
                finished_df,
                "olmo_mix",
                "80M",
                _m,
                include_archs=sel_expcount_80m_arch.value,
                include_reps=[1, 2, 4, 8, 16, 32],
                data_label="olmo_mix (expert count)",
                arch_order=["dense", "moe16", "moe32", "moe64", "moe128"],
                log_y=log_y_g1b.value,
            )
            _paths.append(
                export_pdf(
                    _fig, "olmoe_mix_expert_count", "olmo_mix_expert_count", "80M",
                    arch_tag(sel_expcount_80m_arch.value), metric_tag(_m),
                )
            )
        _out = export_report(_paths)
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_x4_80m = metric_multiselect("4C olmo_mix 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_x4_80m_arch = arch_multiselect(
        "4C olmo_mix 80M — architectures", options=["dense", "moe64"]
    )
    mo.vstack([sel_x4_80m, sel_x4_80m_arch])
    return sel_x4_80m, sel_x4_80m_arch


@app.cell(hide_code=True)
def _(
    X4_TAG,
    log_y_g1b,
    make_repetition_figure,
    render_side_by_side,
    sel_x4_80m,
    sel_x4_80m_arch,
    x4_df,
):
    _out = render_side_by_side(
        sel_x4_80m.value,
        lambda m: make_repetition_figure(
            x4_df,
            "olmo_mix",
            "80M",
            m,
            include_archs=sel_x4_80m_arch.value,
            series_tag=X4_TAG,
         log_y=log_y_g1b.value
        ),
        "olmo_mix_80M_4C",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_highlr_10m = metric_multiselect("10M lr=1e-2 — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_highlr_10m_arch = arch_multiselect("10M lr=1e-2 — architectures")
    mo.vstack([sel_highlr_10m, sel_highlr_10m_arch])
    return sel_highlr_10m, sel_highlr_10m_arch


@app.cell(hide_code=True)
def _(
    high_lr_df,
    log_y_g1b,
    make_repetition_figure,
    render_side_by_side,
    sel_highlr_10m,
    sel_highlr_10m_arch,
):
    # The `data_rep_AC_lr1e-2` 10M sweep. Same figure as the Group 1 10M plot, over
    # `high_lr_df` instead of `finished_df`. Reps 1x / 8x / 32x only.
    _out = render_side_by_side(
        sel_highlr_10m.value,
        lambda m: make_repetition_figure(
            high_lr_df,
            None,
            "10M",
            m,
            include_archs=sel_highlr_10m_arch.value,
            data_label="olmo_mix (lr 1e-2)",
         log_y=log_y_g1b.value
        ),
        "olmo_mix_10M_lr1e-2",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_smallbs_10m = metric_multiselect("10M bs=131072 — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_smallbs_10m_arch = arch_multiselect("10M bs=131072 — architectures")
    mo.vstack([sel_smallbs_10m, sel_smallbs_10m_arch])
    return sel_smallbs_10m, sel_smallbs_10m_arch


@app.cell(hide_code=True)
def _(
    log_y_g1b,
    make_repetition_figure,
    render_side_by_side,
    sel_smallbs_10m,
    sel_smallbs_10m_arch,
    small_bs_df,
):
    # The `new_bs_256plus` 10M runs at a 131072-token global batch. Same figure as
    # the Group 1 10M plot, over `small_bs_df` instead of `finished_df`.
    _out = render_side_by_side(
        sel_smallbs_10m.value,
        lambda m: make_repetition_figure(
            small_bs_df,
            None,
            "10M",
            m,
            include_archs=sel_smallbs_10m_arch.value,
            data_label="olmo_mix (bs 131072)",
         log_y=log_y_g1b.value
        ),
        "olmo_mix_10M_bs131072",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_expsize_80m = metric_multiselect(
        "80M expert size — metrics",
        chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"],
    )
    # Only the architectures this comparison is about, all selected by default and
    # listed in the order they are plotted (by expert size: 1, 1/8, 1/4, 1/2).
    sel_expsize_80m_arch = arch_multiselect(
        "80M expert size — architectures",
        options=["dense", "moe64s8", "moe64", "moe64s32"],
    )
    mo.vstack([sel_expsize_80m, sel_expsize_80m_arch])
    return sel_expsize_80m, sel_expsize_80m_arch


@app.cell(hide_code=True)
def _(
    expert_size_df,
    log_y_g1b,
    make_repetition_figure,
    render_side_by_side,
    sel_expsize_80m,
    sel_expsize_80m_arch,
):
    # 80M expert-size sweep: the 64-expert model at 1/8, 1/4 and 1/2 the dense FFN,
    # against dense. Same figure as the Group 1 80M plot, over `expert_size_df`.
    # The MoE64s* runs only go to rep 32, so the axis is capped there rather than
    # running out to 256 with only dense/moe64 present.
    _out = render_side_by_side(
        sel_expsize_80m.value,
        lambda m: make_repetition_figure(
            expert_size_df,
            None,
            "80M",
            m,
            include_archs=sel_expsize_80m_arch.value,
            include_reps=[1, 2, 4, 8, 16, 32],
            data_label="olmo_mix (expert size)",
            arch_order=["dense", "moe64s8", "moe64", "moe64s32"],
            log_y=log_y_g1b.value,
        ),
        "olmo_mix_80M_expert_size",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_expcount_80m = metric_multiselect(
        "80M expert count — metrics",
        chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"],
    )
    sel_expcount_80m_arch = arch_multiselect(
        "80M expert count — architectures",
        options=["dense", "moe16", "moe32", "moe64", "moe128"],
    )
    mo.vstack([sel_expcount_80m, sel_expcount_80m_arch])
    return sel_expcount_80m, sel_expcount_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_g1b,
    make_repetition_figure,
    render_side_by_side,
    sel_expcount_80m,
    sel_expcount_80m_arch,
):
    # 80M expert-count sweep: 16 / 32 / 64 / 128 experts against dense, all at 1/4
    # the dense FFN. Unlike the expert-size plot this runs over `finished_df` with
    # the normal mixture filter -- these runs are tagged `olmo_mix`.
    _out = render_side_by_side(
        sel_expcount_80m.value,
        lambda m: make_repetition_figure(
            finished_df,
            "olmo_mix",
            "80M",
            m,
            include_reps=[1, 2, 4, 8, 16, 32],
            include_archs=sel_expcount_80m_arch.value,
            data_label="olmo_mix (expert count)",
            arch_order=["dense", "moe16", "moe32", "moe64", "moe128"],
            log_y=log_y_g1b.value,
        ),
        "olmo_mix_80M_expert_count",
    )
    _out
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    export_btn_g3 = mo.ui.run_button(label="Export images")
    log_y_g3 = mo.ui.checkbox(label="Log y axis")
    mo.vstack([
        mo.md(r"""
    ## Group 3 — Regularizer sweeps: dropout, weight decay & max grad norm

    Model type is encoded as **color** (blue = dense, green = moe32, red = moe64).
    Baseline vs. each regularizer value is encoded as **line style**.
    X axis is the repetition count (log); Y is the chosen metric. Runs are restricted
    to the `olmo_mix` mixture at 80M scale (where the regularizer sweeps live). Note
    the sweeps currently only cover dense and moe64. For max grad norm, the baseline
    is the default `1.0`; a `null` (no clipping) setting is shown as its own series.

    **Export images** writes this column (Groups 3 and 3b) to
    `ml/figures/regularizers/`.
    """),
        mo.hstack([export_btn_g3, log_y_g3], justify="start", gap=1),
    ])
    return export_btn_g3, log_y_g3


@app.cell(hide_code=True)
def _(
    DROPOUT_COL,
    EXPERT_DROPOUT_COL,
    MGN_COL,
    WD_COL,
    arch_tag,
    export_btn_g3,
    export_pdf,
    export_report,
    finished_df,
    fom_prob_value,
    log_y_g3,
    make_factor_figure,
    metric_tag,
    mo,
    router_field,
    sel_dropout,
    sel_dropout_arch,
    sel_max_grad_norm,
    sel_max_grad_norm_arch,
    sel_moe_eom,
    sel_moe_eom_arch,
    sel_moe_expert_dropout,
    sel_moe_expert_dropout_arch,
    sel_moe_fom,
    sel_moe_fom_arch,
    sel_moe_jitter,
    sel_moe_jitter_arch,
    sel_weight_decay,
    sel_weight_decay_arch,
):
    # Export every Group 3 / 3b regularizer sweep to regularizers/.
    # Name: olmo_mix__80M__<architectures>__<varied feature>__<metric>.
    if not export_btn_g3.value:
        _out = mo.md("*Click **Export images** to write this column's PDFs.*")
    else:
        # (varied feature, selector, arch selector, extra make_factor_figure kwargs)
        _specs = [
            ("dropout", sel_dropout, sel_dropout_arch,
             dict(factor_col=DROPOUT_COL, baseline_value=None, factor_key="dropout",
                  baseline_label="None (disabled)")),
            ("weight_decay", sel_weight_decay, sel_weight_decay_arch,
             dict(factor_col=WD_COL, baseline_value=0.1, factor_key="weight_decay")),
            ("max_grad_norm", sel_max_grad_norm, sel_max_grad_norm_arch,
             dict(factor_col=MGN_COL, baseline_value=1.0, nan_label="0 (no clip)",
                  factor_key="max_grad_norm")),
            ("jitter_eps", sel_moe_jitter, sel_moe_jitter_arch,
             dict(factor_col=None, baseline_value=None, factor_key="jitter_eps",
                  value_getter=lambda row: router_field(row, "jitter_eps"))),
            ("eom_prob", sel_moe_eom, sel_moe_eom_arch,
             dict(factor_col=None, baseline_value=None, factor_key="eom_prob",
                  value_getter=lambda row: router_field(row, "eom_prob"))),
            ("fom_prob", sel_moe_fom, sel_moe_fom_arch,
             dict(factor_col=None, baseline_value=None, factor_key="fom_prob",
                  value_getter=fom_prob_value)),
            ("expert_dropout", sel_moe_expert_dropout, sel_moe_expert_dropout_arch,
             dict(factor_col=EXPERT_DROPOUT_COL, baseline_value=None,
                  factor_key="expert_dropout")),
        ]
        _paths = []
        for _feature, _sel, _arch, _kw in _specs:
            # One separate image per selected metric, always suffixed with the
            # short metric name.
            for _m in _sel.value:
                _fig = make_factor_figure(
                    finished_df,
                    _kw["factor_col"],
                    _feature,
                    _kw["baseline_value"],
                    _m,
                    include_archs=_arch.value,
                    exclude_reps=[128, 256, 512, 1024],
                    nan_label=_kw.get("nan_label"),
                    baseline_label=_kw.get("baseline_label"),
                    value_getter=_kw.get("value_getter"),
                    factor_key=_kw["factor_key"],
                    log_y=log_y_g3.value,
                )
                _paths.append(
                    export_pdf(
                        _fig, "regularizers", "olmo_mix", "80M",
                        arch_tag(_arch.value), _feature, metric_tag(_m),
                    )
                )
        _out = export_report(_paths)
    _out
    return


@app.cell(hide_code=True)
def _(
    DROPOUT_COL,
    EXPERT_DROPOUT_COL,
    FACTOR_DISPLAY_NAMES,
    METRIC_DISPLAY_NAMES,
    MGN_COL,
    WD_COL,
    apply_log_yaxis,
    apply_pow2_xaxis,
    arch_color,
    arch_dash,
    arch_factor_legend,
    axis_label,
    factor_palette,
    filter_rep_points,
    fom_prob_value,
    get_model_type,
    get_reps,
    get_scale,
    has_tag,
    keep_arch,
    metric_value,
    pd,
    plt,
    router_field,
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
        raw_factor_label,
        baseline_value,
        metric,
        include_archs=None,
        exclude_archs=None,
        include_reps=None,
        exclude_reps=None,
        nan_label=None,
        value_getter=None,
        factor_key=None,
        log_y=False,
        baseline_label=None,
    ):
        """color -> factor value; line style + shade -> architecture.

        Set ``log_y=True`` for a log-scaled y axis (X is always log/power-of-2).

        `baseline_value` is the value that represents 'no regularization' for this
        column (e.g. dropout baseline = None/blank, weight-decay baseline = 0.1).

        `nan_label`: when set, runs whose factor value is NaN/blank are bucketed under
        this label (a real setting, e.g. max_grad_norm=null -> "no clip") instead of
        being merged into "baseline". When None (the default), NaN counts as baseline.

        `value_getter`: optional ``callable(row) -> raw value`` used to read the factor
        value from a row. Defaults to ``row.get(factor_col)``. Use this for values
        nested inside JSON columns (e.g. MoE ``jitter_eps`` / ``eom_prob`` inside
        ``routers_list``).

        `baseline_label`: overrides the legend text for the baseline bucket. Useful
        where the numeric value is misleading — a dropout baseline is not really
        "0.0", it is the knob being switched off.

        `factor_key`: which regularizer this plot sweeps (one of ``"dropout"``,
        ``"expert_dropout"``, ``"weight_decay"``, ``"max_grad_norm"``,
        ``"fom_prob"``, ``"jitter_eps"``,
        ``"eom_prob"``). Only runs whose *other* knobs are all at their defaults are
        included, so e.g. a dropout=0.2 run does not contaminate the weight-decay
        plot's baseline bucket. When None, no such filtering is applied.

        Filters (all default to no-op):

        - ``include_archs`` / ``exclude_archs``: keep/drop architectures.
        - ``include_reps`` / ``exclude_reps``: keep/drop repetition counts.
        """
        if value_getter is None:
            value_getter = lambda row: row.get(factor_col)

        # Each regularizer knob: (reader, is-at-default predicate). A run is a valid
        # baseline / pure single-factor sweep only when every knob EXCEPT the one this
        # plot sweeps (`factor_key`) is at its default. Otherwise e.g. the dropout and
        # jitter experiment runs (which leave weight_decay at its 0.1 default) would
        # leak into the weight-decay plot's baseline bucket and, via the per-rep min,
        # drag the baseline line below its true value.
        def _blank(v):
            return v is None or (isinstance(v, float) and pd.isna(v)) or (
                isinstance(v, str) and v.strip() == ""
            )

        def _num_default(v, default):
            if _blank(v):
                return True
            try:
                return float(v) == default
            except (TypeError, ValueError):
                return False

        _knob_at_default = {
            "dropout": lambda row: _blank(row.get(DROPOUT_COL)),
            "expert_dropout": lambda row: _blank(row.get(EXPERT_DROPOUT_COL)),
            "weight_decay": lambda row: _num_default(row.get(WD_COL), 0.1),
            "max_grad_norm": lambda row: _num_default(row.get(MGN_COL), 1.0),
            "fom_prob": lambda row: _blank(fom_prob_value(row)),
            "jitter_eps": lambda row: _blank(router_field(row, "jitter_eps")),
            "eom_prob": lambda row: _blank(router_field(row, "eom_prob")),
        }

        def _other_knobs_at_default(row):
            if factor_key is None:
                return True
            for _k, _pred in _knob_at_default.items():
                if _k == factor_key:
                    continue
                if not _pred(row):
                    return False
            return True

        # Gather points keyed by (model_type, factor_value) -> {reps: min_value}.
        series: dict = {}
        factor_values = set()
        extra_buckets: list = []  # non-numeric buckets (e.g. nan_label), in first-seen order
        for _, row in df.iterrows():
            if get_scale(row) != "80M":
                continue
            if not has_tag(row, "olmo_mix"):
                continue
            if not _other_knobs_at_default(row):
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

        # Color encodes the factor value; architecture sets the line style + shade.
        # Buckets are ordered by their numeric value: extra buckets (e.g. max_grad_norm
        # "no clip") sort first (treated as 0), then the baseline and swept values are
        # interleaved by magnitude. This puts e.g. max_grad_norm at "0 (no clip)", 0.2,
        # 1.0 rather than grouping the baseline right after the extra bucket.
        baseline_display = baseline_value if baseline_value is not None else 0.0

        def _bucket_sort_key(b):
            if b == "baseline":
                return baseline_display
            if isinstance(b, str):  # extra bucket (e.g. "no clip") -> first
                return float("-inf")
            return b

        ordered_buckets = sorted(
            extra_buckets + ["baseline"] + list(factor_values), key=_bucket_sort_key
        )
        bucket_color = factor_palette(ordered_buckets)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        any_data = False
        max_reps = 0

        def _bucket_label(fval_key):
            if fval_key == "baseline":
                if baseline_label is not None:
                    return baseline_label
                lbl = f"{baseline_display:.10g}"
                return lbl if "." in lbl else f"{baseline_display:.1f}"
            if isinstance(fval_key, str):
                return fval_key
            return f"{fval_key:g}"

        shown_archs = []  # architectures actually plotted, in canonical order
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
            if model_type not in shown_archs:
                shown_archs.append(model_type)
            if fval_key not in shown_buckets:
                shown_buckets.append(fval_key)
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=3,
                linewidth=1.8,
                linestyle=arch_dash(model_type),
                color=arch_color(bucket_color[fval_key], model_type),
            )
        apply_pow2_xaxis(ax, max_reps, any_data)
        if log_y:
            apply_log_yaxis(ax)
        factor_label = FACTOR_DISPLAY_NAMES.get(raw_factor_label, raw_factor_label)
        ax.set_xlabel("Repetition Rate")
        ax.set_ylabel(axis_label(metric))
        ax.set_title(
            f"{factor_label} — {METRIC_DISPLAY_NAMES.get(metric, metric)} "
            f"(OLMoE Mix, 80M)",
            fontsize=10,
        )
        if any_data:
            bucket_order = [b for b in ordered_buckets if b in shown_buckets]
            arch_factor_legend(
                ax,
                shown_archs,
                factor_label,
                bucket_order,
                bucket_color,
                _bucket_label,
            )
        fig.tight_layout()
        return fig

    return (make_factor_figure,)


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_dropout = metric_multiselect("dropout — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_dropout_arch = arch_multiselect("dropout — architectures", chosen_values=["dense", "moe64"])
    mo.vstack([sel_dropout, sel_dropout_arch])
    return sel_dropout, sel_dropout_arch


@app.cell(hide_code=True)
def _(
    DROPOUT_COL,
    finished_df,
    log_y_g3,
    make_factor_figure,
    render_side_by_side,
    sel_dropout,
    sel_dropout_arch,
):
    # Dropout: baseline = no dropout set (blank).
    _out = render_side_by_side(
        sel_dropout.value,
        lambda m: make_factor_figure(
            finished_df, DROPOUT_COL, "dropout", None, m,
            include_archs=sel_dropout_arch.value, exclude_reps=[128,256,512,1024],
            factor_key="dropout",
            baseline_label="None (disabled)",
         log_y=log_y_g3.value
        ),
        "dropout",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_weight_decay = metric_multiselect("weight decay — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_weight_decay_arch = arch_multiselect("weight decay — architectures", chosen_values=["dense", "moe64"])
    mo.vstack([sel_weight_decay, sel_weight_decay_arch])
    return sel_weight_decay, sel_weight_decay_arch


@app.cell(hide_code=True)
def _(
    WD_COL,
    finished_df,
    log_y_g3,
    make_factor_figure,
    render_side_by_side,
    sel_weight_decay,
    sel_weight_decay_arch,
):
    # Weight decay: baseline = the default 0.1; swept values are 0.2 / 0.4.
    _out = render_side_by_side(
        sel_weight_decay.value,
        lambda m: make_factor_figure(
            finished_df, WD_COL, "weight_decay", 0.1, m,
            include_archs=sel_weight_decay_arch.value, exclude_reps=[128,256,512,1024],
            factor_key="weight_decay",
         log_y=log_y_g3.value
        ),
        "weight_decay",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_max_grad_norm = metric_multiselect("max grad norm — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_max_grad_norm_arch = arch_multiselect("max grad norm — architectures", chosen_values=["dense", "moe64"])
    mo.vstack([sel_max_grad_norm, sel_max_grad_norm_arch])
    return sel_max_grad_norm, sel_max_grad_norm_arch


@app.cell(hide_code=True)
def _(
    MGN_COL,
    finished_df,
    log_y_g3,
    make_factor_figure,
    render_side_by_side,
    sel_max_grad_norm,
    sel_max_grad_norm_arch,
):
    # Max grad norm: baseline = the default 1.0; swept value is 0.2; a `null` (no
    # clipping) setting is bucketed separately via nan_label and sorts first as 0.
    _out = render_side_by_side(
        sel_max_grad_norm.value,
        lambda m: make_factor_figure(
            finished_df, MGN_COL, "max_grad_norm", 1.0, m, nan_label="None (no clip)",
            include_archs=sel_max_grad_norm_arch.value, exclude_reps=[128,256,512,1024],
            factor_key="max_grad_norm",
         log_y=log_y_g3.value
        ),
        "max_grad_norm",
    )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group 3b — MoE-specific regularizer sweeps: jitter, EOM, FOM & expert dropout

    Like Group 3, but for the **MoE-only** regularizers — router jitter
    (`jitter_eps`), Expert Output Masking (`eom_prob`), Final Output Masking
    (`fom_prob`), and expert dropout (`expert_dropout`, dropout on the MoE output
    only). Dense runs are excluded (these knobs only apply to MoE).

    Color encodes model type; line style encodes the regularizer value (baseline =
    no regularizer set). `olmo_mix`, 80M, log X with power-of-2 ticks.
    """)
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_moe_expert_dropout = metric_multiselect("MoE expert_dropout — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_moe_expert_dropout_arch = arch_multiselect(
        "MoE expert_dropout — architectures", options=["dense", "moe32", "moe64"], chosen_values=["dense", "moe64"]
    )
    mo.vstack([sel_moe_expert_dropout, sel_moe_expert_dropout_arch])
    return sel_moe_expert_dropout, sel_moe_expert_dropout_arch


@app.cell(hide_code=True)
def _(
    EXPERT_DROPOUT_COL,
    finished_df,
    log_y_g3,
    make_factor_figure,
    render_side_by_side,
    sel_moe_expert_dropout,
    sel_moe_expert_dropout_arch,
):
    # Expert dropout: baseline = not set (blank). The knob only wires into MoE
    # blocks, so the dense series is a control that should track its baseline.
    _out = render_side_by_side(
        sel_moe_expert_dropout.value,
        lambda m: make_factor_figure(
            finished_df, EXPERT_DROPOUT_COL, "expert_dropout", None, m,
            include_archs=sel_moe_expert_dropout_arch.value, exclude_reps=[128,256,512,1024],
            factor_key="expert_dropout",
         log_y=log_y_g3.value
        ),
        "moe_expert_dropout",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_moe_jitter = metric_multiselect("MoE jitter_eps — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_moe_jitter_arch = arch_multiselect(
        "MoE jitter_eps — architectures", options=["dense", "moe32", "moe64"], chosen_values=["dense", "moe64"]
    )
    mo.vstack([sel_moe_jitter, sel_moe_jitter_arch])
    return sel_moe_jitter, sel_moe_jitter_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_g3,
    make_factor_figure,
    render_side_by_side,
    router_field,
    sel_moe_jitter,
    sel_moe_jitter_arch,
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
            include_archs=sel_moe_jitter_arch.value,
            value_getter=lambda row: router_field(row, "jitter_eps"),
            exclude_reps=[128,256,512,1024],
            factor_key="jitter_eps",
         log_y=log_y_g3.value
        ),
        "moe_jitter_eps",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_moe_eom = metric_multiselect("MoE eom_prob — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_moe_eom_arch = arch_multiselect(
        "MoE eom_prob — architectures", options=["dense","moe32", "moe64"], chosen_values=["dense", "moe64"]
    )
    mo.vstack([sel_moe_eom, sel_moe_eom_arch])
    return sel_moe_eom, sel_moe_eom_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_g3,
    make_factor_figure,
    render_side_by_side,
    router_field,
    sel_moe_eom,
    sel_moe_eom_arch,
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
            include_archs=sel_moe_eom_arch.value,
            value_getter=lambda row: router_field(row, "eom_prob"),
            exclude_reps=[128,256,512,1024],
            factor_key="eom_prob",
         log_y=log_y_g3.value
        ),
        "moe_eom_prob",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_moe_fom = metric_multiselect("MoE fom_prob — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_moe_fom_arch = arch_multiselect(
        "MoE fom_prob — architectures", options=["dense", "moe32", "moe64"], chosen_values=["dense", "moe64"]
    )
    mo.vstack([sel_moe_fom, sel_moe_fom_arch])
    return sel_moe_fom, sel_moe_fom_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    fom_prob_value,
    log_y_g3,
    make_factor_figure,
    render_side_by_side,
    sel_moe_fom,
    sel_moe_fom_arch,
):
    # Final Output Masking: baseline = not set (blank); MoE only.
    _out = render_side_by_side(
        sel_moe_fom.value,
        lambda m: make_factor_figure(
            finished_df, None, "fom_prob", None, m,
            include_archs=sel_moe_fom_arch.value, exclude_reps=[128,256,512,1024],
            factor_key="fom_prob", value_getter=fom_prob_value,
         log_y=log_y_g3.value
        ),
        "moe_fom_prob",
    )
    _out
    return


@app.cell(column=4, hide_code=True)
def _(mo):
    export_btn_g4 = mo.ui.run_button(label="Export images")
    log_y_g4 = mo.ui.checkbox(label="Log y axis")
    mo.vstack([
        mo.md(r"""
    ## Group 4 — `famA` runs

    Every finished run whose name contains `famA`. Color = model type (parsed from
    the run-name suffix, since the newer `famAr` rep-sweep runs are only tagged
    `MoE`). Two plots:

    1. **metric vs. %dclm** (0/25/50/75/100 from `dclm{N}` in the name) — line style
       + shade encode the repetition count.
    2. **metric vs. repetition count** (log X; `rep{N}x` in the name, original famA
       runs are rep 1) — line style + shade encode %dclm.

    **Export images** writes this column to `ml/figures/filtering/`.
    """),
        mo.hstack([export_btn_g4, log_y_g4], justify="start", gap=1),
    ])
    return export_btn_g4, log_y_g4


@app.cell(hide_code=True)
def _(
    arch_tag,
    export_btn_g4,
    export_pdf,
    export_report,
    finished_df,
    log_y_g4,
    make_famA_figure,
    make_famA_rep_figure,
    metric_tag,
    mo,
    sel_famA,
    sel_famA_arch,
    sel_famA_rep,
    sel_famA_rep_arch,
):
    # Export the two Group 4 (famA) plots to filtering/.
    # Name: famA__80M__<architectures>__<x axis>__<metric>.
    if not export_btn_g4.value:
        _out = mo.md("*Click **Export images** to write this column's PDFs.*")
    else:
        _specs = [
            ("pct_dclm", sel_famA, sel_famA_arch, make_famA_figure),
            ("repetition", sel_famA_rep, sel_famA_rep_arch, make_famA_rep_figure),
        ]
        _paths = []
        for _feature, _sel, _arch, _builder in _specs:
            # One separate image per selected metric, always suffixed with the
            # short metric name.
            for _m in _sel.value:
                _fig = _builder(
                    finished_df, _m, include_archs=_arch.value,
                    log_y=log_y_g4.value,
                )
                _paths.append(
                    export_pdf(
                        _fig, "filtering", "famA", "80M",
                        arch_tag(_arch.value), _feature, metric_tag(_m),
                    )
                )
        _out = export_report(_paths)
    _out
    return


@app.cell(hide_code=True)
def _(
    apply_log_yaxis,
    apply_pow2_xaxis,
    arch_color,
    arch_dash,
    arch_factor_legend,
    axis_label,
    factor_palette,
    get_model_type,
    get_reps,
    get_scale,
    has_tag,
    keep_arch,
    metric_value,
    pd,
    plt,
    re,
):
    def parse_dclm_pct(name: str):
        """Extract the DCLM percentage from a 'dclm{N}' token in the run name."""
        m = re.search(r"dclm(\d+)", str(name))
        return int(m.group(1)) if m else None

    def famA_model_type(name: str):
        """Model type from the run-name suffix (famAr runs aren't tagged MoE32/64)."""
        m = re.search(r"_(dense|moe32|moe64)$", str(name))
        return m.group(1) if m else None

    def famA_points(df, metric, include_archs=None, exclude_archs=None):
        """Yield ``(model_type, reps, pct, loss)`` for the famA %dclm plots.

        Includes both the famA runs (``%dclm`` and arch parsed from the name) and the
        single-source **dclm baseline** sweep (tag ``baseline``+``dclm`` at 80M),
        which is folded in as the ``pct = 100`` (100% dclm) series — its arch comes
        from the tags. Runs without a usable metric / pct / arch are skipped.
        """
        out = []
        for _, row in df.iterrows():
            name = str(row.get("Name", ""))
            is_famA = "famA" in name
            is_dclm_base = (
                has_tag(row, "baseline")
                and has_tag(row, "dclm")
                and get_scale(row) == "80M"
            )
            if not (is_famA or is_dclm_base):
                continue
            loss = metric_value(row, metric)
            if pd.isna(loss):
                continue
            if is_famA:
                pct = parse_dclm_pct(name)
                mt = famA_model_type(name)
            else:
                pct = 100  # dclm baseline == 100% dclm
                mt = get_model_type(row)
            if pct is None or mt is None:
                continue
            if not keep_arch(mt, include_archs, exclude_archs):
                continue
            out.append((mt, get_reps(row), pct, float(loss)))
        return out

    def make_famA_figure(
        df, metric, include_archs=None, exclude_archs=None, log_y=False, ax=None
    ):
        """famA: metric vs. %dclm. Color = rep count; line style + shade = architecture.

        The dclm baseline runs are included as the 100% dclm points.
        Set ``log_y=True`` for a log-scaled y axis."""
        # Collect: {(model_type, reps): {pct: min_value}}
        series: dict = {}
        rep_set = set()
        for mt, reps, pct, loss in famA_points(df, metric, include_archs, exclude_archs):
            by_pct = series.setdefault((mt, reps), {})
            by_pct[pct] = min(loss, by_pct.get(pct, loss))
            rep_set.add(reps)

        rep_order = sorted(rep_set)
        rep_color = factor_palette(rep_order)
        # Draw into a caller-supplied axes when given, so the paper figure can put
        # this panel and the rep panel side by side without duplicating the body.
        own_fig = ax is None
        fig = None
        if own_fig:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
        shown_archs = []
        for (mt, reps), by_pct in sorted(series.items()):
            pts = sorted(by_pct.items())
            if mt not in shown_archs:
                shown_archs.append(mt)
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                marker="o",
                markersize=5,
                linewidth=1.5,
                linestyle=arch_dash(mt),
                color=arch_color(rep_color[reps], mt),
            )
        if log_y:
            apply_log_yaxis(ax)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("% dclm")
        ax.set_ylabel(axis_label(metric))
        # ax.set_title(f"famA runs: {metric} vs. %dclm", fontsize=10)
        if series:
            arch_factor_legend(
                ax, shown_archs, "Repetitions", rep_order, rep_color, lambda r: f"{r:g}x"
            )
        if not own_fig:
            return None
        fig.tight_layout()
        return fig

    def make_famA_rep_figure(
        df, metric, include_archs=None, exclude_archs=None, log_y=False, ax=None
    ):
        """famA: metric vs. repetition count (log X). Color = %dclm; line style +
        shade = architecture.

        The dclm baseline runs are included as the 100% dclm series.
        Set ``log_y=True`` for a log-scaled y axis (X is always log/power-of-2)."""
        # Collect: {(model_type, pct): {reps: min_value}}
        series: dict = {}
        pct_set = set()
        for mt, reps, pct, loss in famA_points(df, metric, include_archs, exclude_archs):
            by_reps = series.setdefault((mt, pct), {})
            by_reps[reps] = min(loss, by_reps.get(reps, loss))
            pct_set.add(pct)

        pct_order = sorted(pct_set)
        pct_color = factor_palette(pct_order)
        own_fig = ax is None
        fig = None
        if own_fig:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
        shown_archs = []
        max_reps = 0
        any_data = False
        for (mt, pct), by_reps in sorted(series.items()):
            xy = sorted(by_reps.items())
            if not xy:
                continue
            any_data = True
            if mt not in shown_archs:
                shown_archs.append(mt)
            max_reps = max(max_reps, max(r for r, _ in xy))
            ax.plot(
                [r for r, _ in xy],
                [v for _, v in xy],
                marker="o",
                markersize=5,
                linewidth=1.5,
                linestyle=arch_dash(mt),
                color=arch_color(pct_color[pct], mt),
            )
        apply_pow2_xaxis(ax, max_reps, any_data)
        if log_y:
            apply_log_yaxis(ax)
        ax.set_xlabel("Repetition Rate")
        ax.set_ylabel(axis_label(metric))
        # ax.set_title(f"famA runs: {metric} vs. repetition", fontsize=10)
        if any_data:
            arch_factor_legend(
                ax, shown_archs, "% dclm", pct_order, pct_color, lambda p: f"{p:g}%"
            )
        if not own_fig:
            return None
        fig.tight_layout()
        return fig

    return make_famA_figure, make_famA_rep_figure


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_famA = metric_multiselect("famA — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/downstream/hellaswag (CE loss)"])
    sel_famA_arch = arch_multiselect("famA — architectures", chosen_values=["dense", "moe64"])
    mo.vstack([sel_famA, sel_famA_arch])
    return sel_famA, sel_famA_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_g4,
    make_famA_figure,
    render_side_by_side,
    sel_famA,
    sel_famA_arch,
):
    _out = render_side_by_side(
        sel_famA.value,
        lambda m: make_famA_figure(finished_df, m, include_archs=sel_famA_arch.value, log_y=log_y_g4.value),
        "famA",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_famA_rep = metric_multiselect(
        "famA (vs reps) — metrics",
        chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/downstream/hellaswag (CE loss)"],
    )
    sel_famA_rep_arch = arch_multiselect("famA (vs reps) — architectures", chosen_values=["dense", "moe64"])
    mo.vstack([sel_famA_rep, sel_famA_rep_arch])
    return sel_famA_rep, sel_famA_rep_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_g4,
    make_famA_rep_figure,
    render_side_by_side,
    sel_famA_rep,
    sel_famA_rep_arch,
):
    _out = render_side_by_side(
        sel_famA_rep.value,
        lambda m: make_famA_rep_figure(
            finished_df, m, include_archs=sel_famA_rep_arch.value
        , log_y=log_y_g4.value
        ),
        "famA_rep",
    )
    _out
    return


@app.cell(column=5, hide_code=True)
def _(mo):
    export_btn_g5 = mo.ui.run_button(label="Export images")
    log_y_g5 = mo.ui.checkbox(label="Log y axis")
    mo.vstack([
        mo.md(r"""
    ## Group 5 — `famB` runs: metric vs. repetition

    `famB` runs mix a single source into `dclm` at a fixed fraction. **The two
    families name that fraction differently**, so the mixture configs rather than
    the run names are the source of truth here:

    - `sc90` / `sc50` are named for the **dclm** share — 90%/50% dclm, i.e.
      **10%/50% starcoder**.
    - `p2o10` / `p2o50` are named for the **pes2o** share — **10%/50% pes2o**.

    Both are therefore labelled by minority share in the legend (10% / 50% / 100%),
    with 100% being the pure single-source baseline.

    Each figure plots the chosen metric vs. repetition count (log X, power-of-2 ticks)
    with:

    - **line color** = data source, and
    - **line style + shade** = architecture (dense solid/darkest, moe32 dashed,
      moe64 dotted/lightest).

    The reference lines are the `dclm` and single-source **baseline** runs (tag
    `baseline`); duplicate baseline runs at a rep count keep the lowest value (see
    Group 1 & 2).

    **Export images** writes this column to `ml/figures/mix_varied_rep/`.
    """),
        mo.hstack([export_btn_g5, log_y_g5], justify="start", gap=1),
    ])
    return export_btn_g5, log_y_g5


@app.cell(hide_code=True)
def _(
    FAMB_P2O_SOURCES,
    FAMB_SC_SOURCES,
    arch_tag,
    export_btn_g5,
    export_pdf,
    export_report,
    finished_df,
    log_y_g5,
    make_famB_figure,
    metric_tag,
    mo,
    sel_famB_p2o,
    sel_famB_p2o_arch,
    sel_famB_sc,
    sel_famB_sc_arch,
):
    # Export the two Group 5 (famB) family plots to mix_varied_rep/.
    # Name: <family data source>__80M__<architectures>__<metric>.
    if not export_btn_g5.value:
        _out = mo.md("*Click **Export images** to write this column's PDFs.*")
    else:
        _sc_sources = FAMB_SC_SOURCES
        _p2o_sources = FAMB_P2O_SOURCES
        _specs = [
            ("starcoder", sel_famB_sc, sel_famB_sc_arch, _sc_sources, "% StarCoder"),
            ("pes2o", sel_famB_p2o, sel_famB_p2o_arch, _p2o_sources, "% peS2o"),
        ]
        _paths = []
        for _family, _sel, _arch, _sources, _src_title in _specs:
            # One separate image per selected metric, always suffixed with the
            # short metric name.
            for _m in _sel.value:
                _fig = make_famB_figure(
                    finished_df, _m, f"famB ({_family}): {_m} vs. repetition",
                    _sources, include_archs=_arch.value, exclude_reps=[128, 256],
                 log_y=log_y_g5.value
                    , source_title=_src_title
                )
                _paths.append(
                    export_pdf(
                        _fig, "mix_varied_rep", _family, "80M",
                        arch_tag(_arch.value), metric_tag(_m),
                    )
                )
        _out = export_report(_paths)
    _out
    return


@app.cell(hide_code=True)
def _(
    DATA_DISPLAY_NAMES,
    apply_log_yaxis,
    apply_pow2_xaxis,
    arch_color,
    arch_dash,
    arch_factor_legend,
    arch_of_run,
    axis_label,
    collect_baseline_points,
    factor_palette,
    filter_rep_points,
    get_reps,
    keep_arch,
    metric_value,
    pd,
    plt,
):
    def collect_famB_points(df, name_substr, model_type, metric):
        """Sorted (reps, value) points for famB runs matching a name substring.

        Architecture comes from ``arch_of_run`` (run-name suffix, tags as fallback)
        because some famB sweeps — e.g. ``p2o10`` — are only tagged ``MoE`` and would
        otherwise be dropped as "other".
        """
        by_reps: dict = {}
        for _, row in df.iterrows():
            name = str(row.get("Name", ""))
            if "famB" not in name or name_substr not in name:
                continue
            if arch_of_run(row) != model_type:
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
        log_y=False,
        source_title="Data source",
    ):
        """One famB figure.

        Set ``log_y=True`` for a log-scaled y axis (X is always log/power-of-2).

        `sources` is an ordered list of (label, color, kind, key) tuples. `kind` is
        one of:

        - ``"baseline"`` — key is a data tag; plots the 80M baseline runs.
        - ``"famB"`` — key is a run-name substring; plots those runs.
        - ``"hline"`` — draws a dotted horizontal reference line at the value of
          that source's **dense, rep-1** run instead of a per-architecture series,
          annotated inline (so it needs no legend entry). Used for the dclm
          reference, which is one comparison point rather than a sweep.

        The per-source color is ignored: color now encodes the data source and line
        style + shade encode the architecture. An ``"hline"`` source still occupies
        its palette slot, so the other series keep their colors.

        Filters (all default to no-op):

        - ``include_archs`` / ``exclude_archs``: keep/drop architectures.
        - ``include_reps`` / ``exclude_reps``: keep/drop repetition counts.
        """
        source_labels = [src[0] for src in sources]
        source_color = factor_palette(source_labels)
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        max_reps = 0
        any_data = False
        shown_archs = []
        shown_sources = []
        hlines = []  # (label, key, y) reference lines, annotated after scaling
        for label, color, kind, key in sources:
            if kind == "hline":
                # Single reference value: this source's dense, rep-1 run.
                _ref_pts = collect_baseline_points(df, key, "80M", "dense", metric)
                _ref = next((v for r, v in _ref_pts if float(r) == 1.0), None)
                if _ref is not None:
                    ax.axhline(
                        _ref,
                        linestyle=":",
                        linewidth=1.2,
                        color=source_color[label],
                        zorder=1,
                    )
                    hlines.append((label, key, _ref))
                continue
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
                if model_type not in shown_archs:
                    shown_archs.append(model_type)
                if label not in shown_sources:
                    shown_sources.append(label)
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    linestyle=arch_dash(model_type),
                    color=arch_color(source_color[label], model_type),
                )
        apply_pow2_xaxis(ax, max_reps, any_data)
        if log_y:
            apply_log_yaxis(ax)
        # Annotate the reference lines once the scale is settled: x in axes
        # fraction, y in data coords, so each label rides just above its line.
        for _hl_label, _hl_key, _hl_y in hlines:
            ax.text(
                0.012,
                _hl_y,
                f"{DATA_DISPLAY_NAMES.get(_hl_key, _hl_key)} (R=1)",
                transform=ax.get_yaxis_transform(),
                color=source_color[_hl_label],
                fontsize=7,
                ha="left",
                va="bottom",
            )
        ax.set_xlabel("Repetition Rate")
        ax.set_ylabel(axis_label(metric))
        ax.set_title(title, fontsize=10)
        if any_data:
            source_order = [s for s in source_labels if s in shown_sources]
            arch_factor_legend(
                ax, shown_archs, source_title, source_order, source_color, str
            )
        fig.tight_layout()
        return fig

    return collect_famB_points, make_famB_figure


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_famB_sc = metric_multiselect("famB starcoder — metrics", chosen_values=["eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_stack-validation/CE loss"])
    sel_famB_sc_arch = arch_multiselect("famB starcoder — architectures", chosen_values=["dense", "moe64"])
    mo.vstack([sel_famB_sc, sel_famB_sc_arch])
    return sel_famB_sc, sel_famB_sc_arch


@app.cell(hide_code=True)
def _(
    FAMB_SC_SOURCES,
    finished_df,
    log_y_g5,
    make_famB_figure,
    render_side_by_side,
    sel_famB_sc,
    sel_famB_sc_arch,
):
    # Plot 1: starcoder family. Color = source; line style + shade = architecture.
    # Order (and thus palette assignment) is dclm, sc90, sc50, starcoder.
    _sources = FAMB_SC_SOURCES
    _out = render_side_by_side(
        sel_famB_sc.value,
        lambda m: make_famB_figure(
            finished_df, m, f"famB (starcoder): {m} vs. repetition", _sources,
            include_archs=sel_famB_sc_arch.value, exclude_reps=[128,256],
         log_y=log_y_g5.value
            , source_title="% StarCoder"
        ),
        "famB_starcoder",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_famB_p2o = metric_multiselect("famB pes2o — metrics", chosen_values=["eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_pes2o-validation/CE loss"])
    sel_famB_p2o_arch = arch_multiselect("famB pes2o — architectures", chosen_values=["dense", "moe64"])
    mo.vstack([sel_famB_p2o, sel_famB_p2o_arch])
    return sel_famB_p2o, sel_famB_p2o_arch


@app.cell(hide_code=True)
def _(
    FAMB_P2O_SOURCES,
    finished_df,
    log_y_g5,
    make_famB_figure,
    render_side_by_side,
    sel_famB_p2o,
    sel_famB_p2o_arch,
):
    # Plot 2: pes2o family. Color = source; line style + shade = architecture.
    # Order (and thus palette assignment) is dclm, p2o10, p2o50, pes2o.
    _sources = FAMB_P2O_SOURCES
    _out = render_side_by_side(
        sel_famB_p2o.value,
        lambda m: make_famB_figure(
            finished_df, m, f"famB (pes2o): {m} vs. repetition", _sources,
            include_archs=sel_famB_p2o_arch.value, exclude_reps=[128,256],
         log_y=log_y_g5.value
            , source_title="% peS2o"
        ),
        "famB_pes2o",
    )
    _out
    return


@app.cell(column=6, hide_code=True)
def _(mo):
    export_btn_curves = mo.ui.run_button(label="Export images")
    log_y_curves = mo.ui.checkbox(label="Log y axis")
    mo.vstack([
        mo.md(r"""
    ## Training curves — metric vs. train step (baseline runs)

    Per-step training curves from the downloaded W&B history
    (`ml/runs_data/history/`). Baseline `olmo_mix` runs (and the single-source
    mixtures) at 80M / 200M. Each plot is **faceted**: one column per architecture
    (dense / moe32 / moe64) and one row per selected metric, with one line per
    repetition count (color = rep count).

    **Export images** writes this column to `ml/figures/history/`, one file per
    metric (each a single-metric facet row).
    """),
        mo.hstack([export_btn_curves, log_y_curves], justify="start", gap=1),
    ])
    return export_btn_curves, log_y_curves


@app.cell(hide_code=True)
def _(
    arch_tag,
    export_btn_curves,
    export_pdf,
    export_report,
    finished_df,
    log_y_curves,
    make_training_curve_figure,
    metric_tag,
    mo,
    sel_curve_dclm_80m,
    sel_curve_dclm_80m_arch,
    sel_curve_olmo_200m,
    sel_curve_olmo_200m_arch,
    sel_curve_olmo_80m,
    sel_curve_olmo_80m_arch,
    sel_curve_pes2o_80m,
    sel_curve_pes2o_80m_arch,
    sel_curve_starcoder_80m,
    sel_curve_starcoder_80m_arch,
    sel_curve_wiki_80m,
    sel_curve_wiki_80m_arch,
):
    # Export the training-curve facets to history/, one PDF per metric so the
    # metric name is meaningful. Name:
    #   <data source>__<scale>__<architectures>__<metric>.
    if not export_btn_curves.value:
        _out = mo.md("*Click **Export images** to write this column's PDFs.*")
    else:
        _specs = [
            ("olmo_mix", "80M", sel_curve_olmo_80m, sel_curve_olmo_80m_arch),
            ("olmo_mix", "200M", sel_curve_olmo_200m, sel_curve_olmo_200m_arch),
            ("dclm", "80M", sel_curve_dclm_80m, sel_curve_dclm_80m_arch),
            ("starcoder", "80M", sel_curve_starcoder_80m, sel_curve_starcoder_80m_arch),
            ("pes2o", "80M", sel_curve_pes2o_80m, sel_curve_pes2o_80m_arch),
            ("wiki", "80M", sel_curve_wiki_80m, sel_curve_wiki_80m_arch),
        ]
        _paths = []
        for _tag, _scale, _sel, _arch in _specs:
            for _m in _sel.value:
                _fig = make_training_curve_figure(
                    finished_df, _tag, _scale, [_m], include_archs=_arch.value,
                    log_y=True if log_y_curves.value else None,
                )
                _paths.append(
                    export_pdf(
                        _fig, "history", _tag, _scale, arch_tag(_arch.value),
                        metric_tag(_m),
                    )
                )
        _out = export_report(_paths)
    _out
    return


@app.cell(hide_code=True)
def _(
    METRIC_DISPLAY_NAMES,
    MODEL_DISPLAY_NAME,
    apply_log_yaxis,
    factor_palette,
    get_model_type,
    get_reps,
    get_scale,
    has_tag,
    keep_arch,
    load_history_series,
    pd,
    plt,
):
    def make_training_curve_figure(
        df,
        data_tag,
        scale,
        metrics,
        include_archs=None,
        exclude_archs=None,
        include_reps=None,
        exclude_reps=None,
        annotate_reps=True,
        annotate_min_gap=0.01,
        log_y=None,
        smooth_window=100,
    ):
        """Faceted training curves (metric vs. train step) for baseline runs of one
        data mixture + model scale.

        Facets are laid out as a grid: one **column per architecture**
        (dense/moe32/moe64) and one **row per metric** in ``metrics``. Within each
        facet, one line per repetition count (color encodes the rep count). Reads
        per-step values from the downloaded W&B history.

        Smoothing
        ---------
        ``train/*`` metrics are logged every step and are noisy, so their per-step
        curve is replaced by a trailing moving average over ``smooth_window`` steps
        (default 100), matching the last-window averaging used for the final-value
        plots. Non-``train/*`` metrics are drawn raw. Set ``smooth_window`` to 1 (or
        0) to disable.

        Log y-axis
        ----------
        ``log_y`` controls the y scale: ``True``/``False`` forces log/linear for every
        facet, ``None`` (the default) auto-detects per metric — the MoE router load
        metrics (names ending in ``load imbalance`` or ``load balancing loss
        unscaled``) get a log y-axis, everything else stays linear.

        Inline rep labels
        -----------------
        When ``annotate_reps`` is True (the default), each line's rep count (e.g.
        ``"16x"``) is written just above its right endpoint, in the line's color.
        To avoid unreadable pile-ups where curves converge, a label is drawn only if
        its endpoint is vertically far enough from every other endpoint in that facet:
        endpoints are compared on the y axis in facet-relative units (fraction of the
        facet's height, honoring the y scale), and a label is skipped if its nearest
        neighbor is within ``annotate_min_gap`` (so a tightly-converged cluster gets no
        inline labels — the shared legend still covers them). Set
        ``annotate_reps=False`` to disable the inline labels entirely.
        """
        def _is_log_metric(metric):
            m = metric.lower()
            return m.endswith("load imbalance") or m.endswith(
                "load balancing loss unscaled"
            )

        def _trailing_avg(series, window):
            """Trailing moving average of a sorted [(step, value), ...] series.

            Point i becomes the mean of the values in the window ending at i (up to
            `window` points), so the curve is smoothed while keeping every step's x.
            Only applied to train/* metrics, which are logged every step.
            """
            if window is None or window <= 1:
                return series
            out = []
            acc = 0.0
            from collections import deque

            buf = deque()
            for x, v in series:
                buf.append(v)
                acc += v
                if len(buf) > window:
                    acc -= buf.popleft()
                out.append((x, acc / len(buf)))
            return out

        if isinstance(metrics, str):
            metrics = [metrics]

        # Architectures to show, in canonical order, filtered by include/exclude.
        archs = [
            a
            for a in ["dense", "moe32", "moe64"]
            if keep_arch(a, include_archs, exclude_archs)
        ]

        # Collect baseline runs: {(model_type, reps): run_name} (keep last seen), and
        # the global rep set so colors are consistent across facets.
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
            if model_type not in archs:
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

        rep_order = sorted(rep_set)
        rep_color = factor_palette(rep_order)  # color per repetition count

        n_rows = max(1, len(metrics))
        n_cols = max(1, len(archs))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.2 * n_cols, 3.4 * n_rows),
            squeeze=False,
        )

        any_data = False
        shown_reps = []
        for r, metric in enumerate(metrics):
            use_log = _is_log_metric(metric) if log_y is None else log_y
            # train/* metrics are logged every step and noisy -> trailing average.
            smooth = smooth_window if metric.startswith("train/") else None
            for c, model_type in enumerate(archs):
                ax = axes[r][c]
                endpoints = []  # (reps, x_last, y_last) for inline labels
                for reps in rep_order:
                    run_name = runs.get((model_type, reps))
                    if run_name is None:
                        continue
                    series = load_history_series(run_name, metric)
                    if not series:
                        continue
                    series = _trailing_avg(series, smooth)
                    any_data = True
                    if reps not in shown_reps:
                        shown_reps.append(reps)
                    ax.plot(
                        [p[0] for p in series],
                        [p[1] for p in series],
                        linewidth=1.0,
                        color=rep_color[reps],
                    )
                    endpoints.append((reps, series[-1][0], series[-1][1]))

                if use_log:
                    apply_log_yaxis(ax)

                # Inline rep labels: only for endpoints that are vertically well
                # separated from every other endpoint in this facet. Distances are
                # measured in axis-fraction units so the rule works under log scale.
                if annotate_reps and endpoints:
                    import math

                    ymin, ymax = ax.get_ylim()

                    def _yfrac(y):
                        if use_log:
                            if y <= 0 or ymin <= 0:
                                return 0.0
                            lo, hi = math.log10(ymin), math.log10(ymax)
                            return (math.log10(y) - lo) / ((hi - lo) or 1.0)
                        return (y - ymin) / ((ymax - ymin) or 1.0)

                    for reps, x_last, y_last in endpoints:
                        nearest = min(
                            (
                                abs(_yfrac(y_last) - _yfrac(other_y))
                                for o_reps, _, other_y in endpoints
                                if o_reps != reps
                            ),
                            default=float("inf"),
                        )
                        if nearest < annotate_min_gap:
                            continue
                        ax.annotate(
                            f"{reps:g}x",
                            xy=(x_last, y_last),
                            xytext=(3, -3),
                            textcoords="offset points",
                            fontsize=6,
                            color=rep_color[reps],
                            ha="left",
                            va="bottom",
                            clip_on=False,
                        )

                # Column title (architecture) only on the top row.
                if r == 0:
                    ax.set_title(
                        MODEL_DISPLAY_NAME.get(model_type, model_type),
                        fontsize=11,
                        fontweight="bold",
                    )
                # Y label (metric) only on the leftmost column.
                if c == 0:
                    ax.set_ylabel(METRIC_DISPLAY_NAMES.get(metric, metric), fontsize=8)
                # X label only on the bottom row.
                if r == n_rows - 1:
                    ax.set_xlabel("Train Step")

        # fig.suptitle(f"{DATA_DISPLAY_NAMES.get(data_tag, data_tag)} — {scale}: Training Curves ", fontsize=11)

        if any_data:
            from matplotlib.lines import Line2D

            rep_shown = [r for r in rep_order if r in shown_reps]
            handles = [
                Line2D([0], [0], color=rep_color[r], linewidth=2) for r in rep_shown
            ]
            labels = [f"{r:g}x" for r in rep_shown]
            fig.legend(
                handles,
                labels,
                title="Repetitions",
                fontsize=7,
                title_fontsize=8,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                borderaxespad=0.0,
            )
        else:
            axes[0][0].set_title(
                axes[0][0].get_title() + "  [no history found]", fontsize=9
            )

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig

    return (make_training_curve_figure,)


@app.cell(hide_code=True)
def _(arch_multiselect, history_metric_multiselect, mo):
    sel_curve_olmo_80m = history_metric_multiselect(
        "curves: olmo_mix 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_curve_olmo_80m_arch = arch_multiselect("curves: olmo_mix 80M — architectures", chosen_values=["dense", "moe64"])
    mo.vstack([sel_curve_olmo_80m, sel_curve_olmo_80m_arch])
    return sel_curve_olmo_80m, sel_curve_olmo_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_curves,
    make_training_curve_figure,
    render_faceted,
    sel_curve_olmo_80m,
    sel_curve_olmo_80m_arch,
):
    _out = render_faceted(
        sel_curve_olmo_80m.value,
        lambda ms: make_training_curve_figure(
            finished_df, "olmo_mix", "80M", ms, include_archs=sel_curve_olmo_80m_arch.value
        , log_y=True if log_y_curves.value else None
        ),
        "curve_olmo_mix_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, history_metric_multiselect, mo):
    sel_curve_olmo_200m = history_metric_multiselect(
        "curves: olmo_mix 200M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_curve_olmo_200m_arch = arch_multiselect("curves: olmo_mix 200M — architectures", chosen_values=["dense", "moe64"])
    mo.vstack([sel_curve_olmo_200m, sel_curve_olmo_200m_arch])
    return sel_curve_olmo_200m, sel_curve_olmo_200m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_curves,
    make_training_curve_figure,
    render_faceted,
    sel_curve_olmo_200m,
    sel_curve_olmo_200m_arch,
):
    _out = render_faceted(
        sel_curve_olmo_200m.value,
        lambda ms: make_training_curve_figure(
            finished_df, "olmo_mix", "200M", ms, include_archs=sel_curve_olmo_200m_arch.value
        , log_y=True if log_y_curves.value else None
        ),
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
def _(arch_multiselect, history_metric_multiselect, mo):
    sel_curve_dclm_80m = history_metric_multiselect(
        "curves: dclm 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss"])
    sel_curve_dclm_80m_arch = arch_multiselect("curves: dclm 80M — architectures")
    mo.vstack([sel_curve_dclm_80m, sel_curve_dclm_80m_arch])
    return sel_curve_dclm_80m, sel_curve_dclm_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_curves,
    make_training_curve_figure,
    render_faceted,
    sel_curve_dclm_80m,
    sel_curve_dclm_80m_arch,
):
    _out = render_faceted(
        sel_curve_dclm_80m.value,
        lambda ms: make_training_curve_figure(
            finished_df, "dclm", "80M", ms, include_archs=sel_curve_dclm_80m_arch.value
        , log_y=True if log_y_curves.value else None
        ),
        "curve_dclm_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, history_metric_multiselect, mo):
    sel_curve_starcoder_80m = history_metric_multiselect(
        "curves: starcoder 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_stack-validation/CE loss"])
    sel_curve_starcoder_80m_arch = arch_multiselect("curves: starcoder 80M — architectures")
    mo.vstack([sel_curve_starcoder_80m, sel_curve_starcoder_80m_arch])
    return sel_curve_starcoder_80m, sel_curve_starcoder_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_curves,
    make_training_curve_figure,
    render_faceted,
    sel_curve_starcoder_80m,
    sel_curve_starcoder_80m_arch,
):
    _out = render_faceted(
        sel_curve_starcoder_80m.value,
        lambda ms: make_training_curve_figure(
            finished_df, "starcoder", "80M", ms, include_archs=sel_curve_starcoder_80m_arch.value
        , log_y=True if log_y_curves.value else None
        ),
        "curve_starcoder_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, history_metric_multiselect, mo):
    sel_curve_pes2o_80m = history_metric_multiselect(
        "curves: pes2o 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_pes2o-validation/CE loss"])
    sel_curve_pes2o_80m_arch = arch_multiselect("curves: pes2o 80M — architectures")
    mo.vstack([sel_curve_pes2o_80m, sel_curve_pes2o_80m_arch])
    return sel_curve_pes2o_80m, sel_curve_pes2o_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_curves,
    make_training_curve_figure,
    render_faceted,
    sel_curve_pes2o_80m,
    sel_curve_pes2o_80m_arch,
):
    _out = render_faceted(
        sel_curve_pes2o_80m.value,
        lambda ms: make_training_curve_figure(
            finished_df, "pes2o", "80M", ms, include_archs=sel_curve_pes2o_80m_arch.value
        , log_y=True if log_y_curves.value else None
        ),
        "curve_pes2o_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, history_metric_multiselect, mo):
    sel_curve_wiki_80m = history_metric_multiselect(
        "curves: wiki 80M — metrics", chosen_values=["train/CE loss", "eval/lm/dolma_common-crawl-validation/CE loss", "eval/lm/dolma_wiki-validation/CE loss"]
    )
    sel_curve_wiki_80m_arch = arch_multiselect("curves: wiki 80M — architectures")
    mo.vstack([sel_curve_wiki_80m, sel_curve_wiki_80m_arch])
    return sel_curve_wiki_80m, sel_curve_wiki_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_curves,
    make_training_curve_figure,
    render_faceted,
    sel_curve_wiki_80m,
    sel_curve_wiki_80m_arch,
):
    _out = render_faceted(
        sel_curve_wiki_80m.value,
        lambda ms: make_training_curve_figure(
            finished_df, "wiki", "80M", ms, include_archs=sel_curve_wiki_80m_arch.value
        , log_y=True if log_y_curves.value else None
        ),
        "curve_wiki_80M",
    )
    _out
    return


@app.cell(column=7, hide_code=True)
def _(mo):
    export_btn_load = mo.ui.run_button(label="Export images")
    log_y_load = mo.ui.checkbox(label="Log y axis")
    mo.vstack([
        mo.md(r"""
    ## Router load metrics — metric vs. train step (baseline runs)

    Identical faceting to the training-curve plots above (one column per
    architecture, one row per selected metric, one line per repetition count), but
    the metric dropdown lists the **MoE router load metrics** — every series ending
    in `load imbalance` or `load balancing loss unscaled` (the aggregate
    `train/router 0 ...` plus the per-block variants). Dense is excluded since these
    are MoE-only.

    **Export images** writes this column to `ml/figures/history/` (alongside the
    training curves), one file per metric.
    """),
        mo.hstack([export_btn_load, log_y_load], justify="start", gap=1),
    ])
    return export_btn_load, log_y_load


@app.cell(hide_code=True)
def _(
    arch_tag,
    export_btn_load,
    export_pdf,
    export_report,
    finished_df,
    log_y_load,
    make_training_curve_figure,
    metric_tag,
    mo,
    sel_load_dclm_80m,
    sel_load_dclm_80m_arch,
    sel_load_olmo_200m,
    sel_load_olmo_200m_arch,
    sel_load_olmo_80m,
    sel_load_olmo_80m_arch,
    sel_load_pes2o_80m,
    sel_load_pes2o_80m_arch,
    sel_load_starcoder_80m,
    sel_load_starcoder_80m_arch,
    sel_load_wiki_80m,
    sel_load_wiki_80m_arch,
):
    # Export the router-load facets to history/ (same folder as the training
    # curves; the metric name keeps them distinct). Name:
    #   <data source>__<scale>__<architectures>__<metric>.
    if not export_btn_load.value:
        _out = mo.md("*Click **Export images** to write this column's PDFs.*")
    else:
        _specs = [
            ("olmo_mix", "80M", sel_load_olmo_80m, sel_load_olmo_80m_arch),
            ("olmo_mix", "200M", sel_load_olmo_200m, sel_load_olmo_200m_arch),
            ("dclm", "80M", sel_load_dclm_80m, sel_load_dclm_80m_arch),
            ("starcoder", "80M", sel_load_starcoder_80m, sel_load_starcoder_80m_arch),
            ("pes2o", "80M", sel_load_pes2o_80m, sel_load_pes2o_80m_arch),
            ("wiki", "80M", sel_load_wiki_80m, sel_load_wiki_80m_arch),
        ]
        _paths = []
        for _tag, _scale, _sel, _arch in _specs:
            for _m in _sel.value:
                _fig = make_training_curve_figure(
                    finished_df, _tag, _scale, [_m], include_archs=_arch.value,
                    log_y=True if log_y_load.value else None,
                )
                _paths.append(
                    export_pdf(
                        _fig, "history", _tag, _scale, arch_tag(_arch.value),
                        metric_tag(_m),
                    )
                )
        _out = export_report(_paths)
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, load_metric_multiselect, mo):
    sel_load_olmo_80m = load_metric_multiselect("load: olmo_mix 80M — metrics")
    sel_load_olmo_80m_arch = arch_multiselect(
        "load: olmo_mix 80M — architectures", options=["moe32", "moe64"]
    )
    mo.vstack([sel_load_olmo_80m, sel_load_olmo_80m_arch])
    return sel_load_olmo_80m, sel_load_olmo_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_load,
    make_training_curve_figure,
    render_faceted,
    sel_load_olmo_80m,
    sel_load_olmo_80m_arch,
):
    _out = render_faceted(
        sel_load_olmo_80m.value,
        lambda ms: make_training_curve_figure(
            finished_df, "olmo_mix", "80M", ms, include_archs=sel_load_olmo_80m_arch.value
        , log_y=True if log_y_load.value else None
        ),
        "load_olmo_mix_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, load_metric_multiselect, mo):
    sel_load_olmo_200m = load_metric_multiselect("load: olmo_mix 200M — metrics")
    sel_load_olmo_200m_arch = arch_multiselect(
        "load: olmo_mix 200M — architectures", options=["moe32", "moe64"]
    )
    mo.vstack([sel_load_olmo_200m, sel_load_olmo_200m_arch])
    return sel_load_olmo_200m, sel_load_olmo_200m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_load,
    make_training_curve_figure,
    render_faceted,
    sel_load_olmo_200m,
    sel_load_olmo_200m_arch,
):
    _out = render_faceted(
        sel_load_olmo_200m.value,
        lambda ms: make_training_curve_figure(
            finished_df, "olmo_mix", "200M", ms, include_archs=sel_load_olmo_200m_arch.value
        , log_y=True if log_y_load.value else None
        ),
        "load_olmo_mix_200M",
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
def _(arch_multiselect, load_metric_multiselect, mo):
    sel_load_dclm_80m = load_metric_multiselect("load: dclm 80M — metrics")
    sel_load_dclm_80m_arch = arch_multiselect(
        "load: dclm 80M — architectures", options=["moe32", "moe64"]
    )
    mo.vstack([sel_load_dclm_80m, sel_load_dclm_80m_arch])
    return sel_load_dclm_80m, sel_load_dclm_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_load,
    make_training_curve_figure,
    render_faceted,
    sel_load_dclm_80m,
    sel_load_dclm_80m_arch,
):
    _out = render_faceted(
        sel_load_dclm_80m.value,
        lambda ms: make_training_curve_figure(
            finished_df, "dclm", "80M", ms, include_archs=sel_load_dclm_80m_arch.value
        , log_y=True if log_y_load.value else None
        ),
        "load_dclm_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, load_metric_multiselect, mo):
    sel_load_starcoder_80m = load_metric_multiselect("load: starcoder 80M — metrics")
    sel_load_starcoder_80m_arch = arch_multiselect(
        "load: starcoder 80M — architectures", options=["moe32", "moe64"]
    )
    mo.vstack([sel_load_starcoder_80m, sel_load_starcoder_80m_arch])
    return sel_load_starcoder_80m, sel_load_starcoder_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_load,
    make_training_curve_figure,
    render_faceted,
    sel_load_starcoder_80m,
    sel_load_starcoder_80m_arch,
):
    _out = render_faceted(
        sel_load_starcoder_80m.value,
        lambda ms: make_training_curve_figure(
            finished_df, "starcoder", "80M", ms, include_archs=sel_load_starcoder_80m_arch.value
        , log_y=True if log_y_load.value else None
        ),
        "load_starcoder_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, load_metric_multiselect, mo):
    sel_load_pes2o_80m = load_metric_multiselect("load: pes2o 80M — metrics")
    sel_load_pes2o_80m_arch = arch_multiselect(
        "load: pes2o 80M — architectures", options=["moe32", "moe64"]
    )
    mo.vstack([sel_load_pes2o_80m, sel_load_pes2o_80m_arch])
    return sel_load_pes2o_80m, sel_load_pes2o_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_load,
    make_training_curve_figure,
    render_faceted,
    sel_load_pes2o_80m,
    sel_load_pes2o_80m_arch,
):
    _out = render_faceted(
        sel_load_pes2o_80m.value,
        lambda ms: make_training_curve_figure(
            finished_df, "pes2o", "80M", ms, include_archs=sel_load_pes2o_80m_arch.value
        , log_y=True if log_y_load.value else None
        ),
        "load_pes2o_80M",
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, load_metric_multiselect, mo):
    sel_load_wiki_80m = load_metric_multiselect("load: wiki 80M — metrics")
    sel_load_wiki_80m_arch = arch_multiselect(
        "load: wiki 80M — architectures", options=["moe32", "moe64"]
    )
    mo.vstack([sel_load_wiki_80m, sel_load_wiki_80m_arch])
    return sel_load_wiki_80m, sel_load_wiki_80m_arch


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_load,
    make_training_curve_figure,
    render_faceted,
    sel_load_wiki_80m,
    sel_load_wiki_80m_arch,
):
    _out = render_faceted(
        sel_load_wiki_80m.value,
        lambda ms: make_training_curve_figure(
            finished_df, "wiki", "80M", ms, include_archs=sel_load_wiki_80m_arch.value
        , log_y=True if log_y_load.value else None
        ),
        "load_wiki_80M",
    )
    _out
    return


@app.cell(column=8, hide_code=True)
def _(mo):
    export_btn_paper = mo.ui.run_button(label="Export images")
    log_y_paper = mo.ui.checkbox(label="Log y axis")
    mo.vstack([
        mo.md(r"""
    # Paper Figures

    Publication-ready versions of the figures, kept separate from the exploratory
    columns so their styling can be tuned without disturbing anything else.

    **Loss vs. repetition, across model scale.** One figure per selected metric, with
    each model scale as its own subfigure. The subfigures deliberately do **not**
    share a y axis — absolute loss differs enough between 10M and 1B that a shared
    scale flattens the within-scale trend that the figure is about. A single legend
    covers the whole figure.

    **Export images** writes to `ml/figures/paper/`.
    """),
        mo.hstack([export_btn_paper, log_y_paper], justify="start", gap=1),
    ])
    return export_btn_paper, log_y_paper


@app.cell(hide_code=True)
def _(
    arch_tag,
    export_btn_paper,
    export_pdf,
    export_report,
    finished_df,
    log_y_paper,
    make_paper_facet_figure,
    metric_tag,
    mo,
    paper_dom_metrics,
    paper_domain_panels,
    sel_paper_dom_arch,
):
    # One saved figure per metric. Mixture is the facet dimension, so the data-source
    # slot in the filename records the whole set rather than a single tag.
    if not export_btn_paper.value:
        _out = mo.md("*Click **Export images** to write this column's PDFs.*")
    else:
        _fig = make_paper_facet_figure(
            finished_df,
            paper_domain_panels(["dclm", "starcoder", "pes2o", "wiki"]),
            paper_dom_metrics,
            include_archs=sel_paper_dom_arch.value,
            arch_order=["dense", "moe16", "moe32", "moe64", "moe128"],
            log_y=log_y_paper.value,
        )
        # Panels can each use a different metric, so the metric slot lists them in
        # panel order (collapsed to one name when they all agree).
        _tags = [metric_tag(_m) for _m in paper_dom_metrics]
        _metric_part = _tags[0] if len(set(_tags)) == 1 else "_".join(_tags)
        _paths = [
            export_pdf(
                _fig, "paper", "single_domains", "80M",
                arch_tag(sel_paper_dom_arch.value), _metric_part,
            )
        ]
        _out = export_report(_paths)
    _out
    return


@app.cell(hide_code=True)
def _(
    arch_tag,
    export_btn_paper,
    export_pdf,
    export_report,
    finished_df,
    log_y_paper,
    make_paper_facet_figure,
    metric_tag,
    mo,
    paper_scale_panels,
    sel_paper_scale,
    sel_paper_scale_arch,
    sel_paper_scale_sizes,
):
    # One saved figure per selected metric. Scale is the facet dimension, so the
    # filename carries the full set of sizes in place of a single scale.
    if not export_btn_paper.value:
        _out = mo.md("*Click **Export images** to write this column's PDFs.*")
    else:
        _sizes = "_".join(
            s for s in ["10M", "80M", "200M", "1B"] if s in sel_paper_scale_sizes.value
        ) or "none"
        _paths = []
        for _m in sel_paper_scale.value:
            _fig = make_paper_facet_figure(
                finished_df,
                paper_scale_panels(sel_paper_scale_sizes.value),
                _m,
                include_archs=sel_paper_scale_arch.value,
                arch_order=["dense", "moe16", "moe32", "moe64", "moe128"],
                log_y=log_y_paper.value,
            )
            _paths.append(
                export_pdf(
                    _fig, "paper", "olmo_mix", _sizes,
                    arch_tag(sel_paper_scale_arch.value), metric_tag(_m),
                )
            )
        _out = export_report(_paths)
    _out
    return


@app.cell(hide_code=True)
def _(
    arch_tag,
    export_btn_paper,
    export_pdf,
    export_report,
    finished_df,
    log_y_paper,
    make_paper_famA_figure,
    metric_tag,
    mo,
    sel_paper_famA,
    sel_paper_famA_arch,
):
    if not export_btn_paper.value:
        _out = mo.md("*Click **Export images** to write this column's PDFs.*")
    else:
        _paths = []
        for _m in sel_paper_famA.value:
            _fig = make_paper_famA_figure(
                finished_df,
                _m,
                include_archs=sel_paper_famA_arch.value,
                log_y=log_y_paper.value,
            )
            _paths.append(
                export_pdf(
                    _fig, "paper", "famA", "80M",
                    arch_tag(sel_paper_famA_arch.value), metric_tag(_m),
                )
            )
        _out = export_report(_paths)
    _out
    return


@app.cell(hide_code=True)
def _(
    ARCH_COLOR,
    ARCH_ORDER_DEFAULT,
    DATA_DISPLAY_NAMES,
    MODEL_DISPLAY_NAME,
    apply_log_yaxis,
    apply_pow2_xaxis,
    arch_dash,
    axis_label,
    collect_baseline_points,
    filter_rep_points,
    keep_arch,
    plt,
):
    def make_paper_facet_figure(
        df,
        panels,
        metric,
        include_archs=None,
        exclude_archs=None,
        include_reps=None,
        exclude_reps=None,
        arch_order=None,
        log_y=False,
    ):
        """Loss vs. repetition, one subfigure per panel.

        ``panels`` is an ordered list of ``(data_tag, scale, title)`` — the facet
        dimension is whatever varies across it, so the same function draws both the
        across-scale figure (one mixture, many scales) and the across-domain figure
        (one scale, many mixtures).

        Architecture is encoded exactly as in Group 1 (color + line style). Each
        subfigure autoscales its own y axis; a single figure-level legend covers all
        of them, so the panels carry no per-axes legends.
        """
        from matplotlib.lines import Line2D

        panels = list(panels or [])
        # `metric` is either one metric for every panel, or one per panel (used by
        # the single-domain figure, where each domain is judged on its own eval set).
        if isinstance(metric, str):
            panel_metrics = [metric] * len(panels)
        else:
            panel_metrics = list(metric)
            if len(panel_metrics) != len(panels):
                raise ValueError(
                    f"got {len(panel_metrics)} metrics for {len(panels)} panels"
                )
        n_cols = max(1, len(panels))
        fig, axes = plt.subplots(
            1, n_cols, figsize=(4 * n_cols, 4), squeeze=False
        )

        shown_archs = []
        for col, (panel_tag, panel_scale, panel_title) in enumerate(panels):
            ax = axes[0][col]
            panel_metric = panel_metrics[col]
            max_reps = 0
            any_data = False
            for model_type in arch_order or ARCH_ORDER_DEFAULT:
                if not keep_arch(model_type, include_archs, exclude_archs):
                    continue
                points = collect_baseline_points(
                    df, panel_tag, panel_scale, model_type, panel_metric
                )
                points = filter_rep_points(points, include_reps, exclude_reps)
                if not points:
                    continue
                any_data = True
                if model_type not in shown_archs:
                    shown_archs.append(model_type)
                max_reps = max(max_reps, max(p[0] for p in points))
                ax.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                    linestyle=arch_dash(model_type),
                    color=ARCH_COLOR[model_type],
                )
            apply_pow2_xaxis(ax, max_reps, any_data)
            if log_y:
                apply_log_yaxis(ax)
            ax.set_title(panel_title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Repetition Rate")
            # Every panel is labelled: the y axes are independent, so each one needs
            # its own label to be readable in isolation.
            ax.set_ylabel(axis_label(panel_metric))
            if not any_data:
                ax.set_title(f"{panel_title}  [no data]", fontsize=9)

        # One legend for the whole figure, in canonical architecture order.
        arch_shown = [a for a in (arch_order or ARCH_ORDER_DEFAULT) if a in shown_archs]
        if arch_shown:
            handles = [Line2D([0], [0], linestyle="none", marker="")] + [
                Line2D(
                    [0], [0],
                    color=ARCH_COLOR[a], linewidth=1.8, linestyle=arch_dash(a),
                )
                for a in arch_shown
            ]
            labels = ["Model type"] + [
                MODEL_DISPLAY_NAME.get(a, a) for a in arch_shown
            ]
            legend = fig.legend(
                handles,
                labels,
                fontsize=8,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                borderaxespad=0.0,
                handlelength=2.2,
            )
            for text in legend.get_texts():
                if text.get_text() == "Model type":
                    text.set_fontweight("bold")

        fig.tight_layout()
        return fig

    def paper_scale_panels(sizes, data_tag="olmo_mix"):
        """Panels for the across-scale figure: one mixture, one panel per size."""
        return [
            (data_tag, sc, sc)
            for sc in ["10M", "80M", "200M", "1B"]
            if sc in (sizes or [])
        ]

    def paper_domain_panels(domains, scale="80M"):
        """Panels for the across-domain figure: one scale, one panel per mixture."""
        return [
            (d, scale, DATA_DISPLAY_NAMES.get(d, d))
            for d in ["dclm", "starcoder", "pes2o", "wiki"]
            if d in (domains or [])
        ]

    return make_paper_facet_figure, paper_domain_panels, paper_scale_panels


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo, scale_multiselect):
    sel_paper_scale = metric_multiselect(
        "paper: loss vs. repetition — metrics",
        chosen_values=["eval/lm/dolma_common-crawl-validation/CE loss"],
    )
    sel_paper_scale_sizes = scale_multiselect("paper: loss vs. repetition — model sizes")
    sel_paper_scale_arch = arch_multiselect(
        "paper: loss vs. repetition — architectures",
        options=["dense", "moe16", "moe32", "moe64", "moe128"],
        chosen_values=["dense", "moe32", "moe64"],
    )
    mo.vstack([sel_paper_scale, sel_paper_scale_sizes, sel_paper_scale_arch])
    return sel_paper_scale, sel_paper_scale_arch, sel_paper_scale_sizes


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_paper,
    make_paper_facet_figure,
    paper_scale_panels,
    render_side_by_side,
    sel_paper_scale,
    sel_paper_scale_arch,
    sel_paper_scale_sizes,
):
    _out = render_side_by_side(
        sel_paper_scale.value,
        lambda m: make_paper_facet_figure(
            finished_df,
            paper_scale_panels(sel_paper_scale_sizes.value),
            m,
            include_archs=sel_paper_scale_arch.value,
            arch_order=["dense", "moe16", "moe32", "moe64", "moe128"],
            log_y=log_y_paper.value,
        ),
        "paper_loss_vs_rep",
    )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Loss vs. repetition, across single-domain mixtures.** The Group 2 mixtures at
    80M, one subfigure each. Same conventions as the figure above: independent y
    axes per panel, one legend for the whole figure, one saved file per metric.
    """)
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_dropdown, mo):
    # One metric per panel: each domain defaults to its own held-out eval set, which
    # is the comparison the figure is usually making.
    sel_paper_dom_dclm = metric_dropdown(
        "DCLM panel — metric", "eval/lm/dolma_common-crawl-validation/CE loss"
    )
    sel_paper_dom_sc = metric_dropdown(
        "StarCoder panel — metric", "eval/lm/dolma_stack-validation/CE loss"
    )
    sel_paper_dom_p2o = metric_dropdown(
        "peS2o panel — metric", "eval/lm/dolma_pes2o-validation/CE loss"
    )
    sel_paper_dom_wiki = metric_dropdown(
        "Wikipedia panel — metric", "eval/lm/dolma_wiki-validation/CE loss"
    )
    sel_paper_dom_arch = arch_multiselect(
        "paper: single domains — architectures",
        options=["dense", "moe16", "moe32", "moe64", "moe128"],
        chosen_values=["dense", "moe32", "moe64"],
    )
    mo.vstack([
        sel_paper_dom_dclm,
        sel_paper_dom_sc,
        sel_paper_dom_p2o,
        sel_paper_dom_wiki,
        sel_paper_dom_arch,
    ])
    return (
        sel_paper_dom_arch,
        sel_paper_dom_dclm,
        sel_paper_dom_p2o,
        sel_paper_dom_sc,
        sel_paper_dom_wiki,
    )


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_paper,
    make_paper_facet_figure,
    paper_domain_panels,
    sel_paper_dom_arch,
    sel_paper_dom_dclm,
    sel_paper_dom_p2o,
    sel_paper_dom_sc,
    sel_paper_dom_wiki,
):
    # One figure: the panel set is fixed at the four Group 2 mixtures, each drawn
    # against its own metric. The arch dropdown selects the lines within every panel.
    paper_dom_metrics = [
        sel_paper_dom_dclm.value,
        sel_paper_dom_sc.value,
        sel_paper_dom_p2o.value,
        sel_paper_dom_wiki.value,
    ]
    _out = make_paper_facet_figure(
        finished_df,
        paper_domain_panels(["dclm", "starcoder", "pes2o", "wiki"]),
        paper_dom_metrics,
        include_archs=sel_paper_dom_arch.value,
        arch_order=["dense", "moe16", "moe32", "moe64", "moe128"],
        log_y=log_y_paper.value,
    )
    _out
    return (paper_dom_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **famB mixture sweeps.** The Group 5 families, with each selected metric as its
    own subfigure. Color is the data source, line style + shade the architecture, and
    the dotted line is the `dclm` rep-1 reference. Independent y axes, one legend.
    """)
    return


@app.cell(hide_code=True)
def _(
    FAMB_P2O_SOURCES,
    FAMB_P2O_TITLE,
    FAMB_SC_SOURCES,
    FAMB_SC_TITLE,
    arch_tag,
    export_btn_paper,
    export_pdf,
    export_report,
    finished_df,
    log_y_paper,
    make_paper_famB_figure,
    metric_tag,
    mo,
    sel_paper_famB_p2o,
    sel_paper_famB_p2o_arch,
    sel_paper_famB_sc,
    sel_paper_famB_sc_arch,
):
    # One saved figure per family; metric is the facet dimension, so the metric slot
    # lists the panels in order.
    if not export_btn_paper.value:
        _out = mo.md("*Click **Export images** to write this column's PDFs.*")
    else:
        _specs = [
            ("starcoder", FAMB_SC_SOURCES, FAMB_SC_TITLE,
             sel_paper_famB_sc, sel_paper_famB_sc_arch),
            ("pes2o", FAMB_P2O_SOURCES, FAMB_P2O_TITLE,
             sel_paper_famB_p2o, sel_paper_famB_p2o_arch),
        ]
        _paths = []
        for _fam, _srcs, _title, _sel, _arch in _specs:
            if not _sel.value:
                continue
            _fig = make_paper_famB_figure(
                finished_df,
                _srcs,
                _sel.value,
                source_title=_title,
                include_archs=_arch.value,
                arch_order=["dense", "moe16", "moe32", "moe64", "moe128"],
                exclude_reps=[128, 256],
                log_y=log_y_paper.value,
            )
            _paths.append(
                export_pdf(
                    _fig, "paper", f"famB_{_fam}", "80M",
                    arch_tag(_arch.value),
                    "_".join(metric_tag(_m) for _m in _sel.value),
                )
            )
        _out = export_report(_paths)
    _out
    return


@app.cell(hide_code=True)
def _(
    ARCH_ORDER_DEFAULT,
    DATA_DISPLAY_NAMES,
    MODEL_DISPLAY_NAME,
    apply_log_yaxis,
    apply_pow2_xaxis,
    arch_color,
    arch_dash,
    axis_label,
    collect_baseline_points,
    collect_famB_points,
    factor_palette,
    filter_rep_points,
    keep_arch,
    plt,
):
    def make_paper_famB_figure(
        df,
        sources,
        metrics,
        source_title="Data source",
        include_archs=None,
        exclude_archs=None,
        include_reps=None,
        exclude_reps=None,
        arch_order=None,
        log_y=False,
    ):
        """famB mixture sweep, one subfigure per metric.

        Same encoding as the Group 5 plots — color is the data source, line style +
        shade the architecture, and an ``"hline"`` source becomes a dotted reference
        line annotated inline. Faceting is by *metric* here, so each panel carries its
        own y label and the panels do not share a y axis.

        A single figure-level legend covers both sections (architecture and source);
        the panels carry no legends of their own.
        """
        from matplotlib.lines import Line2D

        metrics = list(metrics or [])
        n_cols = max(1, len(metrics))
        fig, axes = plt.subplots(1, n_cols, figsize=(3.7 * n_cols, 3.7), squeeze=False)

        source_labels = [src[0] for src in sources]
        source_color = factor_palette(source_labels)
        shown_archs, shown_sources = [], []

        for col, metric in enumerate(metrics):
            ax = axes[0][col]
            hlines = []
            max_reps = 0
            any_data = False
            for label, _color, kind, key in sources:
                if kind == "hline":
                    _ref_pts = collect_baseline_points(df, key, "80M", "dense", metric)
                    _ref = next((v for r, v in _ref_pts if float(r) == 1.0), None)
                    if _ref is not None:
                        ax.axhline(
                            _ref, linestyle=":", linewidth=1.2,
                            color=source_color[label], zorder=1,
                        )
                        hlines.append((label, key, _ref))
                    continue
                for model_type in arch_order or ARCH_ORDER_DEFAULT:
                    if not keep_arch(model_type, include_archs, exclude_archs):
                        continue
                    if kind == "baseline":
                        points = collect_baseline_points(
                            df, key, "80M", model_type, metric
                        )
                    else:
                        points = collect_famB_points(df, key, model_type, metric)
                    points = filter_rep_points(points, include_reps, exclude_reps)
                    if not points:
                        continue
                    any_data = True
                    if model_type not in shown_archs:
                        shown_archs.append(model_type)
                    if label not in shown_sources:
                        shown_sources.append(label)
                    max_reps = max(max_reps, max(p[0] for p in points))
                    ax.plot(
                        [p[0] for p in points],
                        [p[1] for p in points],
                        marker="o",
                        markersize=3,
                        linewidth=1.5,
                        linestyle=arch_dash(model_type),
                        color=arch_color(source_color[label], model_type),
                    )
            apply_pow2_xaxis(ax, max_reps, any_data)
            if log_y:
                apply_log_yaxis(ax)
            for _hl_label, _hl_key, _hl_y in hlines:
                ax.text(
                    0.012,
                    _hl_y,
                    f"{DATA_DISPLAY_NAMES.get(_hl_key, _hl_key)} (R=1)",
                    transform=ax.get_yaxis_transform(),
                    color=source_color[_hl_label],
                    fontsize=7,
                    ha="left",
                    va="bottom",
                )

            ax.set_title(axis_label(metric), fontsize=11, fontweight="bold")
            ax.set_xlabel("Repetition Rate")
            # The facet dimension is the metric, so the y label names the panel --
            # a separate title would just repeat it.
            ax.set_ylabel(axis_label(metric))

        # One legend for the figure: architecture section, then source section.
        def _header():
            return Line2D([0], [0], linestyle="none", marker="", label="")

        arch_shown = [a for a in (arch_order or ARCH_ORDER_DEFAULT) if a in shown_archs]
        src_shown = [s for s in source_labels if s in shown_sources]
        if arch_shown or src_shown:
            handles = [_header()] + [
                Line2D(
                    [0], [0],
                    color=arch_color("#333333", a),
                    linewidth=1.6,
                    linestyle=arch_dash(a),
                )
                for a in arch_shown
            ]
            labels = ["Model type"] + [MODEL_DISPLAY_NAME.get(a, a) for a in arch_shown]
            if src_shown:
                handles += [_header()] + [
                    Line2D([0], [0], color=source_color[l], linewidth=2)
                    for l in src_shown
                ]
                labels += [source_title] + list(src_shown)
            legend = fig.legend(
                handles,
                labels,
                fontsize=8,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                borderaxespad=0.0,
                handlelength=2.2,
            )
            for text in legend.get_texts():
                if text.get_text() in ("Model type", source_title):
                    text.set_fontweight("bold")

        fig.tight_layout()
        return fig

    return (make_paper_famB_figure,)


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_paper_famB_sc = metric_multiselect(
        "paper: famB StarCoder — metrics",
        chosen_values=[
            "eval/lm/dolma_common-crawl-validation/CE loss",
            "eval/lm/dolma_stack-validation/CE loss",
        ],
    )
    sel_paper_famB_sc_arch = arch_multiselect(
        "paper: famB StarCoder — architectures",
        options=["dense", "moe16", "moe32", "moe64", "moe128"],
        chosen_values=["dense", "moe64"],
    )
    mo.vstack([sel_paper_famB_sc, sel_paper_famB_sc_arch])
    return sel_paper_famB_sc, sel_paper_famB_sc_arch


@app.cell(hide_code=True)
def _(
    FAMB_SC_SOURCES,
    FAMB_SC_TITLE,
    finished_df,
    log_y_paper,
    make_paper_famB_figure,
    sel_paper_famB_sc,
    sel_paper_famB_sc_arch,
):
    _out = make_paper_famB_figure(
        finished_df,
        FAMB_SC_SOURCES,
        sel_paper_famB_sc.value,
        source_title=FAMB_SC_TITLE,
        include_archs=sel_paper_famB_sc_arch.value,
        arch_order=["dense", "moe16", "moe32", "moe64", "moe128"],
        exclude_reps=[128, 256],
        log_y=log_y_paper.value,
    )
    _out
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_paper_famB_p2o = metric_multiselect(
        "paper: famB peS2o — metrics",
        chosen_values=[
            "eval/lm/dolma_common-crawl-validation/CE loss",
            "eval/lm/dolma_pes2o-validation/CE loss",
        ],
    )
    sel_paper_famB_p2o_arch = arch_multiselect(
        "paper: famB peS2o — architectures",
        options=["dense", "moe16", "moe32", "moe64", "moe128"],
        chosen_values=["dense", "moe64"],
    )
    mo.vstack([sel_paper_famB_p2o, sel_paper_famB_p2o_arch])
    return sel_paper_famB_p2o, sel_paper_famB_p2o_arch


@app.cell(hide_code=True)
def _(
    FAMB_P2O_SOURCES,
    FAMB_P2O_TITLE,
    finished_df,
    log_y_paper,
    make_paper_famB_figure,
    sel_paper_famB_p2o,
    sel_paper_famB_p2o_arch,
):
    _out = make_paper_famB_figure(
        finished_df,
        FAMB_P2O_SOURCES,
        sel_paper_famB_p2o.value,
        source_title=FAMB_P2O_TITLE,
        include_archs=sel_paper_famB_p2o_arch.value,
        arch_order=["dense", "moe16", "moe32", "moe64", "moe128"],
        exclude_reps=[128, 256],
        log_y=log_y_paper.value,
    )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **famA filtering sweep.** The two Group 4 views side by side: loss vs. %dclm
    (colored by repetition) and loss vs. repetition (colored by %dclm). One figure per
    selected metric.

    Each panel keeps its own legend here — unlike the other paper figures, the two
    panels encode *different* variables in color, so a single shared key would be
    wrong.
    """)
    return


@app.cell(hide_code=True)
def _(arch_multiselect, metric_multiselect, mo):
    sel_paper_famA = metric_multiselect(
        "paper: famA — metrics",
        chosen_values=["eval/lm/dolma_common-crawl-validation/CE loss"],
    )
    sel_paper_famA_arch = arch_multiselect(
        "paper: famA — architectures",
        options=["dense", "moe16", "moe32", "moe64", "moe128"],
        chosen_values=["dense", "moe32", "moe64"],
    )
    mo.vstack([sel_paper_famA, sel_paper_famA_arch])
    return sel_paper_famA, sel_paper_famA_arch


@app.cell(hide_code=True)
def _(make_famA_figure, make_famA_rep_figure, plt):
    def make_paper_famA_figure(
        df, metric, include_archs=None, exclude_archs=None, log_y=False
    ):
        """The two famA views as subfigures of one figure.

        Both panels are drawn by the same functions that back the Group 4 plots, via
        their ``ax`` argument, so the two stay in sync by construction.
        """
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        make_famA_figure(
            df, metric, include_archs=include_archs,
            exclude_archs=exclude_archs, log_y=log_y, ax=axes[0],
        )
        make_famA_rep_figure(
            df, metric, include_archs=include_archs,
            exclude_archs=exclude_archs, log_y=log_y, ax=axes[1],
        )
        fig.tight_layout()
        return fig

    return (make_paper_famA_figure,)


@app.cell(hide_code=True)
def _(
    finished_df,
    log_y_paper,
    make_paper_famA_figure,
    render_side_by_side,
    sel_paper_famA,
    sel_paper_famA_arch,
):
    _out = render_side_by_side(
        sel_paper_famA.value,
        lambda m: make_paper_famA_figure(
            finished_df,
            m,
            include_archs=sel_paper_famA_arch.value,
            log_y=log_y_paper.value,
        ),
        "paper_famA",
    )
    _out
    return


if __name__ == "__main__":
    app.run()
