"""Self-contained historical W&B downloader for this project's runs.

Downloads run metadata (config, summary, tags) and, optionally, the full
per-step history for every run in a W&B project, writing everything to plain
files under ``ml/runs_data`` by default. Uses only the public ``wandb`` API --
no external export framework required.

The CSV it writes (``wandb_export.csv``) is column-compatible with the manual
W&B web-UI export that ``ml/notebooks/regularization_figures.py`` already reads:
flat dotted config columns (e.g. ``model.block.dropout``,
``train_module.max_grad_norm``, ``model.block.feed_forward_moe.routers_list`` as
a JSON string), ``Name`` / ``State`` / ``Tags`` / ``ID``, and one column per
final-summary metric with its raw name (``train/CE loss``,
``eval/lm/*/CE loss`` ...). So the downloader is a drop-in replacement for the
hand-downloaded CSVs in ``ml/notebooks/data``.

Typical usage::

    # metadata + final-summary metrics for the whole project -> ml/runs_data/
    python -m ml.wandb_download.download --entity ml-moe --project data_rep_moe

    # add full per-step training/eval curves
    python -m ml.wandb_download.download --include-history

    # restrict to the current regularizer sweep by tag / name
    python -m ml.wandb_download.download --tag MoE64 --name-regex data_rep_AC

Metadata (``wandb_export.csv`` / ``runs.jsonl``) is always rebuilt from scratch
so the snapshot is current. History download is partially incremental: history
for terminal runs (finished/failed/crashed) that already have a non-empty file
on disk is skipped, since it can no longer change. Pass ``--refresh-history`` to
force a full re-fetch. History for still-running runs is always re-downloaded.

The output directory (default ``ml/runs_data``) will contain:

    wandb_export.csv        wide, web-export-compatible run table (read by the
                            regularization_figures notebook)
    runs.jsonl              one JSON object per run with the full nested config,
                            summary, tags, and convenience flat columns
    history/<run_id>.jsonl  per-step history rows (only with --include-history)
    history.csv             all history rows concatenated (only with --history-table)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("wandb_download")

# Project defaults (see ml/scripts/constants.py :: PROJECT_SPECS).
DEFAULT_ENTITY = "ml-moe"
DEFAULT_PROJECT = "data_rep_moe"

# Default deposit location: ml/runs_data/ (download.py lives in ml/wandb_download/).
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "runs_data"

# The canonical CSV name; the notebook picks the most recently modified CSV in
# the directory, so overwriting this on each run keeps it the "latest".
WIDE_CSV_NAME = "wandb_export.csv"

# Run states whose history is final and will never change again. History for
# runs in one of these states can be skipped on re-download if it already exists
# on disk (see --refresh-history to force a full re-fetch anyway).
TERMINAL_STATES = frozenset({"finished", "failed", "crashed", "killed"})

# Config fields worth surfacing as extra flat columns in the JSONL (searched for
# recursively, so we don't hardcode the deep path to the MoE router knobs). The
# wide CSV already flattens the whole config; these just make the JSONL handy.
FLATTEN_KEYS = (
    "jitter_eps",
    "eom_prob",
    "fom_prob",
    "dropout",
    "max_grad_norm",
    "lr",
    "weight_decay",
    "d_model",
    "n_layers",
    "z_loss_weight",
    "lb_loss_weight",
    "bias_gamma",
    "normalize_expert_weights",
)


@dataclass
class DownloadConfig:
    entity: str
    project: str
    output_dir: Path
    include_history: bool
    history_table: bool
    refresh_history: bool
    workers: int
    timeout: int
    tags: list[str] = field(default_factory=list)
    name_regex: str | None = None
    states: list[str] | None = None
    limit: int | None = None


def _configure_logging() -> None:
    level = logging.getLevelNamesMapping().get(
        os.getenv("WANDB_DOWNLOAD_LOG_LEVEL", "INFO").upper(), logging.INFO
    )
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def _to_plain(value: Any) -> Any:
    """Coerce W&B / numpy-ish values into JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Dict-like objects that aren't real dicts (e.g. wandb SummarySubDict, whose
    # __getattr__ raises KeyError on missing attrs). Iterate via items().
    if hasattr(value, "keys") and callable(getattr(value, "items", None)):
        try:
            return {str(k): _to_plain(v) for k, v in value.items()}
        except Exception:  # noqa: BLE001 - fall through to other strategies
            pass
    # numpy scalars, wandb summary leaves, etc. Guard getattr itself, since some
    # objects raise (not return None) for unknown attributes.
    for attr in ("item", "tolist"):
        try:
            fn = getattr(value, attr, None)
        except Exception:  # noqa: BLE001 - unknown-attr access may raise
            fn = None
        if callable(fn):
            try:
                return _to_plain(fn())
            except Exception:  # noqa: BLE001 - best-effort coercion
                pass
    try:
        return str(value)
    except Exception:  # noqa: BLE001 - last-resort
        return None


def _find_recursive(config: Any, key: str) -> Any:
    """Return the first value found for ``key`` anywhere in a nested structure."""
    if isinstance(config, dict):
        if key in config:
            return config[key]
        for v in config.values():
            found = _find_recursive(v, key)
            if found is not None:
                return found
    elif isinstance(config, (list, tuple)):
        for v in config:
            found = _find_recursive(v, key)
            if found is not None:
                return found
    return None


def _flatten_selected(config: dict[str, Any]) -> dict[str, Any]:
    return {key: _to_plain(_find_recursive(config, key)) for key in FLATTEN_KEYS}


def _flatten_dotted(obj: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten a nested config into dotted keys, matching the W&B web export.

    - dict -> recurse, joining keys with '.'
    - list/tuple -> kept as a single compact-JSON string (e.g. ``routers_list``)
    - scalar -> value as-is
    """
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_dotted(v, key, out)
    elif isinstance(obj, (list, tuple)):
        out[prefix] = json.dumps(_to_plain(obj))
    else:
        out[prefix] = obj
    return out


def _build_filters(cfg: DownloadConfig) -> dict[str, Any] | None:
    filters: dict[str, Any] = {}
    if cfg.tags:
        # Require every requested tag to be present.
        filters["tags"] = {"$all": cfg.tags}
    if cfg.states:
        filters["state"] = {"$in": cfg.states}
    if cfg.name_regex:
        filters["display_name"] = {"$regex": cfg.name_regex}
    return filters or None


def fetch_runs(api: Any, cfg: DownloadConfig) -> list[Any]:
    path = f"{cfg.entity}/{cfg.project}"
    logger.info("Querying runs in %s", path)
    runs = api.runs(path, filters=_build_filters(cfg), per_page=500)
    selected = []
    name_re = re.compile(cfg.name_regex) if cfg.name_regex else None
    for run in runs:
        # Server-side regex filtering isn't always honored; enforce locally too.
        if name_re is not None and not name_re.search(run.name or ""):
            continue
        selected.append(run)
        if cfg.limit is not None and len(selected) >= cfg.limit:
            break
    logger.info("Selected %s run(s)", len(selected))
    return selected


def run_to_record(run: Any) -> dict[str, Any]:
    """Rich, nested record for runs.jsonl."""
    config = _to_plain(dict(run.config))
    summary = _to_plain({k: v for k, v in run.summary.items()})
    record = {
        "run_id": run.id,
        "name": run.name,
        "state": run.state,
        "tags": list(run.tags or []),
        "created_at": getattr(run, "created_at", None),
        "group": getattr(run, "group", None),
        "url": getattr(run, "url", None),
        "config": config,
        "summary": summary,
    }
    record.update(_flatten_selected(config))
    return record


def run_to_wide_row(run: Any) -> dict[str, Any]:
    """Flat, web-export-compatible row for wandb_export.csv.

    Columns: ``Name`` / ``State`` / ``Tags`` / ``ID``, every config field as a
    dotted column, and every (scalar) final-summary metric under its raw key.
    """
    row: dict[str, Any] = {
        "Name": run.name,
        "State": run.state,
        "Tags": ", ".join(run.tags or []),
        "ID": run.id,
    }
    # Config -> dotted columns (lists become JSON strings, e.g. routers_list).
    row.update(_flatten_dotted(_to_plain(dict(run.config))))
    # Summary -> raw-named metric columns; skip wandb internals and non-scalars.
    for key, value in _to_plain({k: v for k, v in run.summary.items()}).items():
        if key.startswith("_"):
            continue
        if isinstance(value, (dict, list)):
            continue
        row.setdefault(key, value)
    return row


def _history_path(output_dir: Path, run_id: str) -> Path:
    return output_dir / "history" / f"{run_id}.jsonl"


def _has_cached_history(output_dir: Path, run_id: str) -> bool:
    """True if a non-empty history file already exists for this run."""
    path = _history_path(output_dir, run_id)
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _should_skip_history(run: Any, output_dir: Path, refresh: bool) -> bool:
    """Skip re-downloading history for a terminal run that's already cached.

    Terminal runs (finished/failed/crashed/killed) never produce new steps, so
    an existing non-empty history file is complete. ``refresh`` forces a full
    re-fetch regardless.
    """
    if refresh:
        return False
    return (run.state in TERMINAL_STATES) and _has_cached_history(output_dir, run.id)


def fetch_history(run: Any, output_dir: Path) -> tuple[str, int]:
    """Stream one run's full per-step history to history/<run_id>.jsonl."""
    path = _history_path(output_dir, run.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    rows = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for entry in run.scan_history():
            handle.write(json.dumps(_to_plain(entry)))
            handle.write("\n")
            rows += 1
    tmp.replace(path)
    return run.id, rows


def _write_runs(runs: list[Any], output_dir: Path) -> int:
    # Rich nested JSONL.
    records = [run_to_record(run) for run in runs]
    jsonl_path = output_dir / "runs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")
    logger.info("Wrote %s run(s) to %s", len(records), jsonl_path)

    # Wide, web-export-compatible CSV (read by regularization_figures.py).
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available; skipping %s", WIDE_CSV_NAME)
        return len(records)
    wide_rows = [run_to_wide_row(run) for run in runs]
    df = pd.DataFrame(wide_rows)
    csv_path = output_dir / WIDE_CSV_NAME
    df.to_csv(csv_path, index=False)
    logger.info("Wrote wide run table (%s cols) to %s", df.shape[1], csv_path)
    return len(records)


def _write_history_table(run_ids: Iterable[str], output_dir: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available; skipping history.csv")
        return
    frames = []
    for run_id in run_ids:
        path = _history_path(output_dir, run_id)
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        frame.insert(0, "run_id", run_id)
        frames.append(frame)
    if not frames:
        logger.warning("No history rows to concatenate")
        return
    table = pd.concat(frames, ignore_index=True, sort=False)
    csv_path = output_dir / "history.csv"
    table.to_csv(csv_path, index=False)
    logger.info("Wrote combined history table (%s rows) to %s", len(table), csv_path)


def download(cfg: DownloadConfig) -> dict[str, Any]:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "The 'wandb' package is required. Install it with `pip install wandb` "
            "(this repo declares it as an optional extra)."
        ) from exc

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=cfg.timeout)
    runs = fetch_runs(api, cfg)

    run_count = _write_runs(runs, cfg.output_dir)

    history_rows = 0
    skipped = 0
    if cfg.include_history:
        to_fetch = [
            run
            for run in runs
            if not _should_skip_history(run, cfg.output_dir, cfg.refresh_history)
        ]
        skipped = len(runs) - len(to_fetch)
        if skipped:
            logger.info(
                "Skipping history for %s terminal run(s) already cached "
                "(use --refresh-history to force)",
                skipped,
            )
        logger.info(
            "Downloading history for %s run(s) with %s worker(s)", len(to_fetch), cfg.workers
        )
        completed = 0
        with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as executor:
            futures = {
                executor.submit(fetch_history, run, cfg.output_dir): run.id for run in to_fetch
            }
            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    _run_id, rows = future.result()
                except Exception as exc:  # noqa: BLE001 - report and continue
                    logger.error("Failed history for %s: %s", run_id, exc)
                    continue
                history_rows += rows
                completed += 1
                if completed == 1 or completed % 10 == 0 or completed == len(to_fetch):
                    logger.info("History: %s/%s runs", completed, len(to_fetch))
        if cfg.history_table:
            _write_history_table([run.id for run in runs], cfg.output_dir)

    summary = {
        "entity": cfg.entity,
        "project": cfg.project,
        "output_dir": str(cfg.output_dir),
        "csv_path": str(cfg.output_dir / WIDE_CSV_NAME),
        "run_count": run_count,
        "include_history": cfg.include_history,
        "history_row_count": history_rows if cfg.include_history else None,
        "history_skipped_cached": skipped if cfg.include_history else None,
    }
    return summary


def _parse_args(argv: list[str] | None = None) -> DownloadConfig:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to deposit the export (default: ml/runs_data).",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Also download full per-step history for each run.",
    )
    parser.add_argument(
        "--history-table",
        action="store_true",
        help="Concatenate per-run history into a single history.csv.",
    )
    parser.add_argument(
        "--refresh-history",
        action="store_true",
        help=(
            "Re-download history even for terminal runs already cached on disk. "
            "By default, finished/failed/crashed runs with an existing history "
            "file are skipped (their history is final)."
        ),
    )
    parser.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8) + 4))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        dest="tags",
        help="Only runs with this tag (repeatable; all must match).",
    )
    parser.add_argument("--name-regex", default=None, help="Only runs whose name matches this regex.")
    parser.add_argument(
        "--state",
        action="append",
        default=None,
        dest="states",
        help="Only runs in this state, e.g. finished (repeatable).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of runs (for testing).")
    args = parser.parse_args(argv)
    return DownloadConfig(
        entity=args.entity,
        project=args.project,
        output_dir=args.output_dir,
        include_history=args.include_history,
        history_table=args.history_table,
        refresh_history=args.refresh_history,
        workers=args.workers,
        timeout=args.timeout,
        tags=args.tags,
        name_regex=args.name_regex,
        states=args.states,
        limit=args.limit,
    )


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    cfg = _parse_args(argv)
    summary = download(cfg)
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
