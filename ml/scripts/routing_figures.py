"""
Figures for the paper's mechanistic-interpretation section, built from
routing_analysis.py outputs (ossification.json / knockout.json /
co_activation.json per run).

Usage:
    python routing_figures.py \
        --analysis_root /scratch/.../analysis/routing_analysis \
        --out_dir /scratch/.../data_rep_moe_paper/fig_pdfs/routing

Reads every <analysis_root>/<run_name>/ that has all three JSONs, parses
(size, arch, R, dropout) from the run name, pulls (num_experts, top_k) from
the run's checkpoint config.json when available, and emits the section's
PDFs in the house style (whitegrid, viridis, log2 R axis).
"""

import argparse
import json
import math
import os
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
    "axes.grid": True,
    "grid.color": "0.85",
    "axes.edgecolor": "0.7",
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
})

LINESTYLES = ["-", "-.", "--", ":", (0, (1, 1))]


def viridis(n):
    cmap = plt.get_cmap("viridis")
    return [cmap(x) for x in np.linspace(0.05, 0.85, n)]


def parse_run(name):
    m = re.search(r"olmo2_ml_(80M|200M|1B)_((?:dense|moe\d+s?\d*))_rep(\d+)x(?:_dropout=([\d.]+))?", name)
    if not m:
        return None
    size, arch, rep, drop = m.groups()
    return {"size": size, "arch": arch, "rep": int(rep), "dropout": float(drop) if drop else 0.0}


def load_all(analysis_root, models_root=None, exclude=()):
    runs = []
    for d in sorted(os.listdir(analysis_root)):
        p = os.path.join(analysis_root, d)
        if not all(os.path.exists(os.path.join(p, f + ".json"))
                   for f in ("ossification", "knockout", "co_activation")):
            continue
        meta = parse_run(d)
        if meta is None:
            continue
        if any(x in d for x in exclude):
            continue
        meta["name"] = d
        meta["oss"] = json.load(open(os.path.join(p, "ossification.json")))
        meta["ko"] = json.load(open(os.path.join(p, "knockout.json")))
        meta["co"] = json.load(open(os.path.join(p, "co_activation.json")))
        # num_experts / top_k from the final checkpoint's config, if reachable
        meta["num_experts"] = meta["top_k"] = None
        if models_root:
            hits = [q for q in
                    (os.path.join(models_root, s, d) for s in os.listdir(models_root))
                    if os.path.isdir(q)]
            for q in hits:
                steps = [x for x in os.listdir(q) if re.match(r"step\d+$", x)]
                if not steps:
                    continue
                last = max(steps, key=lambda s: int(s[4:]))
                cfg_p = os.path.join(q, last, "config.json")
                if os.path.exists(cfg_p):
                    cfg = json.load(open(cfg_p))
                    moe = cfg["model"]["block"].get("feed_forward_moe")
                    if moe:
                        meta["num_experts"] = moe["num_experts_list"][0]
                        meta["top_k"] = moe["routers_list"][0]["top_k"]
                break
        runs.append(meta)
    return runs


ARCH_FALLBACK = {
    "moe16": (16, 4), "moe32": (32, 4), "moe64": (64, 4), "moe128": (128, 4),
    "moe64s8": (64, 8), "moe64s32": (64, 2),
}


def arch_label(runs_of_family):
    r = runs_of_family[0]
    n, k = r["num_experts"], r["top_k"]
    if not (n and k):
        n, k = ARCH_FALLBACK.get(r["arch"], (None, None))
    if n and k:
        return f"MoE ({n} x 1/{k})"
    return r["arch"]


def family_key(r):
    return (r["size"], r["arch"], r["dropout"])


def mean_stability(oss, idx, keys):
    return float(np.mean([oss["stability_rates"][k][i] for k in keys for i in [idx]]))


def modal_gap_indices(oss):
    pairs = oss["step_pairs"]
    gaps = [b - a for a, b in pairs]
    modal = Counter(gaps).most_common(1)[0][0]
    return [i for i, g in enumerate(gaps) if g == modal], modal


def fig_ossification_curves(runs, out_dir, size, arch, dropout=0.0, gap=None):
    fam = sorted([r for r in runs if family_key(r) == (size, arch, dropout)], key=lambda r: r["rep"])
    if gap is not None:
        fam = [r for r in fam if run_modal_gap(r) == gap]
    if not fam:
        return None
    colors = viridis(len(fam))
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for c, r in zip(colors, fam):
        oss = r["oss"]
        keys = list(oss["stability_rates"])
        steps = oss["steps"]
        mean = [float(np.mean([oss["stability_rates"][k][i] for k in keys]))
                for i in range(len(steps))]
        ax.plot(steps, mean, marker="o", ms=4.5, lw=1.8, color=c, label=f'{r["rep"]}x')
    ax.set_xlabel("Training step")
    ax.set_ylabel("Top-1 routing stability")
    ax.set_ylim(0, 1.0)
    leg = ax.legend(title="Repetition", loc="lower right", frameon=True)
    leg.get_title().set_fontweight("bold")
    fig.tight_layout()
    path = os.path.join(out_dir, f"routing_ossification_curves__{size}__{arch}__dolma_cc.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def run_modal_gap(r):
    gaps = [b - a for a, b in r["oss"]["step_pairs"]]
    return Counter(gaps).most_common(1)[0][0]


def fig_ossification_vs_rep(runs, out_dir, families, fname, late=True, legend_inside=False,
                            gap=None):
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    colors = viridis(len(families))
    all_reps = set()
    for (famk, label), c, ls in zip(families, colors, LINESTYLES * 3):
        fam = sorted([r for r in runs if family_key(r) == famk], key=lambda r: r["rep"])
        if gap is not None:
            fam = [r for r in fam if run_modal_gap(r) == gap]
        elif fam:
            fam_gap = Counter(run_modal_gap(r) for r in fam).most_common(1)[0][0]
            fam = [r for r in fam if run_modal_gap(r) == fam_gap]
        if not fam:
            continue
        reps, vals = [], []
        for r in fam:
            oss = r["oss"]
            keys = list(oss["stability_rates"])
            idx, _ = modal_gap_indices(oss)
            use = idx[-2:] if late else idx[:2]
            vals.append(float(np.mean([oss["stability_rates"][k][i] for k in keys for i in use])))
            reps.append(r["rep"])
        ax.plot(reps, vals, marker="o", ms=6, lw=2, ls=ls, color=c, label=label)
        all_reps.update(reps)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(all_reps))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Repetition Rate")
    ax.set_ylabel("Routing stability, end of training")
    if legend_inside:
        leg = ax.legend(title="Model type", frameon=True, loc=legend_inside
                        if isinstance(legend_inside, str) else "lower right")
    else:
        leg = ax.legend(title="Model type", frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    leg.get_title().set_fontweight("bold")
    path = os.path.join(out_dir, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_knockout_vs_rep(runs, out_dir, families, fname, stat="median"):
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    colors = viridis(len(families))
    all_reps = set()
    for spec, c, ls in zip(families, colors, LINESTYLES * 3):
        famk, label = spec[0], spec[1]
        max_rep = spec[2] if len(spec) > 2 else None
        fam = sorted([r for r in runs if family_key(r) == famk], key=lambda r: r["rep"])
        if not fam:
            continue
        reps, vals = [], []
        for r in fam:
            inc = [v["loss_increase"] for kk in r["ko"]["knockout_losses"].values()
                   for v in kk.values()]
            vals.append(float(np.median(inc)) if stat == "median" else float(np.mean(inc)))
            reps.append(r["rep"])
        if max_rep is not None:
            line = [(x, y) for x, y in zip(reps, vals) if x <= max_rep]
            loose = [(x, y) for x, y in zip(reps, vals) if x > max_rep]
            ax.plot([x for x, _ in line], [y for _, y in line],
                    marker="o", ms=6, lw=2, ls=ls, color=c, label=label)
            if loose:
                ax.plot([x for x, _ in loose], [y for _, y in loose],
                        marker="o", ms=6, lw=0, color=c)
        else:
            ax.plot(reps, vals, marker="o", ms=6, lw=2, ls=ls, color=c, label=label)
        all_reps.update(reps)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(all_reps))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Repetition Rate")
    ax.set_ylabel(f"Expert knockout: {stat} CE increase")
    leg = ax.legend(title="Model type", frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    leg.get_title().set_fontweight("bold")
    path = os.path.join(out_dir, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_coactivation_vs_rep(runs, out_dir, families, fname):
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    colors = viridis(len(families))
    all_reps = set()
    for (famk, label), c, ls in zip(families, colors, LINESTYLES * 3):
        fam = sorted([r for r in runs if family_key(r) == famk], key=lambda r: r["rep"])
        if not fam:
            continue
        reps, vals = [], []
        for r in fam:
            ents = []
            for key, v in r["co"].items():
                E = len(v["co_activation_matrix"])
                hmax = math.log(E * (E - 1)) if E > 1 else 1.0
                ents.append(v["co_activation_entropy"] / hmax)
            if not ents:
                continue
            vals.append(float(np.mean(ents)))
            reps.append(r["rep"])
        ax.plot(reps, vals, marker="o", ms=6, lw=2, ls=ls, color=c, label=label)
        all_reps.update(reps)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(all_reps))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Repetition Rate")
    ax.set_ylabel("Co-activation entropy (normalized)")
    leg = ax.legend(title="Model type", frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    leg.get_title().set_fontweight("bold")
    path = os.path.join(out_dir, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_root", required=True)
    ap.add_argument("--models_root", default=None,
                    help="Root of <sweep>/<run>/stepN checkpoint dirs, for arch labels")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Skip analyzed runs whose name contains any of these substrings")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = load_all(args.analysis_root, args.models_root, exclude=args.exclude)
    print(f"Loaded {len(runs)} analyzed runs")

    def fams(size, dropout, archs):
        out = []
        for a in archs:
            fam = [r for r in runs if family_key(r) == (size, a, dropout)]
            if fam:
                out.append(((size, a, dropout), arch_label(fam)))
        return out

    made = []
    # (a) ossification curves: 200M core moe64 no-dropout (klone ladder, 200-step
    # checkpoints), plus the 80M moe128 shape panel for the appendix
    made.append(fig_ossification_curves(runs, args.out_dir, "200M", "moe64", gap=200))
    made.append(fig_ossification_curves(runs, args.out_dir, "80M", "moe128"))
    # (b) end-of-training stability vs R, no-dropout core + famC/granularity,
    # all at 200-step checkpoint spacing
    f80 = fams("80M", 0.0, ["moe16", "moe32", "moe64", "moe64s8", "moe64s32", "moe128"])
    fcore = (fams("200M", 0.0, ["moe32", "moe64"])
             + fams("80M", 0.0, ["moe32", "moe64"]))
    fcore = [(k, l + (" (200M)" if k[0] == "200M" else " (80M)")) for k, l in fcore]
    made.append(fig_ossification_vs_rep(runs, args.out_dir, fcore,
                "routing_ossification_vs_rep__core__dolma_cc.pdf", gap=200,
                legend_inside="upper left"))
    made.append(fig_ossification_vs_rep(runs, args.out_dir, f80,
                "routing_ossification_vs_rep__80M__dolma_cc.pdf", gap=200))
    # dropout ossification contrast: 1,000-step spacing throughout (Marlowe arms
    # + the single no-dropout R64 baseline)
    fdrop = ([(("200M", "moe64", 0.0), "MoE (64 x 1/4)")]
             + [(("200M", "moe64", 0.1), "MoE (64 x 1/4), drop 0.1"),
                (("200M", "moe64", 0.4), "MoE (64 x 1/4), drop 0.4")])
    made.append(fig_ossification_vs_rep(runs, args.out_dir, fdrop,
                "routing_ossification_vs_rep__200M__dropout__dolma_cc.pdf",
                legend_inside=True, gap=1000))
    # knockout vs R (final checkpoint only, spacing-independent)
    made.append(fig_knockout_vs_rep(runs, args.out_dir, f80,
                "expert_knockout_vs_rep__80M__dolma_cc.pdf"))
    f200 = ([(k, l, 32) for k, l in fams("200M", 0.0, ["moe32", "moe64"])]
            + [(("200M", "moe64", 0.1), "MoE (64 x 1/4), drop 0.1"),
               (("200M", "moe64", 0.4), "MoE (64 x 1/4), drop 0.4")])
    made.append(fig_knockout_vs_rep(runs, args.out_dir, f200,
                "expert_knockout_vs_rep__200M__dropout__dolma_cc.pdf"))
    # co-activation, 80M
    made.append(fig_coactivation_vs_rep(runs, args.out_dir, f80,
                "expert_coactivation_vs_rep__80M__dolma_cc.pdf"))
    for p in made:
        if p:
            print("wrote", p)
