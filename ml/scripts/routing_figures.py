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


def load_all(analysis_root, models_root=None):
    runs = []
    for d in sorted(os.listdir(analysis_root)):
        p = os.path.join(analysis_root, d)
        if not all(os.path.exists(os.path.join(p, f + ".json"))
                   for f in ("ossification", "knockout", "co_activation")):
            continue
        meta = parse_run(d)
        if meta is None:
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


def arch_label(runs_of_family):
    r = runs_of_family[0]
    n, k = r["num_experts"], r["top_k"]
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


def fig_ossification_curves(runs, out_dir, size, arch, dropout=0.0):
    fam = sorted([r for r in runs if family_key(r) == (size, arch, dropout)], key=lambda r: r["rep"])
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


def fig_ossification_vs_rep(runs, out_dir, families, fname, late=True, legend_inside=False):
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    colors = viridis(len(families))
    all_reps = set()
    for (famk, label), c, ls in zip(families, colors, LINESTYLES * 3):
        fam = sorted([r for r in runs if family_key(r) == famk], key=lambda r: r["rep"])
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
        leg = ax.legend(title="Model type", frameon=True, loc="lower right")
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
    for (famk, label), c, ls in zip(families, colors, LINESTYLES * 3):
        fam = sorted([r for r in runs if family_key(r) == famk], key=lambda r: r["rep"])
        if not fam:
            continue
        reps, vals = [], []
        for r in fam:
            inc = [v["loss_increase"] for kk in r["ko"]["knockout_losses"].values()
                   for v in kk.values()]
            vals.append(float(np.median(inc)) if stat == "median" else float(np.mean(inc)))
            reps.append(r["rep"])
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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = load_all(args.analysis_root, args.models_root)
    print(f"Loaded {len(runs)} analyzed runs")

    def fams(size, dropout, archs):
        out = []
        for a in archs:
            fam = [r for r in runs if family_key(r) == (size, a, dropout)]
            if fam:
                out.append(((size, a, dropout), arch_label(fam)))
        return out

    made = []
    # (a) ossification curves for one representative no-dropout family
    for arch in ("moe64s32", "moe128"):
        p = fig_ossification_curves(runs, args.out_dir, "80M", arch)
        if p:
            made.append(p)
    # (b) end-of-training stability vs R
    f80 = fams("80M", 0.0, ["moe16", "moe32", "moe64", "moe64s8", "moe64s32", "moe128"])
    made.append(fig_ossification_vs_rep(runs, args.out_dir, f80,
                "routing_ossification_vs_rep__80M__dolma_cc.pdf"))
    # knockout vs R, 80M no-dropout families
    made.append(fig_knockout_vs_rep(runs, args.out_dir, f80,
                "expert_knockout_vs_rep__80M__dolma_cc.pdf"))
    # 200M dropout contrast (ossification + knockout)
    f200 = (fams("200M", 0.0, ["moe32", "moe64"])
            + [(( "200M", "moe64", 0.1), "MoE (64 x 1/4), drop 0.1"),
               (("200M", "moe64", 0.4), "MoE (64 x 1/4), drop 0.4")])
    made.append(fig_ossification_vs_rep(runs, args.out_dir, f200,
                "routing_ossification_vs_rep__200M__dropout__dolma_cc.pdf", legend_inside=True))
    made.append(fig_knockout_vs_rep(runs, args.out_dir, f200,
                "expert_knockout_vs_rep__200M__dropout__dolma_cc.pdf"))
    # co-activation, 80M
    made.append(fig_coactivation_vs_rep(runs, args.out_dir, f80,
                "expert_coactivation_vs_rep__80M__dolma_cc.pdf"))
    for p in made:
        if p:
            print("wrote", p)
