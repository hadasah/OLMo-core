#!/usr/bin/env python3
"""
Dry-run validator for the Family B (two-domain specialist-injection) data mixtures.

For each (cell, repetition) it prints the realized unique-pool sizes and the effective
repetition, and asserts the invariant the experiment relies on: as the minority repetition R
grows at a fixed mixing ratio, the minority *unique* pool shrinks and is strictly nested
inside the lower-R pool. (Byte-level nesting + full distinct coverage are guaranteed at the
data layer by the deterministic "prefix" sampler over a seed-independent permutation; this
script verifies the size/feasibility preconditions.)

Optionally (--check-paths) it fetches the real per-source token counts to confirm each cell is
feasible -- the lowest-R unique pool must fit inside the source. That requires olmo_core and
network access to the data root.

This script does NOT launch anything and has no side effects.

Keep CELLS/REPS in sync with build_family_b_subgrids() in train_sweep.py.
"""
import argparse

# (cell label, primary datamix, minority datamix, minority fraction)
CELLS = [
    ("sc50", "dclm_only", "starcoder_only", 0.5),
    ("sc90", "dclm_only", "starcoder_only", 0.1),
    ("p2o50", "dclm_only", "pes2o_only", 0.5),
]
REPS = [1, 2, 4, 8, 16, 32]


def _fetch_populations(data_root, tokenizer_name):
    """Return {datamix_name: total_token_count}; empty dict if olmo_core/network unavailable."""
    pops = {}
    try:
        import numpy as np
        from olmo_core.data import DataMix, TokenizerConfig
        from olmo_core.io import get_file_size
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[--check-paths] could not import olmo_core ({e}); skipping real token counts.")
        return pops

    tok = {"dolma2": TokenizerConfig.dolma2}[tokenizer_name]()
    # dolma2 vocab > 65535 => uint32 tokens (4 bytes each).
    itemsize = np.uint32(0).itemsize
    lookup = {
        "dclm_only": DataMix.dclm_only,
        "starcoder_only": DataMix.starcoder_only,
        "pes2o_only": DataMix.pes2o_only,
        "wikipedia_only": DataMix.wikipedia_only,
        "dolma17": DataMix.dolma17,
    }
    needed = set()
    for _, primary, minority, _ in CELLS:
        needed.add(primary)
        needed.add(minority)
    for name in sorted(needed):
        try:
            paths, _ = lookup[name].build(data_root, tok.identifier)
            pops[name] = sum(get_file_size(p) // itemsize for p in paths)
            print(f"[--check-paths] {name}: {len(paths)} shards, {pops[name]:,} tokens")
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[--check-paths] failed to size '{name}': {e}")
    return pops


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=1_600_000_000,
                    help="Total training token budget T (default 1.6B, the olmo2_ml_80M budget).")
    ap.add_argument("--seq-length", type=int, default=2048,
                    help="Sequence length (used to flag shards that would yield <1 instance).")
    ap.add_argument("--check-paths", action="store_true",
                    help="Fetch real source token counts (needs olmo_core + network) and check feasibility.")
    ap.add_argument("--data-root", type=str, default="https://olmo-data.org/")
    ap.add_argument("--tokenizer", type=str, default="dolma2")
    args = ap.parse_args()
    T = args.budget

    populations = _fetch_populations(args.data_root, args.tokenizer) if args.check_paths else {}

    ok = True
    for cell, primary, minority, f in CELLS:
        print(f"\n=== cell {cell}: primary={primary} ({1 - f:.0%}) + minority={minority} ({f:.0%}); T={T:,} ===")
        print(f"{'rep':>5} {'minority_unique':>16} {'primary_unique':>15} {'realized_rep':>13}")
        prev_unique = None
        for r in REPS:
            min_unique = int(f * T / r)
            pri_unique = int((1 - f) * T)  # primary is always 1x in Family B
            realized = (f * T) / min_unique if min_unique else float("inf")
            print(f"{r:>5} {min_unique:>16,} {pri_unique:>15,} {realized:>13.2f}")
            if min_unique <= 0:
                print(f"   !! EMPTY minority unique pool at rep {r}")
                ok = False
            if prev_unique is not None and min_unique > prev_unique:
                print(f"   !! NESTING PRECONDITION VIOLATED: pool grew {prev_unique:,} -> {min_unique:,}")
                ok = False
            prev_unique = min_unique

        if args.check_paths and minority in populations:
            P = populations[minority]
            largest_needed = int(f * T / REPS[0])  # lowest rep needs the largest unique pool
            status = "OK" if largest_needed <= P else "INFEASIBLE (lower fraction or drop the lowest rep)"
            if largest_needed > P:
                ok = False
            print(f"   feasibility: minority population={P:,}; largest pool (rep {REPS[0]}x)={largest_needed:,} -> {status}")
            # Flag shards too small to yield an instance at the highest repetition.
            n_shards_paths = None
            try:
                import numpy as np
                from olmo_core.data import DataMix, TokenizerConfig
                tok = {"dolma2": TokenizerConfig.dolma2}[args.tokenizer]()
                lookup = {"starcoder_only": DataMix.starcoder_only, "pes2o_only": DataMix.pes2o_only}
                if minority in lookup:
                    paths, _ = lookup[minority].build(args.data_root, tok.identifier)
                    n_shards_paths = len(paths)
            except Exception:
                pass
            if n_shards_paths:
                smallest_pool = int(f * T / REPS[-1])
                per_shard = smallest_pool / n_shards_paths
                if per_shard < args.seq_length:
                    print(f"   !! at rep {REPS[-1]}x, ~{per_shard:,.0f} tokens/shard < seq_length "
                          f"({args.seq_length}); some shards may yield 0 instances")
                    ok = False

    print("\nAll checks passed." if ok else "\nThere were problems (see '!!' lines above).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
