"""
Report what has been tokenized so far by tokenize_dclm_slurm.sh, and emit the
path list to feed into training.

The corpus is just a set of files, so it grows monotonically: run the slurm job
on more shard ranges whenever you want, then re-run this to see the new total.

    python ml/scripts/data/dclm_corpus_status.py
    python ml/scripts/data/dclm_corpus_status.py --paths-only > dclm_paths.txt
"""

import argparse
import os
from collections import defaultdict
from glob import glob

import numpy as np

from olmo_core.data import TokenizerConfig

DEFAULT_DATA_DIR = "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data"
CORPUS_NAME = "dclm-pool-400m-1x"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--corpus-name", default=CORPUS_NAME)
    parser.add_argument("--tokenizer", default="dolma2")
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="Print just the sorted .npy paths, one per line.",
    )
    args = parser.parse_args()

    tokenizer_config = getattr(TokenizerConfig, args.tokenizer)()
    # Same dtype inference as NumpyDatasetConfig.get_dtype().
    dtype = next(
        dt
        for dt in (np.uint8, np.uint16, np.uint32, np.uint64)
        if (tokenizer_config.vocab_size - 1) <= np.iinfo(dt).max
    )
    itemsize = np.dtype(dtype).itemsize

    root = os.path.join(
        args.data_dir, "preprocessed", args.corpus_name, str(tokenizer_config.identifier)
    )
    # Sorted so the path list is stable. Directories are named
    # shards-<start>-<end>/a<attempt> with zero padding, so lexical order matches
    # numeric order and appending a later batch never reorders earlier paths.
    paths = sorted(glob(os.path.join(root, "**", "*.npy"), recursive=True))

    if args.paths_only:
        for p in paths:
            print(p)
        return

    if not paths:
        print(f"No tokenized shards found under {root}")
        return

    by_range = defaultdict(list)
    for p in paths:
        # .../<tokenizer>/shards-XXXXXXXX-YYYYYYYY/a<N>/part-*.npy
        rel = os.path.relpath(p, root)
        by_range[rel.split(os.sep)[0]].append(p)

    print(f"Corpus root: {root}")
    print(f"Tokenizer:   {tokenizer_config.identifier} (dtype {np.dtype(dtype).name})\n")

    total_tokens = 0
    total_files = 0
    for range_name in sorted(by_range):
        group = by_range[range_name]
        tokens = sum(os.path.getsize(p) // itemsize for p in group)
        total_tokens += tokens
        total_files += len(group)
        print(f"  {range_name:<28} {len(group):>4} files  {tokens/1e9:>8.3f}B tokens")

    print(f"\n  {'TOTAL':<28} {total_files:>4} files  {total_tokens/1e9:>8.3f}B tokens")

    # Warn about gaps: a missing shard range is easy to miss when building up
    # in batches, and nothing downstream will complain about it.
    seen = set()
    for range_name in by_range:
        parts = range_name.split("-")
        if len(parts) == 3 and parts[0] == "shards":
            seen.update(range(int(parts[1]), int(parts[2]) + 1))
    if seen:
        gaps = sorted(set(range(min(seen), max(seen) + 1)) - seen)
        if gaps:
            print(f"\n  WARNING: {len(gaps)} shard(s) missing in range: {gaps[:20]}")


if __name__ == "__main__":
    main()
