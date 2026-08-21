"""
Tokenize a HuggingFace dataset into the flat uint numpy format that OLMo-core's
``Numpy*Dataset`` classes expect, so it can be mixed with the pre-tokenized
olmo-data mixes.

The output files are *raw* memory-mapped token ID arrays (no ``.npy`` header),
which is what :func:`olmo_core.data.utils.memmap_to_write` produces and what
``np.memmap(path, mode="r", dtype=...)`` reads back. Do not use ``np.save``.

Example::

    python ml/scripts/tokenize_hf_dataset.py \\
        --dataset HuggingFaceFW/fineweb-edu \\
        --config sample-10BT \\
        --split train \\
        --tokenizer dolma2 \\
        --output-dir /path/to/my-data/fineweb-edu/allenai/dolma2-tokenizer

Then point a training run at ``<output-dir>/part-*.npy``.
"""

import argparse
import logging
from pathlib import Path
from typing import Iterator, List

import numpy as np

from olmo_core.data import TokenizerConfig
from olmo_core.data.utils import memmap_to_write, write_document_indices

log = logging.getLogger(__name__)

TOKENIZER_LOOKUP = {
    "dolma2": TokenizerConfig.dolma2,
    "gpt_neox_olmo_dolma_v1_5": TokenizerConfig.gpt_neox_olmo_dolma_v1_5,
}


def get_dtype(tokenizer_config: TokenizerConfig):
    """Same dtype inference as ``NumpyDatasetConfig.get_dtype()``."""
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if (tokenizer_config.vocab_size - 1) <= np.iinfo(dtype).max:
            return dtype
    raise ValueError("vocab size too big!")


def iter_token_batches(
    dataset, tokenizer, text_field: str, eos_token_id: int, bos_token_id, batch_size: int
) -> Iterator[List[int]]:
    """Yield lists of token IDs, one list per batch of documents, EOS-terminated."""
    batch: List[str] = []

    def flush(docs: List[str]) -> List[int]:
        encoded = tokenizer(docs, add_special_tokens=False)["input_ids"]
        out: List[int] = []
        for ids in encoded:
            if bos_token_id is not None:
                out.append(bos_token_id)
            out.extend(ids)
            out.append(eos_token_id)
        return out

    for example in dataset:
        batch.append(example[text_field])
        if len(batch) >= batch_size:
            yield flush(batch)
            batch = []
    if batch:
        yield flush(batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="HF dataset repo, e.g. 'allenai/c4'")
    parser.add_argument("--config", default=None, help="HF dataset config/subset name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text", help="Column holding the raw text")
    parser.add_argument(
        "--tokenizer",
        default="dolma2",
        help=f"One of {list(TOKENIZER_LOOKUP)}, or a raw HF tokenizer identifier",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tokens-per-shard",
        type=int,
        default=500_000_000,
        help="Token IDs per output file. Smaller shards parallelize better across data workers.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Stop after writing roughly this many tokens.",
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Documents per encode call")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream from the Hub instead of downloading the whole dataset first.",
    )
    parser.add_argument("--part-prefix", default="part")
    parser.add_argument(
        "--write-doc-indices",
        action="store_true",
        help=(
            "Also write the '.csv.gz' document-index sidecar next to each shard. Required if you "
            "will serve these shards over a remote URL, since document boundaries can't be scanned "
            "from a remote array."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    if args.tokenizer in TOKENIZER_LOOKUP:
        tokenizer_config = TOKENIZER_LOOKUP[args.tokenizer]()
    else:
        tokenizer_config = TokenizerConfig.from_hf(args.tokenizer)

    assert tokenizer_config.identifier is not None
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.identifier)
    dtype = get_dtype(tokenizer_config)
    log.info(
        f"Tokenizer '{tokenizer_config.identifier}': vocab_size={tokenizer_config.vocab_size}, "
        f"eos={tokenizer_config.eos_token_id}, bos={tokenizer_config.bos_token_id}, dtype={dtype.__name__}"
    )

    dataset = load_dataset(
        args.dataset, args.config, split=args.split, streaming=args.streaming
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer: List[np.ndarray] = []
    buffered = 0
    shard_idx = 0
    total_written = 0

    def write_shard(tokens: np.ndarray) -> None:
        nonlocal shard_idx, total_written
        path = output_dir / f"{args.part_prefix}-{shard_idx:05d}.npy"
        with memmap_to_write(path, shape=(len(tokens),), dtype=dtype) as mmap:
            mmap[:] = tokens
        log.info(f"Wrote {len(tokens):,} tokens to {path}")
        if args.write_doc_indices:
            metadata_path = write_document_indices(
                path, dtype=dtype, eos_token_id=tokenizer_config.eos_token_id
            )
            log.info(f"Wrote document indices to {metadata_path}")
        shard_idx += 1
        total_written += len(tokens)

    for token_ids in iter_token_batches(
        dataset,
        tokenizer,
        args.text_field,
        tokenizer_config.eos_token_id,
        tokenizer_config.bos_token_id,
        args.batch_size,
    ):
        arr = np.array(token_ids, dtype=dtype)
        buffer.append(arr)
        buffered += len(arr)

        while buffered >= args.tokens_per_shard:
            concatenated = np.concatenate(buffer)
            write_shard(concatenated[: args.tokens_per_shard])
            remainder = concatenated[args.tokens_per_shard :]
            buffer = [remainder] if len(remainder) else []
            buffered = len(remainder)

        if args.max_tokens is not None and total_written + buffered >= args.max_tokens:
            break

    if buffered:
        write_shard(np.concatenate(buffer))

    log.info(f"Done. Wrote {total_written:,} tokens across {shard_idx} files in {output_dir}")


if __name__ == "__main__":
    main()
