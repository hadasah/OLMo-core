#!/usr/bin/env bash
# Tokenize mlfoundations/dclm-pool-400m-1x into the flat numpy format OLMo-core
# trains on, using the Dolma toolkit (parallel).
#
# Dolma reads JSON lines directly, including .jsonl.zst -- zstd handlers are
# registered in dolma/core/utils.py. No round-trip through `datasets` needed.
#
# Requires the dedicated tokenization env -- see setup_dolma_env.sh. dolma pins
# tokenizers<=0.19.1 (-> huggingface-hub<1.0), numpy<2 and s3fs==2023.6.0, which
# conflict with olmoe-core-ml.
set -euo pipefail

conda activate "${DOLMA_ENV:-dolma-tok}"

export OLMOE_DIR=/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core
export DATA_DIR=$OLMOE_DIR/ml/data

# ---- edit these ----------------------------------------------------------
HF_DATASET="mlfoundations/dclm-pool-400m-1x"
CORPUS_NAME="dclm-pool-400m-1x"
# The full pool is 2828 shards / 650 GB compressed / ~233B tokens. Each shard is
# ~230 MB compressed and yields roughly 82M tokens, so pick the shard count from
# your token budget rather than downloading everything.
NUM_SHARDS=32                # ~7 GB download, ~2.6B tokens
PROCESSES=20
# --------------------------------------------------------------------------

# dolma2 tokenizer. Must match TokenizerConfig.dolma2() in
# src/olmo_core/data/tokenizer.py -- vocab_size=100278 so dtype MUST be uint32.
TOKENIZER='allenai/dolma2-tokenizer'
EOS_TOKEN_ID=100257
PAD_TOKEN_ID=100277
DTYPE=uint32

RAW_DIR=$DATA_DIR/raw/$CORPUS_NAME
OUT_DIR=$DATA_DIR/preprocessed/$CORPUS_NAME/$TOKENIZER
CONFIG_PATH=$DATA_DIR/raw/$CORPUS_NAME.tokens.yaml

mkdir -p "$RAW_DIR" "$OUT_DIR"

# --- Step 1: download N raw shards ----------------------------------------
# Files are flat at the repo root, named CC_shard_00000000.jsonl.zst ...
# CC_shard_00002827.jsonl.zst -- no subdirectories, so a flat glob is correct.
INCLUDES=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    INCLUDES+=(--include "$(printf 'CC_shard_%08d.jsonl.zst' "$i")")
done

hf download "$HF_DATASET" --repo-type dataset \
    "${INCLUDES[@]}" --local-dir "$RAW_DIR"
# (on huggingface_hub < 0.34 the command is `huggingface-cli download ...`)

# --- Step 2: write the dolma config ---------------------------------------
# Records in this dataset have exactly three top-level keys: `text`, `metadata`,
# and `warcinfo`. There is NO `id` field, and dolma's default id_field_name="id"
# would make every line fail its per-line handler -- which is only *logged*, not
# raised, so you would get an empty corpus rather than an error.
#
# The fix is id_field_name: null, which makes dolma synthesize an id per document
# as f"{sha256(path)}-{line_number}". This must go through a YAML config, not a
# CLI flag: dolma's argparse layer registers no `type=` converter, so
# `--fields.id_field_name null` arrives as the literal string "null". Only the
# YAML path runs through yaml.safe_load, where `null` becomes a real None.
#
# The nested metadata.WARC-Record-ID is NOT a usable alternative even though
# dolma supports dotted field names: make_spec_from_fields builds the schema with
# msgspec.defstruct, which rejects hyphenated names
# ("TypeError: __slots__ must be identifiers").
cat > "$CONFIG_PATH" <<YAML
fields:
  text_field_name: text
  id_field_name: null
YAML

# --- Step 3: tokenize ------------------------------------------------------
# NOTE: --max_size is a TOKEN count, not bytes, despite dolma's help text.
# dolma preallocates a memmap of exactly this many tokens per shard before
# shrinking it on close, so an oversized value costs real disk during the run.
# 500M tokens = 2 GB at uint32.
#
# Add --dryrun to print the resolved config and exit -- do this once first and
# confirm `id_field_name: null` survived the merge before committing CPU hours.
# -c/--config is a GLOBAL option, registered on dolma's top-level parser, so it
# must precede the subcommand: `dolma -c cfg.yaml tokens ...`, not
# `dolma tokens -c cfg.yaml`. The latter fails with "unrecognized arguments: -c".
dolma -c "$CONFIG_PATH" tokens \
    --documents "$RAW_DIR"/'*.jsonl.zst' \
    --destination "$OUT_DIR" \
    --tokenizer.name_or_path "$TOKENIZER" \
    --tokenizer.eos_token_id $EOS_TOKEN_ID \
    --tokenizer.pad_token_id $PAD_TOKEN_ID \
    --dtype $DTYPE \
    --max_size '500_000_000' \
    --seed 0 \
    --processes $PROCESSES

echo "Tokenized shards in $OUT_DIR:"
ls -la "$OUT_DIR"
