#!/bin/bash
# Slurm array job: download + tokenize a range of mlfoundations/dclm-pool-400m-1x
# shards into the flat numpy format OLMo-core trains on.
#
# Designed to be run repeatedly to grow a corpus incrementally. Output is keyed by
# the global shard range each task handled, so separate submissions never collide
# and the union of everything under $OUT_ROOT is always a valid corpus.
#
#   mkdir -p $OLMOE_DIR/ml/data/slurm_logs
#
#   # batch 1: shards 0-31
#   FIRST_SHARD=0 LAST_SHARD=31 SHARDS_PER_TASK=8 \
#     sbatch --export=ALL,FIRST_SHARD,LAST_SHARD,SHARDS_PER_TASK \
#            --array=0-3 ml/scripts/data/tokenize_dclm_slurm.sh
#
#   # batch 2, later: shards 32-63. Same --array indices; the shard range comes
#   # from FIRST_SHARD, so output lands in new directories.
#   FIRST_SHARD=32 LAST_SHARD=63 SHARDS_PER_TASK=8 \
#     sbatch --export=ALL,FIRST_SHARD,LAST_SHARD,SHARDS_PER_TASK \
#            --array=0-3 ml/scripts/data/tokenize_dclm_slurm.sh
#
# Array size must be ceil((LAST_SHARD - FIRST_SHARD + 1) / SHARDS_PER_TASK) - 1.
# Sizing: each shard is ~230 MB compressed and yields roughly 82M tokens; the full
# pool is 2828 shards / 650 GB / ~233B tokens.
#
# Run ml/scripts/data/dclm_corpus_status.py at any point to see what has been
# built so far and how many tokens it adds up to.
#
# --partition below is a guess -- this is a CPU-only job, so it should NOT go to
# the GPU partition the training sweeps use (ckpt-g2). Check `sinfo -s` and
# override at submit time with `sbatch --partition=<cpu-partition>` if needed.
#
#SBATCH --job-name=tokenize-dclm
#SBATCH --account=zlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/slurm_logs/%x-%A_%a.out
#SBATCH --error=/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/slurm_logs/%x-%A_%a.err

# Sourced before `set -u`: interactive rc files routinely reference unbound vars.
source ~/.bashrc
source ~/init_scripts/.olmoe-core-ml.sh 2>/dev/null || true
conda activate dolma

set -euo pipefail

export OLMOE_DIR=/gscratch/zlab/margsli/gitfiles/OLMo-core
export DATA_DIR=$OLMOE_DIR/ml/data

HF_DATASET="mlfoundations/dclm-pool-400m-1x"
CORPUS_NAME="dclm-pool-400m-1x"

FIRST_SHARD=${FIRST_SHARD:-0}
LAST_SHARD=${LAST_SHARD:-31}
SHARDS_PER_TASK=${SHARDS_PER_TASK:-8}
# Free the compressed shards once tokenized. Set to 0 to keep them.
DELETE_RAW=${DELETE_RAW:-1}

# dolma2 tokenizer. Must match TokenizerConfig.dolma2() in
# src/olmo_core/data/tokenizer.py -- vocab_size=100278 so dtype MUST be uint32.
TOKENIZER='allenai/dolma2-tokenizer'
EOS_TOKEN_ID=100257
PAD_TOKEN_ID=100277
DTYPE=uint32

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
# Slurm sets SLURM_RESTART_COUNT only after a requeue. Writing each attempt to
# its own directory keeps a resumed run from colliding with part-* files the
# previous attempt already wrote: dolma derives part numbering from the position
# of a file within the *remaining* (not-yet-done) work, so those indices shift
# between attempts. Training globs across all of them, so extra dirs are free.
ATTEMPT=${SLURM_RESTART_COUNT:-0}

# --- shard range for this task --------------------------------------------
START=$((FIRST_SHARD + TASK_ID * SHARDS_PER_TASK))
END=$((START + SHARDS_PER_TASK - 1))
if [ "$END" -gt "$LAST_SHARD" ]; then END=$LAST_SHARD; fi
if [ "$START" -gt "$LAST_SHARD" ]; then
    echo "Task $TASK_ID starts at shard $START, past LAST_SHARD=$LAST_SHARD; nothing to do."
    exit 0
fi

# Directories are keyed by the GLOBAL shard range, not the array task id, so
# re-submitting with different FIRST_SHARD/LAST_SHARD can never overwrite or
# duplicate an earlier batch. Zero-padded so lexical order matches numeric order,
# which keeps the sorted path list stable as new batches are appended.
RANGE=$(printf 'shards-%08d-%08d' "$START" "$END")

OUT_ROOT=$DATA_DIR/preprocessed/$CORPUS_NAME/$TOKENIZER
RAW_DIR=$DATA_DIR/raw/$CORPUS_NAME/$RANGE
OUT_DIR=$OUT_ROOT/$RANGE/a$ATTEMPT
# Persistent work dir, also keyed by range. dolma writes a "<name>.done.txt"
# marker per finished source file here and skips those files on re-run
# (_get_all_paths). The default is a tempfile.TemporaryDirectory that
# make_workdirs deletes on exit, which would silently disable resume.
WORK_DIR=$DATA_DIR/work/$CORPUS_NAME/$RANGE
CONFIG_PATH=$WORK_DIR/tokens.yaml

mkdir -p "$RAW_DIR" "$OUT_DIR" "$WORK_DIR/input" "$WORK_DIR/output"

FILES=()
for i in $(seq "$START" "$END"); do
    FILES+=("$(printf 'CC_shard_%08d.jsonl.zst' "$i")")
done
echo "Task $TASK_ID (attempt $ATTEMPT): $RANGE (${#FILES[@]} files) -> $OUT_DIR"

# --- step 1: download ------------------------------------------------------
# Filenames are passed positionally; `hf download`'s --include is nargs="*", so
# repeating the flag would overwrite rather than accumulate. Already-complete
# files are skipped, which makes this safe to re-run after a preemption.
hf download "$HF_DATASET" "${FILES[@]}" \
    --repo-type dataset --local-dir "$RAW_DIR"
# (on huggingface_hub < 0.34 the command is `huggingface-cli download ...`)

# --- step 2: dolma config --------------------------------------------------
# Records here have exactly three top-level keys -- text, metadata, warcinfo --
# and no `id`. dolma's default id_field_name="id" would fail every line inside a
# per-line handler that only *logs*, yielding an empty corpus with a zero exit
# code. id_field_name: null makes dolma synthesize f"{sha256(path)}-{lineno}".
#
# This must go through YAML: dolma's argparse layer registers no type= converter,
# so `--fields.id_field_name null` arrives as the string "null". Only read_config
# runs yaml.safe_load, where null becomes a real None.
#
# metadata.WARC-Record-ID is not a usable alternative despite dolma supporting
# dotted names -- make_spec_from_fields builds the schema with msgspec.defstruct,
# which rejects hyphens ("TypeError: __slots__ must be identifiers").
cat > "$CONFIG_PATH" <<YAML
fields:
  text_field_name: text
  id_field_name: null
YAML

# --- step 3: tokenize ------------------------------------------------------
# --max_size is a TOKEN count, not bytes, despite dolma's help text: executor.py
# passes it straight through as MemmapWriter(max_tokens=...). dolma preallocates
# a memmap of exactly that many tokens per concurrent writer before shrinking it
# on close, so with N processes budget N * max_size * 4 bytes of transient disk.
# 250M tokens = 1 GB per writer at uint32.
PROCESSES=${#FILES[@]}
if [ "$PROCESSES" -gt "${SLURM_CPUS_PER_TASK:-8}" ]; then
    PROCESSES=${SLURM_CPUS_PER_TASK:-8}
fi

dolma -c "$CONFIG_PATH" tokens \
    --documents "$RAW_DIR"/'*.jsonl.zst' \
    --destination "$OUT_DIR" \
    --work_dir.input "$WORK_DIR/input" \
    --work_dir.output "$WORK_DIR/output" \
    --tokenizer.name_or_path "$TOKENIZER" \
    --tokenizer.eos_token_id $EOS_TOKEN_ID \
    --tokenizer.pad_token_id $PAD_TOKEN_ID \
    --dtype $DTYPE \
    --max_size '250_000_000' \
    --seed 0 \
    --processes "$PROCESSES"

echo "Tokenized shards in $OUT_DIR:"
ls -la "$OUT_DIR"

if [ "$DELETE_RAW" = "1" ]; then
    echo "Removing raw shards in $RAW_DIR"
    rm -rf "$RAW_DIR"
fi
