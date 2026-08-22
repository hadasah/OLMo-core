#!/usr/bin/env bash
# Create a dedicated conda env for tokenization.
#
# Why a separate env from olmoe-core-ml: dolma carries a set of hard, backdated
# pins that would drag the training env backwards.
#
#   dolma (every published version, 1.1.0 through 1.2.1)
#     -> tokenizers >=0.15.0,<=0.19.1
#     -> numpy <2
#     -> s3fs ==2023.6.0   (which itself pins fsspec ==2023.6.0, aiobotocore ~=2.5.0)
#     -> requires-python  >=3.10,<3.13
#
# The huggingface_hub conflict follows from the tokenizers cap:
#
#   tokenizers <=0.19.1  ->  huggingface-hub >=0.16.4,<1.0
#
# so huggingface_hub 1.x cannot coexist with dolma. The `hf` CLI entry point was
# added in huggingface_hub 0.34.0 (verified against the wheel metadata: 0.33.5
# ships only `huggingface-cli`, 0.34.0 ships `hf = huggingface_hub.cli.hf:main`),
# so the usable window is:
#
#   huggingface_hub >=0.34.0,<1.0
#
# Separately, tokenizers<=0.19.1 is incompatible with any recent transformers
# (4.57 requires tokenizers>=0.22), which is the main reason not to install dolma
# into the training env.
set -euo pipefail

ENV_NAME=${ENV_NAME:-dolma}
# 3.12 is the newest dolma supports. Prebuilt linux x86_64 wheels exist at this
# version for both dolma (cp312 manylinux_2_28) and fasttext-wheel==0.9.2
# (cp312 manylinux_2_17), so nothing needs a compiler. Drop to 3.11 if the
# fasttext-wheel build is attempted from source for any reason -- it does not
# compile cleanly on modern toolchains.
PYTHON_VERSION=${PYTHON_VERSION:-3.12}

conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# The smart_open[zst] extra is NOT optional here, despite dolma already depending
# on `zstandard`. smart_open 7.4.0 changed _handle_zstd's backend from the
# `zstandard` package to `backports.zstd` (on Python < 3.14), and that backport is
# only pulled in by smart_open's own `zst` extra. dolma requests plain
# `smart-open>=7.0.4` with no extras, so a fresh install resolves to 8.x and then
# raises ModuleNotFoundError: backports.zstd the first time it opens a .jsonl.zst.
# Pinning 'smart_open>=7.0.4,<7.4.0' also works; the extra is forward-compatible.
pip install 'dolma==1.2.1' 'huggingface_hub>=0.34.0,<1.0' 'smart_open[zst]'

echo
echo "--- verifying ---"
dolma tokens --help > /dev/null && echo "  dolma tokens: ok"
hf download --help > /dev/null && echo "  hf download: ok"
python - <<'PY'
import json, tempfile, os
import huggingface_hub, tokenizers, numpy, smart_open
print(f"  huggingface_hub {huggingface_hub.__version__}")
print(f"  tokenizers      {tokenizers.__version__}")
print(f"  numpy           {numpy.__version__}")
print(f"  smart_open      {smart_open.__version__}")

# Exercise the actual .zst read path the tokenize scripts depend on, rather than
# introspecting the compressor registry (whose accessor name has moved around).
# dolma registers the zstd handlers in dolma.core.utils at import time.
import dolma.core.utils  # noqa: F401
import zstandard

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "probe.jsonl.zst")
    with open(path, "wb") as f:
        f.write(zstandard.ZstdCompressor().compress(b'{"text": "hello"}\n'))
    with smart_open.open(path, mode="rt") as f:
        assert json.loads(f.readline())["text"] == "hello"
print("  .zst read via smart_open: ok")
PY

echo
echo "Done. Use with:  conda activate $ENV_NAME"