from __future__ import annotations

import logging
from typing import Optional

import typer

from olmo_core.data.mixes import DataMix
from olmo_core.utils import prepare_cli_environment

from .numpy_dataset import NumpyFSLDatasetConfig
from .replay_cache import build_replay_cache
from .tokenizer import TokenizerConfig

app = typer.Typer(help="Build a local replay cache from a seeded training data stream.")

log = logging.getLogger(__name__)


@app.command()
def build(
    output_root: str = typer.Option(..., "--output-root", help="Local output directory for replay data."),
    data_root: str = typer.Option(
        "https://olmo-data.org",
        "--data-root",
        help="Root directory or URL for the source training data.",
    ),
    work_dir: Optional[str] = typer.Option(
        None,
        "--work-dir",
        help="Local dataset work directory. Defaults to '<output-root>/work'.",
    ),
    source_mix: DataMix = typer.Option(
        DataMix.OLMoE_mix_0824,
        "--source-mix",
        help="Source data mix to replay.",
    ),
    source_sequence_length: int = typer.Option(
        4096,
        "--source-sequence-length",
        help="Sequence length used to define the source training stream.",
    ),
    data_seed: int = typer.Option(
        34521,
        "--data-seed",
        help="Data loader seed for the source stream.",
    ),
    epoch: int = typer.Option(1, "--epoch", help="Epoch number to replay."),
    max_tokens: int = typer.Option(
        10_000_000_000,
        "--max-tokens",
        help="Maximum number of tokens to materialize locally.",
    ),
    shard_size_tokens: int = typer.Option(
        500_000_000,
        "--shard-size-tokens",
        help="Target number of tokens per replay shard.",
    ),
) -> None:
    prepare_cli_environment()

    tokenizer = TokenizerConfig.dolma2()
    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        source_mix,
        tokenizer=tokenizer,
        mix_base_dir=data_root,
        sequence_length=source_sequence_length,
        max_target_sequence_length=max(8192, source_sequence_length),
        work_dir=work_dir or f"{output_root.rstrip('/')}/work",
    )

    manifest = build_replay_cache(
        dataset_config,
        data_seed=data_seed,
        max_tokens=max_tokens,
        output_root=output_root,
        epoch=epoch,
        shard_size_tokens=shard_size_tokens,
        work_dir=work_dir or f"{output_root.rstrip('/')}/work",
    )
    log.info(
        "Replay cache ready at '%s' with %s tokens across %d shards",
        output_root,
        f"{manifest.total_tokens_written:,d}",
        len(manifest.shard_paths),
    )


if __name__ == "__main__":
    app()
