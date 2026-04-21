from pathlib import Path

import numpy as np

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.replay_cache import (
    NumpyReplayFSLDatasetConfig,
    apply_replay_cache,
    build_replay_cache,
)
from olmo_core.data.utils import write_array_to_disk


def _write_tokens(path: Path, values: list[int]) -> None:
    write_array_to_disk(np.asarray(values, dtype=np.uint16), path)


def _flatten_source_tokens(
    dataset_config: NumpyFSLDatasetConfig, *, data_seed: int, epoch: int, max_tokens: int
) -> list[int]:
    dataset = dataset_config.build()
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=dataset_config.sequence_length,
        seed=data_seed,
        shuffle=True,
        num_workers=0,
        work_dir=str(dataset_config.work_dir),
    ).build(dataset)
    data_loader.reshuffle(epoch=epoch, in_memory=True)

    ordered_tokens: list[int] = []
    for source_index in data_loader.get_global_indices():
        ordered_tokens.extend(dataset[int(source_index)]["input_ids"].tolist())
        if len(ordered_tokens) >= max_tokens:
            return ordered_tokens[:max_tokens]

    raise AssertionError("source dataset did not contain enough tokens")


def _flatten_replay_tokens(dataset_config: NumpyFSLDatasetConfig | NumpyReplayFSLDatasetConfig) -> list[int]:
    dataset = dataset_config.build()
    tokens: list[int] = []
    for idx in range(len(dataset)):
        tokens.extend(dataset[idx]["input_ids"].tolist())
    return tokens


def test_build_replay_cache_preserves_seeded_token_order(tmp_path: Path) -> None:
    _write_tokens(tmp_path / "tokens_a.npy", list(range(0, 40)))
    _write_tokens(tmp_path / "tokens_b.npy", list(range(40, 88)))

    tokenizer = TokenizerConfig.gpt2()
    dataset_config = NumpyFSLDatasetConfig(
        paths=[str(tmp_path / "tokens_a.npy"), str(tmp_path / "tokens_b.npy")],
        tokenizer=tokenizer,
        sequence_length=4,
        work_dir=str(tmp_path / "dataset-work"),
    )

    max_tokens = 29
    manifest = build_replay_cache(
        dataset_config,
        data_seed=34521,
        max_tokens=max_tokens,
        output_root=tmp_path / "replay-cache",
        epoch=1,
        shard_size_tokens=7,
        work_dir=tmp_path / "replay-work",
    )

    expected_tokens = _flatten_source_tokens(
        dataset_config, data_seed=34521, epoch=1, max_tokens=max_tokens
    )
    replay_dataset_config = NumpyReplayFSLDatasetConfig(
        paths=manifest.resolved_shard_paths(tmp_path / "replay-cache"),
        tokenizer=tokenizer,
        sequence_length=4,
    )
    replay_tokens = _flatten_replay_tokens(replay_dataset_config)
    usable_tokens = (len(expected_tokens) // replay_dataset_config.sequence_length) * replay_dataset_config.sequence_length

    assert manifest.complete
    assert manifest.total_tokens_written == max_tokens
    assert replay_tokens == expected_tokens[:usable_tokens]
    assert not list((tmp_path / "replay-work").glob("global_indices*.npy"))


def test_apply_replay_cache_switches_training_to_local_stream(tmp_path: Path) -> None:
    _write_tokens(tmp_path / "tokens_a.npy", list(range(0, 48)))
    _write_tokens(tmp_path / "tokens_b.npy", list(range(48, 96)))

    tokenizer = TokenizerConfig.gpt2()
    source_dataset_config = NumpyFSLDatasetConfig(
        paths=[str(tmp_path / "tokens_a.npy"), str(tmp_path / "tokens_b.npy")],
        tokenizer=tokenizer,
        sequence_length=4,
        work_dir=str(tmp_path / "dataset-work"),
    )

    max_tokens = 30
    build_replay_cache(
        source_dataset_config,
        data_seed=34521,
        max_tokens=max_tokens,
        output_root=tmp_path / "replay-cache",
        epoch=1,
        shard_size_tokens=8,
        work_dir=tmp_path / "replay-work",
    )
    expected_tokens = _flatten_source_tokens(
        source_dataset_config, data_seed=34521, epoch=1, max_tokens=max_tokens
    )

    training_dataset_config = NumpyFSLDatasetConfig(
        paths=[str(tmp_path / "tokens_a.npy"), str(tmp_path / "tokens_b.npy")],
        tokenizer=tokenizer,
        sequence_length=5,
        work_dir=str(tmp_path / "train-work"),
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=10,
        seed=34521,
        work_dir=str(tmp_path / "train-work"),
    )

    loaded_manifest, replay_dataset_config = apply_replay_cache(
        training_dataset_config, data_loader_config, tmp_path / "replay-cache"
    )
    replay_dataset = replay_dataset_config.build()

    assert not data_loader_config.shuffle
    assert loaded_manifest.replay_dataset_fingerprint == replay_dataset.fingerprint

    replay_tokens = _flatten_replay_tokens(replay_dataset_config)
    usable_tokens = (
        len(expected_tokens) // training_dataset_config.sequence_length
    ) * training_dataset_config.sequence_length
    assert replay_tokens == expected_tokens[:usable_tokens]
