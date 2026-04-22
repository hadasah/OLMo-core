from pathlib import Path
import time

import numpy as np

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data import replay_cache as replay_cache_module
from olmo_core.data import replay_cache_cli
from olmo_core.data.mixes import DataMix
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


def test_build_replay_cache_can_extend_existing_cache(tmp_path: Path) -> None:
    _write_tokens(tmp_path / "tokens_a.npy", list(range(0, 48)))
    _write_tokens(tmp_path / "tokens_b.npy", list(range(48, 96)))

    tokenizer = TokenizerConfig.gpt2()
    dataset_config = NumpyFSLDatasetConfig(
        paths=[str(tmp_path / "tokens_a.npy"), str(tmp_path / "tokens_b.npy")],
        tokenizer=tokenizer,
        sequence_length=4,
        work_dir=str(tmp_path / "dataset-work"),
    )

    initial_manifest = build_replay_cache(
        dataset_config,
        data_seed=34521,
        max_tokens=10,
        output_root=tmp_path / "replay-cache",
        epoch=1,
        shard_size_tokens=6,
        work_dir=tmp_path / "replay-work",
    )
    extended_manifest = build_replay_cache(
        dataset_config,
        data_seed=34521,
        max_tokens=18,
        output_root=tmp_path / "replay-cache",
        epoch=1,
        shard_size_tokens=6,
        work_dir=tmp_path / "replay-work",
    )

    expected_tokens = _flatten_source_tokens(
        dataset_config, data_seed=34521, epoch=1, max_tokens=18
    )
    replay_dataset_config = NumpyReplayFSLDatasetConfig(
        paths=extended_manifest.resolved_shard_paths(tmp_path / "replay-cache"),
        tokenizer=tokenizer,
        sequence_length=4,
    )
    replay_tokens = _flatten_replay_tokens(replay_dataset_config)
    usable_tokens = (len(expected_tokens) // replay_dataset_config.sequence_length) * replay_dataset_config.sequence_length

    assert initial_manifest.shard_token_counts == [6, 4]
    assert extended_manifest.max_tokens_requested == 18
    assert extended_manifest.total_tokens_written == 18
    assert extended_manifest.shard_token_counts == [6, 4, 6, 2]
    assert replay_tokens == expected_tokens[:usable_tokens]


def test_build_replay_cache_parallel_reads_preserve_seeded_token_order(
    tmp_path: Path, monkeypatch
) -> None:
    _write_tokens(tmp_path / "tokens_a.npy", list(range(0, 40)))
    _write_tokens(tmp_path / "tokens_b.npy", list(range(40, 88)))

    tokenizer = TokenizerConfig.gpt2()
    dataset_config = NumpyFSLDatasetConfig(
        paths=[str(tmp_path / "tokens_a.npy"), str(tmp_path / "tokens_b.npy")],
        tokenizer=tokenizer,
        sequence_length=4,
        work_dir=str(tmp_path / "dataset-work"),
    )

    monkeypatch.setattr(
        replay_cache_module,
        "_should_use_parallel_sparse_reads",
        lambda *args, **kwargs: True,
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
        read_workers=4,
        lookahead_instances=8,
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
    usable_tokens = (
        len(expected_tokens) // replay_dataset_config.sequence_length
    ) * replay_dataset_config.sequence_length

    assert manifest.complete
    assert manifest.total_tokens_written == max_tokens
    assert replay_tokens == expected_tokens[:usable_tokens]


def test_build_replay_cache_parallel_reads_handle_out_of_order_completion(
    tmp_path: Path, monkeypatch
) -> None:
    _write_tokens(tmp_path / "tokens_a.npy", list(range(0, 48)))
    _write_tokens(tmp_path / "tokens_b.npy", list(range(48, 96)))

    tokenizer = TokenizerConfig.gpt2()
    dataset_config = NumpyFSLDatasetConfig(
        paths=[str(tmp_path / "tokens_a.npy"), str(tmp_path / "tokens_b.npy")],
        tokenizer=tokenizer,
        sequence_length=4,
        work_dir=str(tmp_path / "dataset-work"),
    )

    original_fetch = replay_cache_module._fetch_replay_source_instance

    def delayed_fetch(path, local_instance_index, *, sequence_length, dtype):
        time.sleep((local_instance_index % 4) * 0.01)
        return original_fetch(
            path,
            local_instance_index,
            sequence_length=sequence_length,
            dtype=dtype,
        )

    monkeypatch.setattr(
        replay_cache_module,
        "_should_use_parallel_sparse_reads",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(replay_cache_module, "_fetch_replay_source_instance", delayed_fetch)

    max_tokens = 30
    manifest = build_replay_cache(
        dataset_config,
        data_seed=34521,
        max_tokens=max_tokens,
        output_root=tmp_path / "replay-cache",
        epoch=1,
        shard_size_tokens=8,
        work_dir=tmp_path / "replay-work",
        read_workers=4,
        lookahead_instances=8,
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
    usable_tokens = (
        len(expected_tokens) // replay_dataset_config.sequence_length
    ) * replay_dataset_config.sequence_length

    assert replay_tokens == expected_tokens[:usable_tokens]


def test_replay_cache_cli_build_passes_parallel_read_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Manifest:
        total_tokens_written = 123
        shard_paths = ["shards/part-00000.npy"]

    def fake_prepare_cli_environment() -> None:
        return None

    def fake_build_replay_cache(dataset_config, **kwargs):
        captured["dataset_config"] = dataset_config
        captured["kwargs"] = kwargs
        return _Manifest()

    monkeypatch.setattr(
        replay_cache_cli, "prepare_cli_environment", fake_prepare_cli_environment
    )
    monkeypatch.setattr(replay_cache_cli, "build_replay_cache", fake_build_replay_cache)

    replay_cache_cli.build(
        output_root="data/replay",
        data_root="https://olmo-data.org",
        work_dir=None,
        source_mix=DataMix.OLMoE_mix_0824,
        source_sequence_length=4096,
        data_seed=34521,
        epoch=1,
        max_tokens=100,
        shard_size_tokens=50,
        read_workers=7,
        lookahead_instances=123,
    )

    dataset_config = captured["dataset_config"]
    kwargs = captured["kwargs"]

    assert isinstance(dataset_config, NumpyFSLDatasetConfig)
    assert kwargs["read_workers"] == 7
    assert kwargs["lookahead_instances"] == 123
