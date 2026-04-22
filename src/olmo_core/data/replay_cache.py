from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import is_url, normalize_path

from .data_loader import NumpyDataLoaderConfig
from .numpy_dataset import (
    NumpyDatasetBase,
    NumpyDatasetConfig,
    NumpyFSLDataset,
    NumpyFSLDatasetBase,
    NumpyFSLDatasetConfig,
)
from .utils import get_document_lengths, load_array_slice, load_array_slice_into_tensor

log = logging.getLogger(__name__)

REPLAY_CACHE_MANIFEST = "replay_manifest.json"
REPLAY_CACHE_FORMAT_VERSION = 1


def _mix_value(mix: object) -> Optional[str]:
    if mix is None:
        return None
    return getattr(mix, "value", mix)


class ReplayCacheManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_version: int = REPLAY_CACHE_FORMAT_VERSION
    complete: bool = False
    source_mix: Optional[str] = None
    source_data_root: Optional[str] = None
    tokenizer_identifier: Optional[str] = None
    source_sequence_length: int
    source_data_seed: int
    source_epoch: int
    source_dataset_fingerprint: str
    replay_dtype: str
    replay_dataset_fingerprint: Optional[str] = None
    max_tokens_requested: int
    total_tokens_written: int = 0
    shard_size_tokens: int
    shard_paths: list[str] = []
    shard_token_counts: list[int] = []

    def resolved_shard_paths(self, replay_root: PathOrStr) -> list[str]:
        root = Path(normalize_path(replay_root))
        return [str(root / shard_path) for shard_path in self.shard_paths]


def get_replay_manifest_path(replay_root: PathOrStr) -> Path:
    return Path(normalize_path(replay_root)) / REPLAY_CACHE_MANIFEST


def load_replay_manifest(replay_root: PathOrStr) -> ReplayCacheManifest:
    manifest_path = get_replay_manifest_path(replay_root)
    with manifest_path.open("r", encoding="utf-8") as f:
        return ReplayCacheManifest.model_validate_json(f.read())


def save_replay_manifest(replay_root: PathOrStr, manifest: ReplayCacheManifest) -> None:
    manifest_path = get_replay_manifest_path(replay_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def apply_replay_cache(
    dataset_config: NumpyFSLDatasetConfig,
    data_loader_config: NumpyDataLoaderConfig,
    replay_root: PathOrStr,
) -> tuple[ReplayCacheManifest, "NumpyReplayFSLDatasetConfig"]:
    manifest = load_replay_manifest(replay_root)
    if not manifest.complete:
        raise RuntimeError(
            f"Replay cache '{normalize_path(replay_root)}' is incomplete and cannot be used for training"
        )

    tokenizer_identifier = dataset_config.tokenizer.identifier
    if tokenizer_identifier != manifest.tokenizer_identifier:
        raise RuntimeError(
            "Replay cache tokenizer does not match training config: "
            f"{manifest.tokenizer_identifier!r} != {tokenizer_identifier!r}"
        )

    if data_loader_config.seed != manifest.source_data_seed:
        raise RuntimeError(
            "Replay cache data seed does not match training config: "
            f"{manifest.source_data_seed} != {data_loader_config.seed}"
        )

    source_mix = _mix_value(dataset_config.mix)
    if source_mix != manifest.source_mix:
        raise RuntimeError(
            "Replay cache source mix does not match training config: "
            f"{manifest.source_mix!r} != {source_mix!r}"
        )

    source_dataset = dataset_config.build()
    if source_dataset.fingerprint != manifest.source_dataset_fingerprint:
        raise RuntimeError(
            "Replay cache source dataset fingerprint does not match training config: "
            f"{manifest.source_dataset_fingerprint!r} != {source_dataset.fingerprint!r}"
        )

    data_loader_config.shuffle = False
    replay_dataset_config = NumpyReplayFSLDatasetConfig(
        paths=manifest.resolved_shard_paths(replay_root),
        tokenizer=dataset_config.tokenizer,
        sequence_length=dataset_config.sequence_length,
        generate_doc_lengths=dataset_config.generate_doc_lengths,
        include_instance_metadata=False,
        instance_filter_config=dataset_config.instance_filter_config,
        work_dir=dataset_config.work_dir,
    )
    return manifest, replay_dataset_config


def build_replay_cache(
    dataset_config: NumpyFSLDatasetConfig,
    *,
    data_seed: int,
    max_tokens: int,
    output_root: PathOrStr,
    epoch: int = 1,
    shard_size_tokens: int = 500_000_000,
    work_dir: Optional[PathOrStr] = None,
    read_workers: int = 32,
    lookahead_instances: int = 10_000,
) -> ReplayCacheManifest:
    if max_tokens <= 0:
        raise ValueError("'max_tokens' must be positive")
    if epoch <= 0:
        raise ValueError("'epoch' must be positive")
    if shard_size_tokens <= 0:
        raise ValueError("'shard_size_tokens' must be positive")
    if read_workers <= 0:
        raise ValueError("'read_workers' must be positive")
    if lookahead_instances <= 0:
        raise ValueError("'lookahead_instances' must be positive")
    if dataset_config.tokenizer.identifier is None:
        raise ValueError("Replay cache generation requires a tokenizer identifier")
    if dataset_config.sequence_length <= 0:
        raise ValueError("'sequence_length' must be positive")

    output_dir = Path(normalize_path(output_root))
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = dataset_config.build()
    npdtype = dataset_config.get_dtype()
    item_size = npdtype(0).itemsize

    data_loader = NumpyDataLoaderConfig(
        global_batch_size=dataset_config.sequence_length,
        work_dir=work_dir or dataset_config.work_dir or output_dir / "work",
        seed=data_seed,
        shuffle=True,
        num_workers=0,
    ).build(source_dataset)
    data_loader.reshuffle(epoch=epoch, in_memory=False)

    instances_needed = math.ceil(max_tokens / dataset_config.sequence_length)
    try:
        # We only need the replay prefix, so copy just that slice into memory and remove
        # the large on-disk permutation file immediately afterwards.
        global_indices = np.array(
            data_loader.get_global_indices()[:instances_needed], dtype=np.uint32, copy=True
        )
    finally:
        _cleanup_global_indices_file(data_loader)

    available_tokens = len(source_dataset) * dataset_config.sequence_length
    if max_tokens > available_tokens:
        raise RuntimeError(
            f"Requested {max_tokens:,d} tokens, but dataset only has {available_tokens:,d} tokens in epoch {epoch}"
        )

    manifest = _load_or_init_manifest(
        output_dir,
        dataset_config=dataset_config,
        data_seed=data_seed,
        epoch=epoch,
        max_tokens=max_tokens,
        shard_size_tokens=shard_size_tokens,
        source_dataset_fingerprint=source_dataset.fingerprint,
        replay_dtype=npdtype.__name__,
    )

    if manifest.complete and manifest.total_tokens_written >= max_tokens:
        log.info(
            "Replay cache already complete at '%s' with %s tokens",
            output_dir,
            f"{manifest.total_tokens_written:,d}",
        )
        return manifest

    _validate_existing_shards(output_dir, manifest, item_size=item_size)

    token_offset = manifest.total_tokens_written
    instance_index = token_offset // dataset_config.sequence_length
    intra_instance_offset = token_offset % dataset_config.sequence_length

    if _should_use_parallel_sparse_reads(
        source_dataset, read_workers=read_workers, lookahead_instances=lookahead_instances
    ):
        log.info(
            "Using parallel sparse replay reads with %d workers and %d-instance lookahead",
            read_workers,
            lookahead_instances,
        )
        _build_replay_cache_parallel(
            source_dataset,
            global_indices=global_indices,
            output_dir=output_dir,
            manifest=manifest,
            max_tokens=max_tokens,
            shard_size_tokens=shard_size_tokens,
            dtype=npdtype,
            item_size=item_size,
            token_offset=token_offset,
            instance_index=instance_index,
            intra_instance_offset=intra_instance_offset,
            read_workers=read_workers,
            lookahead_instances=lookahead_instances,
        )
    else:
        _build_replay_cache_serial(
            source_dataset,
            global_indices=global_indices,
            output_dir=output_dir,
            manifest=manifest,
            max_tokens=max_tokens,
            shard_size_tokens=shard_size_tokens,
            dtype=npdtype,
            token_offset=token_offset,
            instance_index=instance_index,
            intra_instance_offset=intra_instance_offset,
        )

    replay_dataset_config = NumpyReplayFSLDatasetConfig(
        paths=manifest.resolved_shard_paths(output_dir),
        tokenizer=dataset_config.tokenizer,
        sequence_length=dataset_config.sequence_length,
    )
    replay_dataset = replay_dataset_config.build()
    manifest.replay_dataset_fingerprint = replay_dataset.fingerprint
    manifest.complete = True
    save_replay_manifest(output_dir, manifest)
    return manifest


def _should_use_parallel_sparse_reads(
    source_dataset: NumpyDatasetBase, *, read_workers: int, lookahead_instances: int
) -> bool:
    if read_workers <= 1 or lookahead_instances <= 1:
        return False
    if type(source_dataset) is not NumpyFSLDataset:
        return False
    return all(is_url(path) for path in source_dataset.paths)


def _build_replay_cache_serial(
    source_dataset: NumpyFSLDatasetBase,
    *,
    global_indices: np.ndarray,
    output_dir: Path,
    manifest: ReplayCacheManifest,
    max_tokens: int,
    shard_size_tokens: int,
    dtype,
    token_offset: int,
    instance_index: int,
    intra_instance_offset: int,
) -> None:
    while token_offset < max_tokens:
        remaining_tokens = max_tokens - token_offset
        target_shard_tokens = min(shard_size_tokens, remaining_tokens)
        shard_index = len(manifest.shard_paths)
        shard_rel_path = str(Path("shards") / f"part-{shard_index:05d}.npy")
        shard_abs_path = output_dir / shard_rel_path

        writer = _ReplayShardWriter(shard_abs_path, target_tokens=target_shard_tokens, dtype=dtype)
        shard_tokens_written = 0
        try:
            while not writer.is_full:
                source_idx = int(global_indices[instance_index])
                instance_tokens = source_dataset[source_idx]["input_ids"].numpy().astype(
                    dtype, copy=False
                )
                if intra_instance_offset > 0:
                    instance_tokens = instance_tokens[intra_instance_offset:]
                    intra_instance_offset = 0

                consumed = writer.write(instance_tokens)
                shard_tokens_written += consumed
                token_offset += consumed

                if consumed < len(instance_tokens):
                    intra_instance_offset = consumed
                else:
                    instance_index += 1

            writer.finalize()
        except BaseException:
            writer.abort()
            raise

        manifest.shard_paths.append(shard_rel_path)
        manifest.shard_token_counts.append(shard_tokens_written)
        manifest.total_tokens_written = token_offset
        save_replay_manifest(output_dir, manifest)
        log.info(
            "Wrote replay shard %s with %s tokens (%s / %s total)",
            shard_rel_path,
            f"{shard_tokens_written:,d}",
            f"{manifest.total_tokens_written:,d}",
            f"{max_tokens:,d}",
        )


def _build_replay_cache_parallel(
    source_dataset: NumpyFSLDataset,
    *,
    global_indices: np.ndarray,
    output_dir: Path,
    manifest: ReplayCacheManifest,
    max_tokens: int,
    shard_size_tokens: int,
    dtype,
    item_size: int,
    token_offset: int,
    instance_index: int,
    intra_instance_offset: int,
    read_workers: int,
    lookahead_instances: int,
) -> None:
    total_instances = len(global_indices)
    source_path_indices, local_instance_indices = _build_replay_source_plan(
        source_dataset, global_indices
    )
    next_output_position = instance_index
    next_submission_position = instance_index
    pending_results: dict[int, np.ndarray] = {}

    with ThreadPoolExecutor(max_workers=read_workers) as executor:
        while token_offset < max_tokens:
            remaining_tokens = max_tokens - token_offset
            target_shard_tokens = min(shard_size_tokens, remaining_tokens)
            shard_index = len(manifest.shard_paths)
            shard_rel_path = str(Path("shards") / f"part-{shard_index:05d}.npy")
            shard_abs_path = output_dir / shard_rel_path

            writer = _ReplayShardWriter(shard_abs_path, target_tokens=target_shard_tokens, dtype=dtype)
            shard_tokens_written = 0
            try:
                while not writer.is_full:
                    (
                        token_offset,
                        shard_tokens_written,
                        next_output_position,
                        intra_instance_offset,
                    ) = _drain_pending_replay_results(
                        writer,
                        pending_results,
                        token_offset=token_offset,
                        shard_tokens_written=shard_tokens_written,
                        next_output_position=next_output_position,
                        intra_instance_offset=intra_instance_offset,
                    )
                    if writer.is_full:
                        break

                    if next_submission_position >= total_instances:
                        raise RuntimeError(
                            "Replay planner ran out of source instances before filling the cache"
                        )

                    window_end = min(total_instances, next_submission_position + lookahead_instances)
                    window_positions = np.arange(next_submission_position, window_end, dtype=np.int64)
                    submit_order = np.lexsort(
                        (
                            local_instance_indices[window_positions],
                            source_path_indices[window_positions],
                        )
                    )

                    futures = {
                        executor.submit(
                            _fetch_replay_source_instance,
                            source_dataset.paths[int(source_path_indices[pos])],
                            int(local_instance_indices[pos]),
                            sequence_length=source_dataset.sequence_length,
                            dtype=dtype,
                        ): int(pos)
                        for pos in window_positions[submit_order]
                    }
                    try:
                        for future in as_completed(futures):
                            output_position = futures[future]
                            pending_results[output_position] = future.result()
                            (
                                token_offset,
                                shard_tokens_written,
                                next_output_position,
                                intra_instance_offset,
                            ) = _drain_pending_replay_results(
                                writer,
                                pending_results,
                                token_offset=token_offset,
                                shard_tokens_written=shard_tokens_written,
                                next_output_position=next_output_position,
                                intra_instance_offset=intra_instance_offset,
                            )
                    except BaseException:
                        for future in futures:
                            future.cancel()
                        raise

                    next_submission_position = window_end

                writer.finalize()
            except BaseException:
                writer.abort()
                raise

            manifest.shard_paths.append(shard_rel_path)
            manifest.shard_token_counts.append(shard_tokens_written)
            manifest.total_tokens_written = token_offset
            save_replay_manifest(output_dir, manifest)
            log.info(
                "Wrote replay shard %s with %s tokens (%s / %s total)",
                shard_rel_path,
                f"{shard_tokens_written:,d}",
                f"{manifest.total_tokens_written:,d}",
                f"{max_tokens:,d}",
            )


def _build_replay_source_plan(
    source_dataset: NumpyFSLDataset, global_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(source_dataset.offsets, dtype=np.int64)
    global_instance_indices = global_indices.astype(np.int64, copy=False)
    offset_starts = offsets[:, 0]
    offset_ends = offsets[:, 1]
    source_path_indices = np.searchsorted(offset_ends, global_instance_indices, side="right")

    if np.any(source_path_indices >= len(source_dataset.paths)):
        raise RuntimeError("Replay planner produced an invalid source path index")

    local_instance_indices = global_instance_indices - offset_starts[source_path_indices]
    return source_path_indices, local_instance_indices


def _fetch_replay_source_instance(
    path: PathOrStr, local_instance_index: int, *, sequence_length: int, dtype
) -> np.ndarray:
    start_idx = local_instance_index * sequence_length
    return load_array_slice(path, start_idx, start_idx + sequence_length, dtype)


def _drain_pending_replay_results(
    writer: _ReplayShardWriter,
    pending_results: dict[int, np.ndarray],
    *,
    token_offset: int,
    shard_tokens_written: int,
    next_output_position: int,
    intra_instance_offset: int,
) -> tuple[int, int, int, int]:
    while not writer.is_full and next_output_position in pending_results:
        instance_tokens = pending_results[next_output_position]
        tokens_to_write = (
            instance_tokens[intra_instance_offset:]
            if intra_instance_offset > 0
            else instance_tokens
        )
        consumed = writer.write(tokens_to_write)
        shard_tokens_written += consumed
        token_offset += consumed

        if consumed < len(tokens_to_write):
            intra_instance_offset += consumed
            break

        intra_instance_offset = 0
        del pending_results[next_output_position]
        next_output_position += 1

    return token_offset, shard_tokens_written, next_output_position, intra_instance_offset


def _cleanup_global_indices_file(data_loader) -> None:
    global_indices = getattr(data_loader, "_global_indices", None)
    if global_indices is not None:
        del global_indices
        data_loader._global_indices = None

    global_indices_file = getattr(data_loader, "_global_indices_file", None)
    if global_indices_file is None:
        return
    try:
        Path(global_indices_file).unlink(missing_ok=True)
    except OSError:
        log.warning("Failed to delete replay temporary global indices file '%s'", global_indices_file)


def _load_or_init_manifest(
    replay_root: Path,
    *,
    dataset_config: NumpyFSLDatasetConfig,
    data_seed: int,
    epoch: int,
    max_tokens: int,
    shard_size_tokens: int,
    source_dataset_fingerprint: str,
    replay_dtype: str,
) -> ReplayCacheManifest:
    manifest_path = get_replay_manifest_path(replay_root)
    if not manifest_path.is_file():
        manifest = ReplayCacheManifest(
            source_mix=_mix_value(dataset_config.mix),
            source_data_root=dataset_config.mix_base_dir,
            tokenizer_identifier=dataset_config.tokenizer.identifier,
            source_sequence_length=dataset_config.sequence_length,
            source_data_seed=data_seed,
            source_epoch=epoch,
            source_dataset_fingerprint=source_dataset_fingerprint,
            replay_dtype=replay_dtype,
            max_tokens_requested=max_tokens,
            shard_size_tokens=shard_size_tokens,
        )
        save_replay_manifest(replay_root, manifest)
        return manifest

    manifest = load_replay_manifest(replay_root)
    immutable_fields = {
        "source_mix": _mix_value(dataset_config.mix),
        "source_data_root": dataset_config.mix_base_dir,
        "tokenizer_identifier": dataset_config.tokenizer.identifier,
        "source_sequence_length": dataset_config.sequence_length,
        "source_data_seed": data_seed,
        "source_epoch": epoch,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "replay_dtype": replay_dtype,
        "shard_size_tokens": shard_size_tokens,
    }
    for field_name, expected_value in immutable_fields.items():
        if getattr(manifest, field_name) != expected_value:
            raise RuntimeError(
                f"Existing replay manifest at '{manifest_path}' does not match the requested cache parameters"
            )

    if max_tokens < manifest.max_tokens_requested:
        raise RuntimeError(
            f"Existing replay manifest at '{manifest_path}' does not match the requested cache parameters"
        )

    if max_tokens > manifest.max_tokens_requested:
        manifest.max_tokens_requested = max_tokens
        manifest.complete = manifest.total_tokens_written >= max_tokens
        save_replay_manifest(replay_root, manifest)

    return manifest


def _validate_existing_shards(replay_root: Path, manifest: ReplayCacheManifest, *, item_size: int) -> None:
    if len(manifest.shard_paths) != len(manifest.shard_token_counts):
        raise RuntimeError("Replay manifest is invalid: shard path/count lengths do not match")

    total_tokens = 0
    for rel_path, expected_tokens in zip(manifest.shard_paths, manifest.shard_token_counts):
        shard_path = replay_root / rel_path
        if not shard_path.is_file():
            raise RuntimeError(f"Replay shard listed in manifest is missing: '{shard_path}'")

        actual_tokens = os.stat(shard_path).st_size // item_size
        if actual_tokens != expected_tokens:
            raise RuntimeError(
                f"Replay shard size mismatch for '{shard_path}': expected {expected_tokens:,d} tokens, "
                f"found {actual_tokens:,d}"
            )
        total_tokens += actual_tokens

    if total_tokens != manifest.total_tokens_written:
        raise RuntimeError(
            "Replay manifest token count does not match committed shards: "
            f"{manifest.total_tokens_written:,d} != {total_tokens:,d}"
        )


class _ReplayShardWriter:
    def __init__(self, path: Path, *, target_tokens: int, dtype):
        if target_tokens <= 0:
            raise ValueError("'target_tokens' must be positive")

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.target_tokens = target_tokens
        self.dtype = dtype
        self.position = 0
        self.tmp_path = path.with_suffix(
            f".{random.SystemRandom().randint(0, 2**32)}.npy.tmp"
        )
        self._mmap = np.memmap(self.tmp_path, dtype=dtype, mode="w+", shape=(target_tokens,))

    @property
    def is_full(self) -> bool:
        return self.position >= self.target_tokens

    def write(self, arr: np.ndarray) -> int:
        remaining = self.target_tokens - self.position
        num_tokens = min(remaining, arr.shape[0])
        if num_tokens > 0:
            self._mmap[self.position : self.position + num_tokens] = arr[:num_tokens]
            self.position += num_tokens
        return num_tokens

    def finalize(self) -> None:
        if self.position != self.target_tokens:
            raise RuntimeError(
                f"Cannot finalize replay shard '{self.path}': "
                f"expected {self.target_tokens:,d} tokens, wrote {self.position:,d}"
            )
        self._mmap.flush()
        del self._mmap
        self.tmp_path.replace(self.path)

    def abort(self) -> None:
        try:
            del self._mmap
        finally:
            self.tmp_path.unlink(missing_ok=True)


class NumpyReplayFSLDataset(NumpyFSLDatasetBase):
    def __init__(
        self,
        *paths: PathOrStr,
        sequence_length: int,
        pad_token_id: int,
        eos_token_id: int,
        vocab_size: int,
        dtype=np.uint16,
        metadata=None,
        include_instance_metadata: Optional[bool] = None,
        generate_doc_lengths: bool = False,
        bos_token_id: Optional[int] = None,
        instance_filter_config=None,
    ):
        super().__init__(
            *paths,
            sequence_length=sequence_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            dtype=dtype,
            metadata=metadata,
            include_instance_metadata=include_instance_metadata,
            generate_doc_lengths=generate_doc_lengths,
            bos_token_id=bos_token_id,
            instance_filter_config=instance_filter_config,
        )
        self._source_offsets: Optional[tuple[tuple[int, int], ...]] = None
        self._num_instances: Optional[int] = None

    @property
    def num_tokens(self) -> int:
        return len(self) * self.sequence_length

    @property
    def source_offsets(self) -> tuple[tuple[int, int], ...]:
        if self._source_offsets is None or self._array_file_sizes is None:
            file_sizes = self.file_sizes
            item_size = self.dtype(0).itemsize
            offsets = []
            start = 0
            for size in file_sizes:
                token_count = size // item_size
                end = start + token_count
                offsets.append((start, end))
                start = end
            self._source_offsets = tuple(offsets)
        return self._source_offsets

    def prepare(self):
        len(self)

    def __len__(self) -> int:
        if self._num_instances is None:
            if not self.source_offsets:
                raise OLMoConfigurationError("Replay dataset has no source offsets")
            total_tokens = self.source_offsets[-1][1]
            self._num_instances = total_tokens // self.sequence_length
        return self._num_instances

    def __getitem__(self, index: int) -> dict[str, object]:
        index = int(index)
        pos_index = index if index >= 0 else len(self) + index
        if pos_index < 0 or pos_index >= len(self):
            raise IndexError(f"{index} is out of bounds for dataset of size {len(self)}")

        start_token = pos_index * self.sequence_length
        end_token = start_token + self.sequence_length
        chunks = []
        first_source_index: Optional[int] = None
        last_source_index: Optional[int] = None
        for source_index, (source_start, source_end) in enumerate(self.source_offsets):
            if source_end <= start_token:
                continue
            if source_start >= end_token:
                break
            local_start = max(start_token, source_start) - source_start
            local_end = min(end_token, source_end) - source_start
            chunks.append(
                load_array_slice_into_tensor(
                    self.paths[source_index], int(local_start), int(local_end), self.dtype
                )
            )
            if first_source_index is None:
                first_source_index = source_index
            last_source_index = source_index

        input_ids = torch.cat(chunks, dim=0)
        if input_ids.numel() != self.sequence_length:
            raise RuntimeError(
                f"Expected replay instance length {self.sequence_length}, got {input_ids.numel()}"
            )

        out: dict[str, object] = {"input_ids": input_ids}
        if self.instance_filter_config is not None:
            out["instance_mask"] = self._validate_instance(input_ids, self.instance_filter_config)
        if self._include_instance_metadata and first_source_index == last_source_index:
            out["metadata"] = dict(self._metadata[first_source_index])
        if self._generate_doc_lengths:
            out["doc_lens"] = get_document_lengths(
                input_ids, self.eos_token_id, bos_token_id=self.bos_token_id
            )
        return out


@dataclass(kw_only=True)
class NumpyReplayFSLDatasetConfig(NumpyDatasetConfig):
    sequence_length: int
    generate_doc_lengths: bool = False

    def validate(self):
        if self.sequence_length <= 0:
            raise OLMoConfigurationError("'sequence_length' must be positive")
        if self.mix is not None:
            raise OLMoConfigurationError("'mix' is not supported for replay datasets")

    def build(self) -> NumpyDatasetBase:
        self.validate()
        paths, metadata, _ = self._resolve_paths_metadata(allow_mix=False)
        dataset = NumpyReplayFSLDataset(
            *paths,
            sequence_length=self.sequence_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            vocab_size=self.tokenizer.vocab_size,
            dtype=self.get_dtype(),
            metadata=metadata,
            include_instance_metadata=self.include_instance_metadata,
            generate_doc_lengths=self.generate_doc_lengths,
            bos_token_id=self.tokenizer.bos_token_id,
            instance_filter_config=self.instance_filter_config,
        )
        return self._finalize(dataset)
