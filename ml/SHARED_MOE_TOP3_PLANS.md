# Top-3 Shared-MoE Follow-up Plans

Implementation plans for the three highest-priority items from `ml/SHARED_MOE_TODOS.md`:

1. `apply_tp` / `apply_cp` dedup (TODO #8) — TP/CP correctness gap of the same class as the EP/FSDP fix.
2. `init_weights` per-block dedup (TODO #9) — silent correctness issue with depth-scaled init.
3. `state_dict` deduplication (TODO #3) — checkpoint bloat + silent load-time winner.

Each plan is structured as: **Goal / Approach / Files / Code sketch / Tests / Verification**.

---

## Plan 1 — `apply_tp` and `apply_cp` dedup

### Goal
Calling `Transformer.apply_tp(...)` or `Transformer.apply_cp(...)` with `shared_blocks` set must apply TP/CP exactly once per unique submodule — never re-wrap a shared submodule.

### Approach
Mirror the dedup pattern already used in `MoETransformer.apply_ep` (model.py:1062-1073). The wrinkle vs. `apply_ep`: the per-block `apply_tp` does *two* things — it wraps the block itself with `parallelize_module(self, ...)` (which is always safe, since blocks are unique even when sharing), AND it recurses into inner submodules like `self.attention` and `self.feed_forward_moe`. The double-wrap risk is only on the inner recursion, so we plumb a "skip set" into the block-level `apply_tp` rather than skipping the whole block.

For `apply_cp` the story is simpler — the block-level `apply_cp` only delegates inward (no block-level parallelize wrap), so we can do the dedup entirely at the model level by inspecting shared identity.

### Files to modify
- `src/olmo_core/nn/transformer/model.py`:
  - `Transformer.apply_tp` (lines 649-687) — drive dedup
  - `Transformer.apply_cp` (lines 689-710) — drive dedup
- `src/olmo_core/nn/transformer/block.py`:
  - `TransformerBlock.apply_tp` (~lines 155-199) — accept optional `skip_modules` arg
  - `MoETransformerBlock.apply_tp` (lines 605-643) — accept optional `skip_modules` arg, gate the `self.attention.apply_tp(...)` and `self.feed_forward_moe.apply_tp(...)` calls on `id(m) not in skip_modules`
  - Any other block class that overrides `apply_tp` (hybrid + reordered_norm variants) — same change
  - `MoETransformerBlock.apply_cp` — for shared attention/MoE, only call `apply_cp` once per unique identity (gated similarly)
- `src/test/nn/transformer/model_test.py` — multi-GPU regression test (see Tests).

### Code sketch

```python
# model.py — Transformer.apply_tp
def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: Optional[bool] = None):
    ...
    # Build the set of inner submodule ids that have already been parallelized.
    # The block itself is always unique, so we only need to track inner subs.
    seen_inner: Set[int] = set()
    for block in self.blocks.values():
        block = cast(TransformerBlockBase, block)
        block.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            float8_enabled=float8_enabled,
            skip_modules=seen_inner,  # NEW kwarg, mutated by block.apply_tp
        )
    ...

# block.py — MoETransformerBlock.apply_tp
def apply_tp(self, tp_mesh, *, input_layout, float8_enabled=False, skip_modules=None):
    skip_modules = skip_modules if skip_modules is not None else set()
    parallelize_module(self, ...)  # always unique — block itself is not shared
    parallelize_module(self.attention_norm, ...)  # norms either unique OR shared-but-norms
    # ^ NB: if share_norms=True, attention_norm is shared. Handle the same way:
    if id(self.attention_norm) not in skip_modules:
        parallelize_module(self.attention_norm, ...)
        skip_modules.add(id(self.attention_norm))
    if id(self.attention) not in skip_modules:
        self.attention.apply_tp(tp_mesh, ...)
        skip_modules.add(id(self.attention))
    if id(self.feed_forward_norm) not in skip_modules:
        parallelize_module(self.feed_forward_norm, ...)
        skip_modules.add(id(self.feed_forward_norm))
    if id(self.feed_forward_moe) not in skip_modules:
        self.feed_forward_moe.apply_tp(tp_mesh, ...)
        skip_modules.add(id(self.feed_forward_moe))
    parallelize_module(self.dropout, ...)  # unique
    self._tp_enabled = True
```

```python
# model.py — Transformer.apply_cp
def apply_cp(self, cp_mesh, ring=None, uly=None):
    ...
    seen: Set[int] = set()
    for block in self.blocks.values():
        block = cast(TransformerBlockBase, block)
        block.apply_cp(cp_mesh, ring=ring, uly=uly, skip_modules=seen)
    if self.lm_head is not None:
        self.lm_head.apply_cp(cp_mesh)
```

Mirror the `skip_modules` plumbing into `MoETransformerBlock.apply_cp` (block.py:645-652) — gate the two `apply_cp` calls inside on `id(...)`.

### Tests
- CPU: a `test_shared_blocks_apply_tp_no_double_wrap` unit test that builds a small shared-blocks model and asserts each unique submodule's `parallelize_module` is invoked exactly once. Use a mock for `parallelize_module` to count calls per `id()`. No actual TP mesh needed.
- Multi-GPU (mark `@requires_multi_gpu`): apply TP with `shared_blocks` enabled on a 2-GPU mesh, do a forward+backward+optim step, assert loss is finite and decreases on a second step. This is the real validation.

### Verification
1. Existing tests still pass: `pytest src/test/nn/transformer/ -k "shared_blocks"`.
2. New mock-based TP dedup test passes on CPU.
3. Manually exercise TP path with `shared_blocks` set in a 2-GPU run if a multi-GPU box is available.

### Effort estimate
~half a day. The block subclasses that override `apply_tp` (TransformerBlock, MoETransformerBlock, MoEReorderedNormTransformerBlock, MoEHybridTransformerBlock + its reordered variant) all need the same plumbing pass.

---

## Plan 2 — `init_weights` per-block dedup

### Goal
When a submodule is shared across N blocks, `init_method.init_*(submodule, block_idx=…)` should run exactly once with the **canonical** `block_idx` (the lowest block index in the shared range), not N times with N different `block_idx`s.

### Why this matters concretely
From `src/olmo_core/nn/transformer/init.py`:
- `InitMethod.llama` scales std by `1 / sqrt(2 * num_blocks)` — same across all blocks, so order doesn't matter (any init is correct).
- `InitMethod.llama_depth` scales std by `1 / sqrt(2 * (block_idx + 1))` — **depends on `block_idx`**.
- `InitMethod.normalized` scales by `num_blocks` only — order doesn't matter.
- `InitMethod.normal` — no scaling.

So under `llama_depth` (which is the only currently-affected variant), a shared param initialized N times with `block_idx` ∈ {1, 2, 3, 4} ends up with the std for `block_idx=4`, silently. Other variants happen to be order-independent, but the principle (don't re-init shared params) applies to any future depth-scaled init we add.

### Approach
Dedup the inner loop by `id(submodule)`. Use the canonical (lowest-index) block's `block_idx` so depth-scaling matches the intuitive "this is layer K" semantics.

Notable subtlety: `init_weights` also calls cache-warmup functions (`block.feed_forward_moe.warmup_cache(...)`, `att.backend.warmup_cache(...)`, `att.rope.warmup_cache(...)`) inside the same loop. Those are *not* init and don't depend on `block_idx` — they can run multiple times harmlessly. But for cleanliness, dedup them too: there's no value in warming up the same cache N times.

### Files to modify
- `src/olmo_core/nn/transformer/model.py`:
  - `Transformer.init_weights` (lines 295-374) — gate the per-block `init_attention` / `init_feed_forward` / `init_feed_forward_moe` calls on `id(submodule) not in seen`, populated in iteration order so the canonical (first-encountered = lowest block_idx) wins.
- `src/test/nn/transformer/model_test.py` — regression test.

### Code sketch

```python
# model.py — Transformer.init_weights
def init_weights(self, *, max_seq_len=None, max_local_microbatch_size=None,
                 device=None, world_mesh=None) -> torch.Generator:
    device = device or self.device
    self.to_empty(device=device)
    for module in self.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    seed = self.init_seed
    if world_mesh is not None and self.pp_enabled:
        seed += get_pp_mesh(world_mesh).get_local_rank()
    generator = torch.Generator(device).manual_seed(seed)

    if self.embeddings is not None:
        self.init_method.init_embeddings(...)

    seen_attn: Set[int] = set()
    seen_ff: Set[int] = set()
    seen_moe: Set[int] = set()
    seen_warmup: Set[int] = set()

    for block in self.blocks.values():
        block = cast(TransformerBlock, block)
        att = cast(Union[Attention, FusedAttention], block.attention)

        if id(att) not in seen_attn:
            seen_attn.add(id(att))
            self.init_method.init_attention(
                att, d_model=self.d_model, block_idx=block.block_idx,
                num_blocks=self.n_layers, std=self.init_std, generator=generator,
            )

        if hasattr(block, "feed_forward") and id(block.feed_forward) not in seen_ff:
            seen_ff.add(id(block.feed_forward))
            self.init_method.init_feed_forward(
                block.feed_forward, d_model=self.d_model, block_idx=block.block_idx,
                num_blocks=self.n_layers, std=self.init_std, generator=generator,
            )

        if hasattr(block, "feed_forward_moe") and id(block.feed_forward_moe) not in seen_moe:
            seen_moe.add(id(block.feed_forward_moe))
            block = cast(MoETransformerBlock, block)
            if max_local_microbatch_size is not None:
                block.feed_forward_moe.warmup_cache(max_local_microbatch_size)
            self.init_method.init_feed_forward_moe(
                block.feed_forward_moe, d_model=self.d_model, block_idx=block.block_idx,
                num_blocks=self.n_layers, std=self.init_std, generator=generator,
            )

        # Cache warmups — also dedup'd for cleanliness, not correctness
        if max_seq_len is not None and att.backend is not None and id(att.backend) not in seen_warmup:
            seen_warmup.add(id(att.backend))
            att.backend.warmup_cache(max_seq_len, device)
        if max_seq_len is not None and att.rope is not None and id(att.rope) not in seen_warmup:
            seen_warmup.add(id(att.rope))
            att.rope.warmup_cache(max_seq_len, device)

    if self.lm_head is not None:
        self.init_method.init_lm_head(self.lm_head, ...)
    return generator
```

Note: iterating `self.blocks.values()` visits blocks in dict insertion order, which is `'0', '1', '2', ...`. So the canonical (lowest-block_idx) entry is encountered first naturally. No need to pre-sort.

### Tests
- A CPU test using `InitMethod.llama_depth` that builds a shared-blocks model, calls `init_weights()`, and asserts that the shared param's std matches the std computed for the **canonical** `block_idx`, not the highest one in the range. Approach: capture the std on the first init call (e.g. via monkey-patching `torch.nn.init.trunc_normal_` to record the std arg), and compare.
- Simpler alternative: build two models with `llama_depth` init — one with `shared_blocks=(start=1, n=3)` and one without — and assert the shared param in the sharing model matches block 1's params in the unshared model (not block 3's).

### Verification
1. `pytest src/test/nn/transformer/ -k "shared_blocks"` — all existing pass.
2. New regression test passes.
3. Sanity: train a tiny shared-blocks model with `llama_depth` init for a few steps; loss should look comparable to an equivalent unshared model with the same canonical-layer weights.

### Effort estimate
~half a day. Small surface, but needs a careful test to lock in the "canonical block_idx wins" semantics.

---

## Plan 3 — `state_dict` deduplication

### Goal
1. Saving a shared-blocks model should produce a checkpoint with each shared `Parameter` stored exactly once (under its canonical-block path), not N times.
2. Loading a checkpoint (either shared-saved or unshared-saved) into a shared-blocks model should be unambiguous: either it works correctly, or it raises a clear error if the source has inconsistent values for what's now a shared param.

### Findings from exploration (anchored in the actual code)

- `save_model_and_optim_state` (`src/olmo_core/distributed/checkpoint/__init__.py:189-255`) calls `dist_cp_sd.get_model_state_dict(model, options=StateDictOptions(full_state_dict=False, cpu_offload=True, ...))` which internally calls `model.state_dict()`. PyTorch's `state_dict()` does **not** dedup shared-Parameter paths — every parent module's path appears.
- The DCP planner uses `dedup_save_to_lowest_rank=True`, which **only dedups REPLICATED tensors across ranks** (DTensor identity), **not shared-Parameter aliasing within a single rank**. So the on-disk file still has N copies of every shared expert weight.
- `load_model_and_optim_state` (lines 302-393) calls `dist_cp.load(...)` then `set_model_state_dict(model, sd, ...)` which calls `model.load_state_dict(sd, strict=False)`. PyTorch's `load_state_dict` walks every key and calls `param.data.copy_(value)` — for shared params this means N copies, last-write-wins, silently. If source values disagree (e.g. loading an unshared checkpoint into a shared model), N−1 of the source values are silently discarded.

### Approach
Two-sided fix:

**Save side (dedup):** Override `Transformer.state_dict()` to walk the state dict PyTorch built and drop entries whose tensor `id()` was already encountered at a lexicographically-earlier (canonical) path.

**Load side (correctness):** Override `Transformer.load_state_dict()` to:
1. Detect duplicate paths in the incoming state_dict that resolve to the same Parameter under the current model (i.e., paths in the shared range beyond the canonical one).
2. For each shared Parameter, if the source state_dict has ≥1 entry, use the canonical-path entry preferentially. If the source has multiple entries and they disagree, **raise a clear error** explaining that the source has per-layer weights for what's now a shared module — the user must pick the desired warm-start strategy explicitly.
3. After resolving, call into `super().load_state_dict(filtered_sd, strict=False)`.

This keeps load behavior strict and audible. It does *not* try to auto-merge unshared → shared (averaging, picking layer K, etc.) — that's an explicit policy decision the user should make.

### Files to modify
- `src/olmo_core/nn/transformer/model.py`:
  - Add a helper `_shared_param_paths_map(self) -> dict[int, list[str]]` returning `{param_id: [canonical_path, *other_paths]}` for every shared Parameter. Walk `self.named_parameters(remove_duplicate=False)` and group by `id()`.
  - Override `state_dict(self, *args, destination=None, prefix='', keep_vars=False, **kw)` — call `super().state_dict(...)` then remove non-canonical entries.
  - Override `load_state_dict(self, state_dict, strict=True, **kw)` — preprocess `state_dict` as described above, then defer to `super().load_state_dict(...)`. Raise on disagreeing-values for shared params.
- (Optional) `src/olmo_core/distributed/checkpoint/__init__.py` — no changes needed if the model-level override is enough; DCP calls `model.state_dict()` and `model.load_state_dict()` so the override takes effect transparently.
- `src/test/nn/transformer/model_test.py` — round-trip tests.

### Code sketch

```python
# model.py
def _shared_param_paths_map(self) -> Dict[int, List[str]]:
    """For each Parameter referenced by >1 path, return canonical-first list of all paths."""
    by_id: Dict[int, List[str]] = {}
    for name, p in self.named_parameters(remove_duplicate=False):
        by_id.setdefault(id(p), []).append(name)
    return {pid: sorted(paths) for pid, paths in by_id.items() if len(paths) > 1}

def state_dict(self, *args, destination=None, prefix='', keep_vars=False, **kw):
    sd = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars, **kw)
    if self.shared_blocks is None:
        return sd
    drop: Set[str] = set()
    for pid, paths in self._shared_param_paths_map().items():
        # Keep canonical (sorted-first) path; drop the rest.
        canonical, *aliases = paths
        for a in aliases:
            full = f"{prefix}{a}"
            if full in sd:
                drop.add(full)
    for k in drop:
        del sd[k]
    return sd

def load_state_dict(self, state_dict, strict=True, **kw):
    if self.shared_blocks is None:
        return super().load_state_dict(state_dict, strict=strict, **kw)
    shared_map = self._shared_param_paths_map()
    if not shared_map:
        return super().load_state_dict(state_dict, strict=strict, **kw)

    # For each shared param, pick the value to load and detect disagreement.
    sd = dict(state_dict)  # shallow copy so we can mutate
    for pid, paths in shared_map.items():
        canonical, *aliases = paths
        present = [p for p in paths if p in sd]
        if len(present) == 0:
            continue  # let super() handle missing-key reporting
        if len(present) > 1:
            # Check for value disagreement; error if so.
            ref = sd[present[0]]
            for p in present[1:]:
                if not torch.equal(sd[p], ref):
                    raise RuntimeError(
                        f"State dict has per-layer values for {paths} but the model "
                        f"physically shares this Parameter across those layers. "
                        f"Source values disagree — pick a warm-start strategy explicitly "
                        f"(e.g. preload only the canonical-layer weights, or average)."
                    )
        # Ensure the canonical path is populated; drop aliases.
        if canonical not in sd:
            sd[canonical] = sd[present[0]]
        for a in aliases:
            sd.pop(a, None)

    return super().load_state_dict(sd, strict=strict, **kw)
```

Caveats:
- The `prefix` handling in `state_dict` is subtle — when the model is wrapped (FSDP, etc.) the paths may have prefixes. The helper needs to match the actual emitted keys; the sketch above assumes paths line up with `named_parameters` names, which holds for the unwrapped model. For FSDP-wrapped models, DCP calls `get_model_state_dict` which handles the wrapping internally; we just need the override to work on the unwrapped path PyTorch sees.
- `torch.equal` on DTensor params requires care — may need `get_full_tensor(...)` first, or compare local shards on a single rank. For a first cut, only compare if both are plain Tensors; skip the disagreement check on DTensors and add a TODO.

### Tests
- **Round-trip with sharing.** Build a shared-blocks model, save its `state_dict()`, assert no duplicate keys for shared params. Load it back into a fresh shared-blocks model, assert weights identical.
- **Round-trip with sharing → unshared.** Save shared-model state_dict, load into an unshared model — the canonical key should fan out to all blocks (or, more practically, the unshared model has all paths but only the canonical key is in the dict → expect strict=False or pre-expansion).
- **Unshared → shared (warm-start).** Save unshared model, attempt to load into shared model. Assert RuntimeError mentioning "values disagree" when source layers have non-identical params (the common case).
- **DCP integration.** `save_model_and_optim_state(checkpoint_dir, shared_model)` → `load_model_and_optim_state(checkpoint_dir, fresh_shared_model)` round-trip. Multi-GPU test (`@requires_multi_gpu`).

### Verification
1. Existing unit tests: `pytest src/test/nn/transformer/ -k "shared_blocks"`.
2. New round-trip tests pass.
3. Manual: save a real shared-blocks training checkpoint, inspect the on-disk file size vs. the unshared baseline — should be smaller by `(n_layers_shared - 1) * shared_module_size`.
4. End-to-end: resume a paused shared-blocks training run from a saved checkpoint, confirm loss matches the pre-pause trajectory.

### Effort estimate
~1-1.5 days. The save-side dedup is simple. The load-side correctness checks plus the DCP integration test are where the time goes. Worth running an actual `save_model_and_optim_state` → `load_model_and_optim_state` round-trip multi-GPU to be sure DCP doesn't drop the override somewhere upstream.

---

## Suggested ordering

Do **#9 (`init_weights`)** first — smallest surface, locks in a real correctness fix that affects every shared-blocks training run with depth-scaled init.

Then **#8 (TP/CP dedup)** — also a correctness fix, mechanical, no design needed. Unblocks anyone wanting to combine sharing with TP/CP.

Then **#3 (state_dict dedup)** — the longest of the three but no correctness blocker until checkpointing matters (i.e., not for the first training run on a fresh model).

All three can land independently; no cross-dependencies.
