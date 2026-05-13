# Shared-MoE Follow-ups

Backlog of follow-ups for the `SharedBlockConfig` feature (shared routers/experts/attention/norms across a contiguous range of middle layers). Nothing here is required for correctness in single-host training; these are improvements / hardening for production and large-scale use.

## Status at a glance

| # | Item | Status | Commit |
|---|------|--------|--------|
| 1 | Memory-efficient construction | ⏳ open | — |
| 2 | Fine-grained FSDP support | ⏳ open | — |
| 3 | `state_dict` deduplication | ✅ done | `a1f04c2` |
| 4 | EP + FSDP-blocks integration test | ✅ done (multi-GPU only) | `de45303` |
| 5 | Router metric regression test | ✅ done | `850009c` |
| 6 | `num_flops_per_token` regression test | ✅ done | `850009c` |
| 7 | Pipeline-parallelism interaction | ⏳ open | — |
| 8 | `apply_tp` / `apply_cp` dedup | ✅ done | `f273d77` |
| 9 | `init_weights` per-block dedup | ✅ done | `df04868` |
| 10 | `block_overrides` ∩ `shared_blocks` validation | ✅ done | `a7d9785` |
| 11 | Tests for `share_attention` / `share_norms` / `share_shared_mlp` | ✅ done | `850009c` |
| 12 | Usage example for `SharedBlockConfig` | 🚧 partial — `ml/README.md` has the section; `SharedBlockConfig` docstring still has no inline example |

**Completion summary:** 9 of 12 closed, 1 partial, 3 open. The three open items (#1, #2, #7) are all efficiency / large-scale hardening — none are correctness blockers for single-node training.

Per-fix walkthroughs for completed items live in `SHARED_MOE_FIX_NOTES.md`. Implementation plans for the highest-priority items are in `SHARED_MOE_TOP3_PLANS.md`.

---

# Open

## 1. Memory-efficient construction

**What:** Today `Transformer.__init__` builds every block — including the non-canonical ones in the shared range — and then `_apply_shared_blocks` reassigns the shared submodules and lets GC reclaim the discards.

**Why it might matter:** With `init_device="meta"` (the standard pretraining path) this is essentially free — no real allocations happen until materialization. With `init_device="cuda"` and large MoE experts, we transiently allocate N copies before freeing N−1, which can OOM at the boundary.

**Suggested fix:** Add a `build_skipping_moe(...)` (or similar) path on `TransformerBlockConfig` that lets `Transformer.__init__` construct non-canonical blocks without instantiating the components that will be overwritten by sharing. Walk the `SharedBlockConfig` flags to decide which components to skip per index.

## 2. Fine-grained FSDP support for shared modules

**What:** `Transformer.apply_fsdp` currently raises `OLMoConfigurationError` if `wrapping_strategy="fine_grained"` is requested together with `shared_blocks`. The reason: fine-grained wrapping calls `fully_shard(self.feed_forward_moe, ...)` from inside each block's `apply_fsdp`, and wrapping the same module twice corrupts FSDP state.

**Why it might matter:** Fine-grained wrapping can improve memory and overlap for very large models. We're forcing users to fall back to whole-block wrapping when sharing is enabled.

**Suggested fix:** At the `Transformer` level, identify the unique shared submodules up front and wrap them with `fully_shard` once. Pass a `skip_modules: set[int]` (set of `id(...)`) into `block.apply_fsdp` so each block's fine-grained pass skips already-wrapped children. The pattern already exists for TP/CP (`#8` below) — same trick, applied to FSDP. Touches every block class that overrides `apply_fsdp` (TransformerBlock, MoETransformerBlock, hybrid variants), so it's mechanical but not zero-cost to land.

## 7. Pipeline-parallelism interaction

**What we haven't thought about:** `Transformer.apply_pp` and the various `_pp_*` state on the model. PP slices the block list across stages. If a shared range straddles a stage boundary, the same module would need to live on multiple stages — which doesn't work without explicit replication or weight sync.

**Suggested behavior:** In `apply_pp`, detect whether `shared_blocks.layer_indices()` is fully contained within a single PP stage's block range; raise `OLMoConfigurationError` otherwise. Document the constraint.

## 12. Usage example for `SharedBlockConfig` (partial)

**What's done:** `ml/README.md` has a "Shared MoE blocks" section with the flag reference, three worked examples, and a sweep-subgrid snippet. The CLI-side documentation is complete.

**What's still missing:** The `SharedBlockConfig` dataclass itself (in `src/olmo_core/nn/transformer/config.py`) has per-field descriptions but no inline usage example in its docstring. A library consumer reading the source would not see a recommended config without crossing over into `ml/README.md`.

**Suggested fix:** Add a short example block (maybe 10-15 lines) to the `SharedBlockConfig` docstring. Cover the typical case (`share_routers=True, share_experts=True`, middle range of an N-layer model) plus a note that `n_layers` here is the *count of layers in the shared range*, not the total model depth (an easy thing to misread).

---

# Completed

Each entry below points at its implementing commit and the walkthrough section in `SHARED_MOE_FIX_NOTES.md`.

## 3. `state_dict` deduplication ✅

**Landed in:** `a1f04c2`. Walkthrough: `SHARED_MOE_FIX_NOTES.md` → "TODO #3 — state_dict deduplication".

**Summary:** Override `Transformer.state_dict()` to emit only canonical paths for shared `nn.Parameter`s; override `Transformer.load_state_dict()` to fan canonical entries back out to alias paths so PyTorch's `strict=True` walk is satisfied, and raise `RuntimeError` when the source dict has *differing* per-layer values for what's now a single shared Parameter (catches silent corruption when warm-starting from an unshared pretrained checkpoint). Works transparently through DCP's `save_model_and_optim_state` / `load_model_and_optim_state`.

## 4. EP + FSDP-blocks integration test ✅

**Landed in:** `de45303`. The test is `test_shared_blocks_ep_fsdp_integration` in `src/test/nn/transformer/model_test.py`, marked `@requires_multi_gpu` so it skips on single-GPU/CPU but runs on multi-GPU CI.

**Summary:** Builds a small shared-MoE model on `init_device="meta"`, applies EP on a 2-GPU mesh, then FSDP with `wrapping_strategy="blocks"`, runs forward+backward+optim across 2 steps, asserts loss is finite both times. Validates that the parallelism-dedup fixes (#8 + EP + FSDP) compose cleanly when DTensor and FSDP layer on top of physically-shared parameters.

## 5. Router metric regression test ✅

**Landed in:** `850009c`. Test: `test_shared_router_metric_accumulates_across_calls`.

**Summary:** Builds an `MoELinearRouter` directly (router forward is plain PyTorch, runs on CPU) and calls it twice with identical inputs. Asserts `load_balancing_loss` and `batch_size_per_expert` accumulators double — i.e. `+=` is being used, not overwrite. This is the property that makes shared-router-across-N-blocks produce sensible metrics.

## 6. `num_flops_per_token` regression test ✅

**Landed in:** `850009c`. Test: `test_shared_blocks_num_flops_per_token_equals_unshared`.

**Summary:** Builds a shared-blocks model and an equivalent unshared model with the same `n_layers`, asserts `shared.num_flops_per_token(seq_len) == unshared.num_flops_per_token(seq_len)`. Locks in the invariant that sharing only changes parameter footprint, not per-token work.

## 8. `apply_tp` and `apply_cp` dedup ✅

**Landed in:** `f273d77`. Walkthrough: `SHARED_MOE_FIX_NOTES.md` → "TODO #8 — apply_tp / apply_cp dedup".

**Summary:** Threaded an optional `skip_modules: Set[int]` kwarg through the abstract `TransformerBlockBase.apply_tp` / `.apply_cp` signatures and all concrete overrides. The model-level driver constructs one set and passes it to each per-block call; each block's method gates its inner-submodule recursion on `id(submodule) not in skip_modules`. Result: a shared `feed_forward_moe` / `attention` / norm is parallelized exactly once across N referencing blocks, never N times.

## 9. `init_weights` per-block dedup ✅

**Landed in:** `df04868`. Walkthrough: `SHARED_MOE_FIX_NOTES.md` → "TODO #9 — init_weights per-block dedup".

**Summary:** Added four `seen_*: Set[int]` tracking sets in `Transformer.init_weights` and gated each `init_method.init_*` and `*.warmup_cache(...)` call on `id(submodule)`. Because `self.blocks` is a `ModuleDict` iterated in numerical key order, the *canonical* (lowest) `block_idx` drives init for shared submodules — which is what depth-scaled init schemes like `InitMethod.llama_depth` actually want. Pre-fix, the last block_idx in the shared range silently won.

## 10. `block_overrides` ∩ `shared_blocks` validation ✅

**Landed in:** `a7d9785`. Walkthrough: `SHARED_MOE_FIX_NOTES.md` → "TODO #10 — block_overrides × shared_blocks validation".

**Summary:** In `Transformer.__init__`, intersect `block_overrides` keys with `shared_blocks.layer_indices()` and raise `OLMoConfigurationError` if any overlap. Prevents the silent footgun where a `block_overrides` entry at an index inside the shared range gets clobbered by `_apply_shared_blocks` (silently producing a wrong model if shapes match, or a confusing far-away crash if they don't).

## 11. Tests for `share_attention` / `share_norms` / `share_shared_mlp` ✅

**Landed in:** `850009c`. Three tests: `test_shared_blocks_share_attention_identity`, `test_shared_blocks_share_norms_identity`, `test_shared_blocks_share_shared_mlp_identity`.

**Summary:** Module-identity tests modeled on the existing `test_shared_blocks_module_identity`. Each builds a small model with the relevant flag enabled in isolation and asserts `is` aliasing of the right submodule across the shared range, plus distinct identity outside the range. `share_shared_mlp` uses the hybrid block type (`moe_hybrid_reordered_norm`) because the dense `shared_mlp` only exists there.
