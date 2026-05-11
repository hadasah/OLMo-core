# Shared-MoE Follow-ups

Open items left over from the `SharedBlockConfig` implementation (shared routers/experts/attention/norms across a contiguous range of middle layers). Nothing here is required for correctness in single-host training; these are improvements / hardening for production and large-scale use.

## 1. Memory-efficient construction

**What:** Today `Transformer.__init__` builds every block — including the non-canonical ones in the shared range — and then `_apply_shared_blocks` reassigns the shared submodules and lets GC reclaim the discards.

**Why it might matter:** With `init_device="meta"` (the standard pretraining path) this is essentially free — no real allocations happen until materialization. With `init_device="cuda"` and large MoE experts, we transiently allocate N copies before freeing N−1, which can OOM at the boundary.

**Suggested fix:** Add a `build_skipping_moe(...)` (or similar) path on `TransformerBlockConfig` that lets `Transformer.__init__` construct non-canonical blocks without instantiating the components that will be overwritten by sharing. Walk the `SharedBlockConfig` flags to decide which components to skip per index.

## 2. Fine-grained FSDP support for shared modules

**What:** `Transformer.apply_fsdp` currently raises `OLMoConfigurationError` if `wrapping_strategy="fine_grained"` is requested together with `shared_blocks`. The reason: fine-grained wrapping calls `fully_shard(self.feed_forward_moe, ...)` from inside each block's `apply_fsdp`, and wrapping the same module twice corrupts FSDP state.

**Why it might matter:** Fine-grained wrapping can improve memory and overlap for very large models. We're forcing users to fall back to whole-block wrapping when sharing is enabled.

**Suggested fix:** At the `Transformer` level, identify the unique shared submodules up front and wrap them with `fully_shard` once. Pass a `skip_modules: set[int]` (set of `id(...)`) into `block.apply_fsdp` so each block's fine-grained pass skips already-wrapped children. This touches every block class that overrides `apply_fsdp` (TransformerBlock, MoETransformerBlock, hybrid variants), so it's mechanical but not zero-cost to land.

## 3. `state_dict` deduplication

**What:** When the same `nn.Parameter` is referenced by N blocks, `state_dict()` walks the module tree without dedup and emits N keys (e.g., `blocks.1.feed_forward_moe.routers_list.0.weight` through `blocks.4...`), all pointing to the same tensor.

**Three concrete consequences:**

1. **Checkpoint bloat.** The serialized checkpoint contains N copies of every shared tensor. For a real MoE most of the checkpoint mass is in experts — could be 3–4× larger than necessary depending on the shared range.
2. **Silent winner on load.** `load_state_dict` resolves all N keys to the same Parameter, then calls `param.data.copy_(value)` once per key. If you load an unshared checkpoint into a sharing-enabled model (e.g., a fine-tuning warm-start), N−1 of the source values get discarded — last-write-wins, no warning.
3. **Strict-mode round-trip surprises** when converting between shared/unshared model variants. Going `shared → unshared` works but produces N identical copies. Going `unshared → shared` hits #2.

**Open question before fixing:** OLMo-core saves via DCP (`save_model_and_optim_state` / `load_model_and_optim_state`), which has its own DTensor-aware behavior. **Verify what DCP does today with shared params** before adding any override — it may already dedup.

**Suggested fix (if DCP doesn't dedup):** Override `Transformer.state_dict` and `_load_from_state_dict` to write/expect each shared Parameter only under its canonical path (the start-layer path). The model stays internally consistent either way; this is purely about the on-disk representation.

## 4. End-to-end EP + FSDP-blocks interaction test

**What:** The plan calls for a 2-GPU integration test where `apply_ep(ep_mesh)` is applied first, then `apply_fsdp(wrapping_strategy="blocks")`, with `shared_blocks` enabled. We've unit-tested module identity, param counts, validation, and partial sharing on CPU; the GPU forward/backward test is `@pytest.mark.gpu` but doesn't exercise the EP+FSDP interaction.

**Why it might matter:** Each shared expert weight becomes a DTensor under EP. The "blocks" FSDP wrap then encloses each block (including its reference to the now-DTensor-ified shared experts). FSDP 2's behavior when a DTensor parameter lives under multiple FSDP units needs empirical confirmation.

**Suggested test:** A multi-GPU pytest (`@requires_multi_gpu`) that builds a small shared-MoE model, applies EP then FSDP, runs a forward + backward + optimizer step, and checks loss decreases. Run with `n_layers=4, shared_blocks=(start=1, n=2, share_routers+experts)`.

## 5. Router metric audit — done, but worth a regression test

**Status:** Inspected `MoERouter.forward` — `load_balancing_loss`, `z_loss`, and `batch_size_per_expert` all accumulate via `+=`. A shared router called N times in one forward pass naturally accumulates N contributions. `MoETransformer.compute_auxiliary_metrics` and `reset_auxiliary_metrics` were updated to iterate by unique MoE id so we don't read-then-reset-then-read-zeros, and don't re-aggregate already-summed state.

**Suggested test:** A small CPU-friendly (or GPU) test that runs two forward+backward passes through a model with a shared router and asserts that the `load_balancing_loss` metric reported in `compute_auxiliary_metrics` equals the sum of per-call lb-losses across all N shared layer applications, not just one. Currently nothing exercises this path.

## 6. `num_flops_per_token` audit under sharing

**Status (informal):** `Transformer.num_flops_per_token` iterates `self.blocks.values()` and sums each block's contribution. For a shared MoE module, `block.num_flops_per_token` is called once per referencing block — each call returns the same per-token FLOPs — so the total is correctly usage-weighted. The `num_param_uses` / `num_active_param_uses` properties were added so a researcher can write `flops ≈ 6 × num_active_param_uses × tokens` and get the right answer.

**Suggested check:** Add a test that builds a shared-blocks model and an equivalent unshared model with the same `n_layers`, and asserts `shared.num_flops_per_token(seq_len) == unshared.num_flops_per_token(seq_len)`. Should already hold; the test would lock it in.

## 7. Pipeline parallelism interaction

**What we haven't thought about:** `Transformer.apply_pp` and the various `_pp_*` state on the model. PP slices the block list across stages. If a shared range straddles a stage boundary, the same module would need to live on multiple stages — which doesn't work without explicit replication or weight sync.

**Suggested behavior:** In `apply_pp`, detect whether `shared_blocks.layer_indices()` is fully contained within a single PP stage's block range; raise `OLMoConfigurationError` otherwise. Document the constraint.

## 8. `apply_tp` and `apply_cp` are not dedup'd

**What:** `Transformer.apply_tp` (model.py:649-687) and `Transformer.apply_cp` (model.py:689-710) iterate `for block in self.blocks.values()` and call `block.apply_tp(...)` / `block.apply_cp(...)` per-block. Those block methods then call `parallelize_module(self.attention, ...)`, `parallelize_module(self.feed_forward_moe, ...)`, and `self.feed_forward_moe.apply_cp(...)` on the inner submodules. When the same `feed_forward_moe` is shared across N blocks, those calls run N times against the same module.

**Why it matters:** This is the exact same correctness gap that `apply_ep` and `apply_fsdp` had — wrapping the same module with `parallelize_module` twice corrupts TP state. Anyone enabling `shared_blocks` together with TP or CP will hit this. The fact that it doesn't fail in our existing CPU tests is just because we don't exercise TP/CP there.

**Suggested fix:** Mirror the `seen_moes: set[int]` dedup pattern used in `MoETransformer.apply_ep` (model.py:1062-1073) inside `Transformer.apply_tp` and `Transformer.apply_cp`. For TP: track which inner submodules have already been `parallelize_module`'d; skip them on subsequent blocks. For CP: similar dedup, also extend to shared attention if `share_attention=True`.

## 9. `init_weights` re-initializes shared submodules N times

**What:** `Transformer.init_weights` (model.py:295-374) iterates `for block in self.blocks.values()` and calls `self.init_method.init_attention(block.attention, block_idx=block.block_idx, num_blocks=self.n_layers, ...)`, `init_feed_forward_moe(...)`, etc. on each block. For a shared submodule, this runs N times against the same `Parameter`s with N different `block_idx` values — last write wins.

**Why it matters:** Many `InitMethod`s scale the init std by depth (`block_idx` / `num_blocks`), e.g. `std / sqrt(2 * (block_idx + 1))`. With sharing, the shared weights silently end up with the init produced for the *highest* `block_idx` in the shared range, not the canonical (start) one, and N−1 RNG draws are wasted. There's no warning and the result is a wrong-but-trains-OK model. Also: RoPE and attention-backend warmup-cache calls inside the loop run N times if `share_attention=True` (wasteful, not incorrect).

**Suggested fix:** Dedup the per-block init pass by `id(submodule)`. When the loop encounters a block whose `attention` or `feed_forward_moe` has already been init'd, skip the init call (but still warm up per-block caches that depend on block_idx, if any). Equivalently: build a set of canonical (block_idx, submodule) pairs up front and only init those.

## 10. `block_overrides` ∩ `shared_blocks` is unvalidated

**What:** `_apply_shared_blocks` (model.py) overwrites the non-canonical blocks' submodules with references to the canonical block's submodules. If a layer in `shared_blocks.layer_indices()` also has an entry in `block_overrides` that produces a structurally different block (e.g. different `num_experts_list`, different block type, MoE vs hybrid), the override gets silently clobbered.

**Why it matters:** No error is raised. In the lucky case shapes match and the user gets a model different from what they configured. In the unlucky case shapes differ and you get a runtime shape error far from the config site.

**Suggested fix:** In `SharedBlockConfig.validate_against` (or in `_apply_shared_blocks`), check that no `block_overrides` index lies in `layer_indices()`, OR that the override resolves to a config structurally identical to the canonical block. Raise `OLMoConfigurationError` with a clear message otherwise.

## 11. Test coverage gaps for `share_attention` / `share_norms` / `share_shared_mlp`

**What:** The five CPU tests in `src/test/nn/transformer/model_test.py` exercise `share_routers`, `share_experts`, both together, partial-sharing, validation, and the GPU-only forward/backward path. The remaining three flags — `share_attention`, `share_norms`, `share_shared_mlp` — have no direct test.

**Suggested fix:** Add three tests modeled on `test_shared_blocks_module_identity` that build a small model with each flag set in isolation and assert `id()` equality of the right submodule across the shared range. For `share_shared_mlp` use the hybrid block type (`moe_hybrid_reordered_norm`) so the dense `shared_mlp` actually exists.

## 12. Usage example for `SharedBlockConfig`

**What:** Beyond the per-field descriptions on the dataclass, there's no user-facing snippet showing how to enable shared blocks from a config or how to think about the `start_layer`/`n_layers` choice for a given model depth.

**Suggested fix:** Add a short example block (maybe 10-15 lines) to the `SharedBlockConfig` docstring in `src/olmo_core/nn/transformer/config.py`, and a paragraph + example subgrid pointer in `ml/README.md`. Cover the typical case (`share_routers=True, share_experts=True`, middle range of an N-layer model) plus a note that `n_layers` here is the *count of layers in the shared range*, not the total model depth (an easy thing to misread).
