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
