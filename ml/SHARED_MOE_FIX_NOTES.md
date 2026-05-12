# Shared-MoE Fix Notes

Walkthroughs of each correctness fix landed for the `SharedBlockConfig` feature (physical parameter sharing across a contiguous range of transformer blocks). Each entry explains the bug in concrete terms, what the fix does, and any related concepts worth knowing.

**Companion files:**
- `ml/SHARED_MOE_TODOS.md` — backlog index of follow-ups.
- `ml/SHARED_MOE_TOP3_PLANS.md` — implementation plans for the highest-priority items.

---

## TODO #9 — `init_weights` per-block dedup

### What "sharing" actually does at the object level

When `shared_blocks` is enabled with `share_routers=True, share_experts=True`, this assignment runs in `_apply_shared_blocks`:

```python
model.blocks['2'].feed_forward_moe = model.blocks['1'].feed_forward_moe
model.blocks['3'].feed_forward_moe = model.blocks['1'].feed_forward_moe
model.blocks['4'].feed_forward_moe = model.blocks['1'].feed_forward_moe
```

All four blocks now hold references to the **same Python object**. The parameters live in memory exactly once; the four `block.feed_forward_moe` attributes are just four pointers to the same `MoEBase` instance. `id(model.blocks['2'].feed_forward_moe) == id(model.blocks['1'].feed_forward_moe)` is `True`.

### The bug

The original `init_weights` did this:

```python
for block in self.blocks.values():
    ...
    self.init_method.init_feed_forward_moe(
        block.feed_forward_moe,
        block_idx=block.block_idx,    # 1, then 2, then 3, then 4
        num_blocks=self.n_layers,
        std=self.init_std,
        generator=generator,
    )
```

It walks every block and calls init on its `feed_forward_moe`. Without sharing, that's correct — each block has its own submodule. **With sharing, the same parameters get initialized four times in a row.**

This is silently wrong under depth-scaled init methods. `InitMethod.llama_depth` computes:

```python
std = init_std / sqrt(2 * (block_idx + 1))
```

So for `init_std=0.02` and a shared range covering block_idx 1..4:

| iteration | block_idx | std it writes with    |
|-----------|-----------|-----------------------|
| 1         | 1         | 0.02 / √4  = 0.0100   |
| 2         | 2         | 0.02 / √6  ≈ 0.00816  |
| 3         | 3         | 0.02 / √8  ≈ 0.00707  |
| 4         | 4         | 0.02 / √10 ≈ 0.00632  |

Each call **overwrites** the previous draw with a fresh draw from a different distribution. The shared parameters end up with std ≈ 0.00632 — the std from the *last* block in the range. No error, no warning, just a quietly mis-initialized model. N−1 RNG draws are also wasted, which advances the generator state and changes everything initialized after (e.g., the LM head).

### What the fix does

Track which submodule objects have already been touched, keyed by Python `id()`:

```python
seen_moe: Set[int] = set()
for block in self.blocks.values():
    ...
    if hasattr(block, "feed_forward_moe") and id(block.feed_forward_moe) not in seen_moe:
        seen_moe.add(id(block.feed_forward_moe))
        self.init_method.init_feed_forward_moe(
            block.feed_forward_moe,
            block_idx=block.block_idx,
            ...
        )
```

`id(x)` returns the memory address of `x` as an int — stable and hashable, perfect for set membership. The first iteration that encounters a given `feed_forward_moe` records its id and runs init; subsequent iterations see the id already in the set and skip. Because `self.blocks` is a `ModuleDict` keyed `'0'`, `'1'`, `'2'`, ..., we iterate in numerical order, so the *lowest* block_idx in the shared range drives init — which matches the intuitive "this submodule lives at the start of the shared range" semantics.

There are four separate sets (`seen_attn`, `seen_ff`, `seen_moe`, `seen_warmup`) because the five `share_*` flags combine independently — a user might share routers/experts but not attention, in which case `feed_forward_moe` is shared across blocks but `attention` is unique per block. One set per submodule type keeps each gate clean.

### Aside: warmup caches

The same `init_weights` loop calls `*.warmup_cache(...)` on three things:

- **`att.backend.warmup_cache(max_seq_len, device)`** — The attention backend (torch SDPA, FlashAttention 2/3, etc.) sometimes allocates scratch tensors sized by `max_seq_len`. Pre-allocating avoids a mid-training stall.
- **`att.rope.warmup_cache(max_seq_len, device)`** — RoPE precomputes a `(max_seq_len, head_dim/2)` table of `cos(θ)`/`sin(θ)` values. Without warmup, the first forward pass would build it; with warmup, it's already there.
- **`block.feed_forward_moe.warmup_cache(max_local_microbatch_size)`** — The MoE dispatch path keeps scratch buffers for the token-to-expert routing step.

These are **lazily-allocated internal buffers** that get materialized on first use. Allocating them during the first forward pass would cause a memory spike and latency hiccup at an inconvenient time, so we materialize them up front during init.

Calling `warmup_cache(...)` twice is idempotent — the cache is already populated, you just recompute the same thing and overwrite with the same result. So unlike the init issue, deduplicating warmup is a cleanliness/efficiency thing, not a correctness bug. The dedup pattern was added anyway because the cost is one line per call and it's symmetric with the rest of the loop.

### Why this matters more for some init methods than others

Under `InitMethod.normal` or `InitMethod.normalized` (which don't use `block_idx`), the bug never produces a visibly wrong model — the std is the same regardless of which iteration "wins". You'd still waste RNG draws and silently shift the generator state, which would affect any subsequent init (LM head, etc.), but the shared params would still be drawn from a valid distribution.

Under `InitMethod.llama_depth` the bug is a real correctness issue: the model would train but with an init scale that's wrong-but-plausible, and you'd never know unless you specifically went looking. That's the worst kind of bug — it doesn't crash, it just makes your experiment slightly meaningless.

### Files touched

- `src/olmo_core/nn/transformer/model.py` — `Transformer.init_weights`, added the four `seen_*` sets and gated each init/warmup call.
- `src/test/nn/transformer/model_test.py` — `test_shared_blocks_init_uses_canonical_block_idx` (empirical-std check) and `test_shared_blocks_init_called_once_per_shared_submodule` (call-count check via monkey-patch).

---

## TODO #8 — `apply_tp` / `apply_cp` dedup

### What this is about

`apply_tp` (tensor parallelism) and `apply_cp` (context parallelism) are the per-model methods you call to slice the model across a TP/CP device mesh. They both work the same way: `Transformer.apply_tp(mesh, ...)` iterates `self.blocks.values()` and calls `block.apply_tp(...)` on each block. The block-level methods then recurse into inner submodules — attention, norms, the MoE — and either call `torch.distributed.tensor.parallel.parallelize_module(submodule, ...)` (which decorates parameters with DTensor placement metadata) or delegate to `submodule.apply_tp(...)` / `submodule.apply_cp(...)` on those submodules.

With `shared_blocks` enabled, multiple blocks point to the same `attention` / `attention_norm` / `feed_forward_norm` / `feed_forward_moe` instance. So the per-block loop ends up calling `parallelize_module` (or the inner `apply_tp`) on the **same Python object N times**. This is the same bug class we already fixed for `apply_ep` and `apply_fsdp` — the difference is that `apply_ep` got dedup'd as part of the original `SharedBlockConfig` work, but `apply_tp` and `apply_cp` were missed.

### Why double-parallelizing breaks things

`parallelize_module(submodule, ...)` rewrites the submodule's parameters as **DTensors** — wrapped tensors that carry placement metadata (sharded along which dim, on which device mesh, etc.). The second call sees parameters that are already DTensors and tries to re-wrap them, which either:
- raises somewhere deep in `torch.distributed.tensor.parallel` (best case),
- silently nests placements in a way that breaks the forward pass at the first all-gather (subtle case), or
- produces inconsistent placements across the N "copies" of the module reference (corruption case).

None of these are bugs you want to debug from a stack trace at training-time on a multi-GPU box.

### The shape of the fix

Mirror the `seen_moes: set[int]` pattern already used in `MoETransformer.apply_ep`, but with one twist: the block-level `apply_tp` does both **block-unique work** (it wraps `self` with `PrepareModuleInput`, `parallelize_module`s its own dropout, etc.) and **inner-submodule work** (calls `self.attention.apply_tp(...)`, `parallelize_module(self.attention_norm, ...)`, etc.). We only need to dedup the inner work — the block itself is always unique even when sharing.

So instead of skipping entire blocks, the fix plumbs a **shared `skip_modules: Set[int]`** through the per-block `apply_tp` and `apply_cp` calls. The model-level method constructs one set and passes it to every block; each block's method gates its inner calls on `id(submodule) not in skip_modules`, and adds the id to the set after running. The first block in iteration order parallelizes the shared inner submodule; subsequent blocks see it in the set and skip.

```python
# model.py — Transformer.apply_tp (sketch)
seen_tp_modules: Set[int] = set()
for block in self.blocks.values():
    block.apply_tp(
        tp_mesh,
        input_layout=Shard(1),
        float8_enabled=float8_enabled,
        skip_modules=seen_tp_modules,  # mutated by block.apply_tp
    )

# block.py — MoETransformerBlock.apply_tp (sketch)
def apply_tp(self, tp_mesh, *, input_layout, float8_enabled=False, skip_modules=None):
    skip_modules = skip_modules if skip_modules is not None else set()
    parallelize_module(self, ...)  # block itself — always unique
    if id(self.attention) not in skip_modules:
        skip_modules.add(id(self.attention))
        self.attention.apply_tp(tp_mesh, ...)
    if id(self.feed_forward_moe) not in skip_modules:
        skip_modules.add(id(self.feed_forward_moe))
        self.feed_forward_moe.apply_tp(tp_mesh, ...)
    ...
```

The signature gains a single optional `skip_modules` kwarg with a default of `None` (which means "treat each call as fresh", preserving the old behavior for callers that don't use sharing). Internally, `None` is normalized to a fresh empty set.

### Surface — which classes had to change

I had to update `apply_tp` / `apply_cp` on every class that overrides them:

- `TransformerBlockBase` (the abstract signatures) — added the kwarg with a docstring describing the contract.
- `TransformerBlock` — both methods. Gates `self.attention_norm`, `self.attention`, `self.feed_forward_norm`. The dense `self.feed_forward` is *not* gated because no `share_feed_forward` flag exists in `SharedBlockConfig` today (dense FFs are always block-unique).
- `PeriNormTransformerBlock` — extends `TransformerBlock`, calls `super().apply_tp(skip_modules=...)` then parallelizes its own `post_attention_norm` / `post_feed_forward_norm` (block-unique, no gating needed).
- `NormalizedTransformerBlock` — `apply_tp` raises `NotImplementedError`, but updated the signature for type consistency; `apply_cp` gates `self.attention`.
- `MoETransformerBlock` — both methods. Gates `self.attention_norm`, `self.attention`, `self.feed_forward_norm`, `self.feed_forward_moe`. The block's own `self.dropout` is unique, no gate.
- `MoEHybridTransformerBlockBase` — extends `MoETransformerBlock`, calls `super().apply_tp(skip_modules=...)` then handles its own dense `feed_forward` and `feed_forward_moe_norm` (block-unique, no gate).

`apply_cp` is structurally simpler than `apply_tp` because the block-level method only delegates inward (no block-unique `parallelize_module(self, ...)` wrap to do alongside). So the `skip_modules` plumbing there is just "if not in set, call apply_cp, add to set" with no other code around it.

### Why `apply_ep` didn't need the `skip_modules` plumbing trick

`apply_ep` only cares about one thing: `feed_forward_moe.apply_ep(ep_mesh, ...)`. It's a single inner call, with no block-level work alongside it. So the dedup can happen entirely at the model level: walk blocks, skip if you've seen the `feed_forward_moe` before. That's what the existing `MoETransformer.apply_ep` does, and there was no need to thread anything through the block API.

`apply_tp` and `apply_cp` are different because the block-level method does its own work *in addition to* recursing into inner submodules. We can't skip the whole block call — we have to let it do the block-unique work and selectively skip the inner submodule calls. That's why the new kwarg exists.

### What didn't need to change

- `parallelize_module(self, ...)` (the block itself) — block instances are always unique even with sharing. The block-level wrap is safe.
- `parallelize_module(self.dropout, ...)`, `self.attention_residual_stream.dropout`, etc. — these submodules aren't in the `SharedBlockConfig` surface, so they're always block-unique.
- `MoEHybridTransformerBlockBase`'s `feed_forward_moe_norm` and dense `feed_forward` — block-unique.
- `Transformer.apply_fsdp` — already handled separately by the FSDP fine-grained guard in the original `SharedBlockConfig` work.

### Files touched

- `src/olmo_core/nn/transformer/block.py` — added `skip_modules: Optional[Set[int]] = None` kwarg to the abstract `apply_tp` / `apply_cp` signatures; gated inner-submodule calls in `TransformerBlock`, `PeriNormTransformerBlock`, `NormalizedTransformerBlock`, `MoETransformerBlock`, `MoEHybridTransformerBlockBase`.
- `src/olmo_core/nn/transformer/model.py` — `Transformer.apply_tp` and `Transformer.apply_cp` build a `seen_*_modules: Set[int]` and thread it through each `block.apply_*` call.
- `src/test/nn/transformer/model_test.py` — `test_shared_blocks_apply_tp_dedup` and `test_shared_blocks_apply_cp_dedup`, both monkey-patching `parallelize_module` / `attention.apply_tp` / `feed_forward_moe.apply_tp` to record calls by `id()` and asserting each shared submodule is touched exactly once across all blocks.
