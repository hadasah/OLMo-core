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
