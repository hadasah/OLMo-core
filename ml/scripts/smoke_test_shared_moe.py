#!/usr/bin/env python
"""
Pre-flight smoke test for shared-MoE training jobs.

Builds a small shared-blocks MoE model end-to-end via the same code path
``single_train_launch.py`` uses (the ``MODEL_CONFIG_LOOKUP`` factory + post-build
``SharedBlockConfig`` assignment), then validates the resulting structure:

- Shared submodules in the configured range alias the canonical block's instance.
- ``model.num_params`` (unique) is strictly smaller than an equivalent unshared
  model's, and equals the manual unique-parameter sum.
- ``model.num_param_uses`` (usage-weighted) matches the unshared model's
  ``num_params`` — i.e. the "FLOPs-equivalent" param count is preserved.
- ``state_dict()`` emits only canonical paths (no alias keys leak).
- W&B tag generation produces the expected ``sharedMoE`` + ``share[…]@N+M`` tags.

Runs entirely on CPU and ``init_device="meta"`` — no GPU, no data, no SLURM
required. Useful as a fast sanity check before submitting an actual training
run. ``python ml/scripts/smoke_test_shared_moe.py`` exits 0 on success and
prints a structured report.
"""

import sys
from pathlib import Path

# Allow running this file directly via `python ml/scripts/smoke_test_shared_moe.py`.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch  # noqa: E402

from olmo_core.data import TokenizerConfig  # noqa: E402
from olmo_core.nn.transformer.config import SharedBlockConfig  # noqa: E402

from single_train_launch import MODEL_CONFIG_LOOKUP, get_wandb_tags  # noqa: E402


def _build_shared(model_name: str, sb: SharedBlockConfig, *, dropless: bool = True):
    tok = TokenizerConfig.dolma2()
    model_config = MODEL_CONFIG_LOOKUP[model_name](
        vocab_size=tok.padded_vocab_size(),
        use_moe=True,
        num_experts_list=[8],
        hidden_multipliers_list=[0.25],
        router_top_ks_list=[2],
        moe_generalist_hidden_multiplier=0,
        dropless_moe=dropless,
        bias_gamma=None,
        z_loss_weight=0.001,
        lb_loss_weight=0.01,
    )
    model_config.shared_blocks = sb
    return model_config, model_config.build(init_device="meta")


def _build_unshared(model_name: str, *, dropless: bool = True):
    tok = TokenizerConfig.dolma2()
    model_config = MODEL_CONFIG_LOOKUP[model_name](
        vocab_size=tok.padded_vocab_size(),
        use_moe=True,
        num_experts_list=[8],
        hidden_multipliers_list=[0.25],
        router_top_ks_list=[2],
        moe_generalist_hidden_multiplier=0,
        dropless_moe=dropless,
        bias_gamma=None,
        z_loss_weight=0.001,
        lb_loss_weight=0.01,
    )
    return model_config, model_config.build(init_device="meta")


def main() -> int:
    print("== shared-MoE smoke test ==\n")
    model_name = "olmo2_ml_10M"
    sb = SharedBlockConfig(
        start_layer=0,
        n_layers=2,
        share_routers=True,
        share_experts=True,
    )

    print(f"Building shared model: {model_name}, shared_blocks={sb}")
    shared_config, shared_model = _build_shared(model_name, sb)
    unshared_config, unshared_model = _build_unshared(model_name)

    # --- 1) Module identity: shared blocks reference the same MoE instance.
    canonical = shared_model.blocks[str(sb.start_layer)].feed_forward_moe
    for idx in sb.layer_indices():
        if shared_model.blocks[str(idx)].feed_forward_moe is not canonical:
            print(f"FAIL: block {idx} feed_forward_moe is not the canonical instance")
            return 1
    print(f"  module identity:        OK  (blocks {list(sb.layer_indices())} share one feed_forward_moe)")

    # --- 2) Parameter counts.
    seen: set[int] = set()
    unique_param_sum = 0
    for p in shared_model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        unique_param_sum += p.numel()
    if shared_config.num_params != unique_param_sum:
        print(f"FAIL: config.num_params={shared_config.num_params}, "
              f"sum-by-id={unique_param_sum}")
        return 1
    if shared_model.num_params != unique_param_sum:
        print(f"FAIL: model.num_params={shared_model.num_params}, "
              f"sum-by-id={unique_param_sum}")
        return 1
    if shared_config.num_params >= unshared_config.num_params:
        print(f"FAIL: sharing didn't reduce param count: "
              f"shared={shared_config.num_params}, unshared={unshared_config.num_params}")
        return 1
    print(f"  num_params (unique):    OK  shared={shared_config.num_params:,} "
          f"unshared={unshared_config.num_params:,} "
          f"saving={unshared_config.num_params - shared_config.num_params:,}")

    # --- 3) num_param_uses matches unshared num_params.
    if shared_model.num_param_uses != unshared_model.num_params:
        print(f"FAIL: num_param_uses={shared_model.num_param_uses} != "
              f"unshared.num_params={unshared_model.num_params}")
        return 1
    print(f"  num_param_uses:         OK  shared.num_param_uses={shared_model.num_param_uses:,} "
          f"equals unshared.num_params")

    # --- 4) state_dict dedup: no alias keys leak.
    sd = shared_model.state_dict()
    for alias_idx in sb.layer_indices():
        if alias_idx == sb.start_layer:
            continue
        alias_prefix = f"blocks.{alias_idx}.feed_forward_moe."
        leaked = [k for k in sd if k.startswith(alias_prefix)]
        if leaked:
            print(f"FAIL: alias block {alias_idx} leaked {len(leaked)} keys in state_dict")
            return 1
    print(f"  state_dict dedup:       OK  ({len(sd)} keys, no aliases for shared MoE)")

    # --- 5) W&B tag generation.
    tags = get_wandb_tags(
        "smoke_test", model_name,
        moe_num_experts_list=[8],
        moe_generalist_hidden_multiplier=0,
        moe_type="dropless",
        shared_blocks=sb,
    )
    if "sharedMoE" not in tags:
        print(f"FAIL: 'sharedMoE' tag missing from {tags}")
        return 1
    range_tag = next((t for t in tags if t.startswith("share[")), None)
    if range_tag is None:
        print(f"FAIL: no share[…]@N+M tag in {tags}")
        return 1
    print(f"  W&B tags:               OK  {tags}")

    print("\nAll checks passed. Ready to launch a real training run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
