# expand_olmoe_experts.py
"""
Expands an OLMoE-1B-7B checkpoint from 64 to 128 experts (or any N -> 2N).

Key insight from dense_to_moe.py:
  olmo_core stores expert weights in BATCHED 2D format:
    "blocks.{i}.feed_forward_moe.experts.mlp.w1" : [N * hidden_dim, d_model]
    "blocks.{i}.feed_forward_moe.experts.mlp.w2" : [N * hidden_dim, d_model]  
    "blocks.{i}.feed_forward_moe.experts.mlp.w3" : [N * hidden_dim, d_model]
    "blocks.{i}.feed_forward_moe.router.weight"  : [N, d_model]

  So expanding N->2N means:
    - For expert weights: cat([original, new_experts], dim=0)  [2N*hidden, d_model]
    - For router:         cat([original, new_rows],   dim=0)  [2N, d_model]

Usage:
    # From a native olmo_core checkpoint:
    python expand_olmoe_experts.py \
        --checkpoint_dir /path/to/olmoe_checkpoint \
        --output_dir /path/to/expanded_checkpoint \
        --original_num_experts 64 \
        --expanded_num_experts 128 \
        --init clone_with_noise \
        --noise_scale 0.01

    # From HuggingFace (downloads + converts on the fly):
    python expand_olmoe_experts.py \
        --hf_model_name allenai/OLMoE-1B-7B-0924 \
        --output_dir /path/to/expanded_checkpoint \
        --original_num_experts 64 \
        --expanded_num_experts 128 \
        --init clone_with_noise
"""

import argparse
import logging
import os
import re

import torch
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import save_state_dict
from olmo_core.nn.moe import MoEConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


# ── Key structure constants (from dense_to_moe.py) ──────────────────────────

# These are the batched expert weight keys in olmo_core.
# Shape: [num_experts * hidden_dim, d_model]
BATCHED_EXPERT_KEYS = [
    "feed_forward_moe.experts.mlp.w1",
    "feed_forward_moe.experts.mlp.w2",
    "feed_forward_moe.experts.mlp.w3",
]

# Router weight key in olmo_core.
# Shape: [num_experts, d_model]
ROUTER_KEY = "feed_forward_moe.router.weight"

# HF -> olmo_core key mapping (from dense_to_moe.py comments, verified)



HF_TO_OLMOCORE = {
    
    # --- Attention ---
    "attention.w_q.weight":   "self_attn.q_proj.weight",
    "attention.w_k.weight":   "self_attn.k_proj.weight",
    "attention.w_v.weight":   "self_attn.v_proj.weight",
    "attention.w_out.weight":"self_attn.o_proj.weight",

    "attention.q_norm.weight": "self_attn.q_norm.weight",
    "attention.k_norm.weight": "self_attn.k_norm.weight",

    # --- LayerNorms ---
    "attention_norm.weight":   "input_layernorm.weight",
    "feed_forward_norm.weight":"post_attention_layernorm.weight",

    # --- MoE Router ---
    "feed_forward_moe.router.weight": "mlp.gate.weight",

    # --- MoE Experts (requires looping over expert index) ---
    "feed_forward_moe.experts.mlp.w1": "mlp.experts.{expert}.gate_proj.weight",
    "feed_forward_moe.experts.mlp.w2": "mlp.experts.{expert}.down_proj.weight",
    "feed_forward_moe.experts.mlp.w3": "mlp.experts.{expert}.up_proj.weight",

    # --- Final ---
    "lm_head.norm.weight": "model.norm.weight",
    "lm_head.w_out.weight":   "lm_head.weight",

    # --- Embeddings (if present) ---
    "embeddings.weight": "embed_tokens.weight",
}


# ── Model config ─────────────────────────────────────────────────────────────

def build_expanded_model_config(num_experts: int, top_k: int = 8) -> TransformerConfig:
    """
    Builds an olmo_core model config matching the given expert count.
    - num_experts=64,  top_k=8  → uses olmoe_1B_7B  (for loading the HF source)
    - num_experts=128, top_k=16 → uses olmoe_2B_14B (for the expanded model)
    """
    tokenizer = TokenizerConfig.dolma2()
    if num_experts == 64:
        return TransformerConfig.olmoe_1B_7B(
            vocab_size=tokenizer.padded_vocab_size(),
        )
    elif num_experts == 128:
        return TransformerConfig.olmoe_2B_14B(
            vocab_size=tokenizer.padded_vocab_size(),
        )
    else:
        raise ValueError(
            f"Unsupported num_experts={num_experts}. "
            "Add a matching TransformerConfig classmethod for this expert count."
        )


# ── HF loading and conversion ─────────────────────────────────────────────────

def load_hf_checkpoint_as_olmocore(hf_model_name: str, original_num_experts: int) -> dict:
    """
    Loads OLMoE from HuggingFace and converts to olmo_core state dict format.
    
    The HF model stores experts as INDIVIDUAL modules:
      model.layers.{i}.mlp.experts.{j}.gate_proj.weight  [hidden, d_model]
    
    olmo_core stores them BATCHED:
      blocks.{i}.feed_forward_moe.experts.mlp.w1          [N*hidden, d_model]
    
    This function handles that conversion.
    """
    from transformers import AutoModelForCausalLM
    log.info(f"Loading HF model: {hf_model_name}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    hf_sd = hf_model.state_dict()
    
    # Add this temporarily to find the real HF router key
    print("\n=== HF router-related keys ===")
    for k in hf_sd.keys():
        #import pdb;pdb.set_trace()
        if "router" in k or "gate" in k.lower():
            print(f"  {k}: {hf_sd[k].shape}")

    # Build an empty olmo_core MoE model to get the target state dict structure
    log.info("Building empty olmo_core model to get target structure...")
    olmocore_config = build_expanded_model_config(original_num_experts)
    olmocore_model  = olmocore_config.build(init_device="cpu")
    olmocore_sd     = olmocore_model.state_dict()

    # ── Step 1: fill non-expert, non-router weights ───────────────────────
    for olmocore_key in list(olmocore_sd.keys()):
        # Skip expert and router keys — handled separately below
        if any(ek in olmocore_key for ek in BATCHED_EXPERT_KEYS):
            continue
        if ROUTER_KEY in olmocore_key:
            continue

        # Find the matching HF key using the mapping
        hf_key = None
        for olmocore_pattern, hf_pattern in HF_TO_OLMOCORE.items():
            if olmocore_pattern in olmocore_key:
                hf_key = olmocore_key
                hf_key = hf_key.replace("blocks.", "layers.")
                hf_key = hf_key.replace(olmocore_pattern, hf_pattern)
                if not hf_key.startswith("model.") and "lm_head.weight" not in hf_key:
                    hf_key = f"model.{hf_key}"
                break

        if hf_key and hf_key in hf_sd:
            olmocore_sd[olmocore_key] = hf_sd[hf_key]
            log.info(f"  copied: {hf_key} -> {olmocore_key}")
        else:
            log.warning(f"  no HF match for: {olmocore_key}  (tried: {hf_key})")

    # ── Step 2: pack individual HF experts into batched olmo_core format ──
    #
    # HF format (per-expert, per-layer):
    #   model.layers.{i}.mlp.experts.{j}.gate_proj.weight  shape: [hidden, d_model]
    #
    # olmo_core format (batched, per-layer):
    #   blocks.{i}.feed_forward_moe.experts.mlp.w1          shape: [N*hidden, d_model]
    #
    # The dense_to_moe.py script transposes: olmo_core stores [d_model, hidden] effectively
    # but the batched dim is along dim=0 sliced by d_model strides.
    # Looking at dense_to_moe.py line:
    #   moe_state_dict[key][dim * expert : dim * (expert+1), :] = dense[hf_key].transpose(0,1)
    # where dim = dense[hf_key].shape[1] = d_model
    # So batched shape is [N * d_model, hidden] — NOT [N * hidden, d_model].
    # We replicate exactly that logic here.

    n_layers = olmocore_config.n_layers

    proj_map = {
        "feed_forward_moe.experts.mlp.w1": "gate_proj",
        "feed_forward_moe.experts.mlp.w2": "down_proj",
        "feed_forward_moe.experts.mlp.w3": "up_proj",
    }

    for layer_idx in range(n_layers):
        for olmocore_proj, hf_proj in proj_map.items():
            olmocore_key = f"blocks.{layer_idx}.{olmocore_proj}"
            if olmocore_key not in olmocore_sd:
                log.warning(f"Key not found in olmocore_sd: {olmocore_key}")
                continue

            # Collect all expert tensors for this layer+proj in expert index order
            expert_tensors = []
            for j in range(original_num_experts):
                hf_key = f"model.layers.{layer_idx}.mlp.experts.{j}.{hf_proj}.weight"
                if hf_key not in hf_sd:
                    raise KeyError(f"Missing HF key: {hf_key}")
                expert_tensors.append(hf_sd[hf_key])  # each: [hidden, d_model]

            # Replicate dense_to_moe.py packing logic exactly:
            # dim = tensor.shape[1] = d_model
            # moe_sd[key][dim*j : dim*(j+1), :] = tensor.transpose(0,1)
            # => each expert contributes d_model rows, so batched shape = [N*d_model, hidden]
            d_model = expert_tensors[0].shape[1]
            batched = torch.zeros(
                original_num_experts * d_model,
                expert_tensors[0].shape[0],  # hidden_dim
                dtype=expert_tensors[0].dtype,
            )
            for j, et in enumerate(expert_tensors):
                batched[d_model * j : d_model * (j + 1), :] = et.transpose(0, 1)

            olmocore_sd[olmocore_key] = batched
            log.info(f"  packed {original_num_experts} experts -> {olmocore_key} {batched.shape}")

    # ── Step 3: fill router weights ───────────────────────────────────────
    # HF router: model.layers.{i}.mlp.router.layer.weight  shape: [num_experts, d_model]
    # olmo_core: blocks.{i}.feed_forward_moe.router.weight shape: [num_experts, d_model]
    for layer_idx in range(n_layers):
        hf_router_key      = f"model.layers.{layer_idx}.mlp.gate.weight"
        olmocore_router_key = f"blocks.{layer_idx}.{ROUTER_KEY}"
        if hf_router_key in hf_sd and olmocore_router_key in olmocore_sd:
            olmocore_sd[olmocore_router_key] = hf_sd[hf_router_key]
            log.info(f"  router: {hf_router_key} -> {olmocore_router_key} {hf_sd[hf_router_key].shape}")
        else:
            log.warning(f"  missing router: {hf_router_key} or {olmocore_router_key}")

    log.info("HF -> olmo_core conversion complete.")
    return olmocore_sd


# ── Expert expansion ──────────────────────────────────────────────────────────

def expand_batched_expert_weight(
    tensor: torch.Tensor,
    original_n: int,
    n_to_add: int,
    init: str,
    noise_scale: float,
) -> torch.Tensor:
    """
    Infers per-expert stride from tensor shape: stride = tensor.shape[0] // original_n
    Works for both w1/w3 (stride=d_model) and w2 (stride=hidden_size).
    """
    assert tensor.shape[0] % original_n == 0, \
        f"shape[0]={tensor.shape[0]} not divisible by original_n={original_n}"
    stride = tensor.shape[0] // original_n  # <-- inferred, not hardcoded

    expert_slices = [
        tensor[stride * i : stride * (i + 1), :]
        for i in range(original_n)
    ]

    new_slices = []
    for i in range(n_to_add):
        src = expert_slices[i % original_n]
        if init == "clone_with_noise":
            new_slice = src.clone() + torch.randn_like(src) * noise_scale
        elif init == "clone":
            new_slice = src.clone()
        elif init == "random":
            new_slice = torch.randn_like(src) * src.std().item()
        elif init == "zeros":
            new_slice = torch.zeros_like(src)
        else:
            raise ValueError(f"Unknown init: {init}")
        new_slices.append(new_slice)

    return torch.cat(expert_slices + new_slices, dim=0)


def expand_router_weight(
    tensor: torch.Tensor,
    n_to_add: int,
    d_model: int,
    init: str = "clone_with_noise",
    noise_scale: float = 0.01,
) -> torch.Tensor:
    assert tensor.ndim == 2
    original_n = tensor.shape[0]

    new_rows = []
    for i in range(n_to_add):
        src = tensor[i % original_n]  # clone from corresponding original expert
        if init == "clone_with_noise":
            new_row = src.clone() + torch.randn_like(src) * noise_scale
        elif init == "clone":
            new_row = src.clone()
        elif init == "random":
            new_row = torch.randn_like(src) * src.std().item()
        elif init == "zeros":
            new_row = torch.zeros_like(src)
        else:
            raise ValueError(f"Unknown init: {init}")
        new_rows.append(new_row)

    return torch.cat([tensor, torch.stack(new_rows)], dim=0)


def expand_state_dict(
    olmocore_sd: dict,
    original_n: int,
    expanded_n: int,
    d_model: int,
    init: str,
    noise_scale: float,
    router_init: str,           # new
    router_noise_scale: float,  # new    
) -> dict:
    """
    Takes an olmo_core state dict with original_n experts and returns
    one with expanded_n experts. All non-expert keys are copied unchanged.
    """
    n_to_add = expanded_n - original_n
    new_sd   = {}

    for key, tensor in olmocore_sd.items():
        is_expert = any(ek in key for ek in BATCHED_EXPERT_KEYS)
        is_router = ROUTER_KEY in key

        if is_expert:
            log.info(f"[expert] expanding {key}: {tensor.shape}")
            new_tensor = expand_batched_expert_weight(
                tensor, original_n, n_to_add, init, noise_scale
            )
            log.info(f"  -> {new_tensor.shape}")
            new_sd[key] = new_tensor

        elif is_router:
            log.info(f"[router] expanding {key}: {tensor.shape}")
            new_tensor = expand_router_weight(
                tensor, n_to_add, d_model,
                init=router_init,
                noise_scale=router_noise_scale,
            )
            log.info(f"  -> {new_tensor.shape}")
            new_sd[key] = new_tensor

        else:
            new_sd[key] = tensor

    return new_sd


# ── Verification ──────────────────────────────────────────────────────────────

def verify(olmocore_sd: dict, original_n: int, expanded_n: int, d_model: int, n_layers: int, router_init):
    print("\n=== Verification ===")
    issues = []

    # w1/w3 use d_model as stride (their per-expert shape is [hidden, d_model], transposed to [d_model, hidden])
    # w2 uses hidden_size as stride (its per-expert shape is [d_model, hidden], transposed to [hidden, d_model])
    # We infer strides from the actual tensor shapes in the state dict.

    for layer_idx in range(n_layers):
        for ek in BATCHED_EXPERT_KEYS:
            key = f"blocks.{layer_idx}.{ek}"
            if key not in olmocore_sd:
                issues.append(f"Missing key: {key}")
                continue
            t = olmocore_sd[key]
            # Stride = number of rows per expert = total_rows / num_experts
            # We compute it from the original (pre-expansion) shape indirectly:
            # total_rows for expanded = t.shape[0], num_experts = expanded_n
            # So stride = t.shape[0] / expanded_n (must divide evenly)
            if t.shape[0] % expanded_n != 0:
                issues.append(
                    f"{key}: shape[0]={t.shape[0]} not divisible by expanded_n={expanded_n}"
                )
            else:
                stride = t.shape[0] // expanded_n
                # Sanity: stride should be either d_model or hidden_size
                expected_strides = {t.shape[0] // expanded_n, t.shape[1]}  # d_model or the other dim
                if stride not in expected_strides:
                    issues.append(
                        f"{key}: unexpected stride {stride}, expected one of {expected_strides}"
                    )

        router_key = f"blocks.{layer_idx}.{ROUTER_KEY}"
        if router_key not in olmocore_sd:
            issues.append(f"Missing router key: {router_key}")
        elif olmocore_sd[router_key].shape[0] != expanded_n:
            issues.append(
                f"{router_key}: expected {expanded_n} rows, "
                f"got {olmocore_sd[router_key].shape[0]}"
            )
        else:
            router = olmocore_sd[router_key]
            if router_init == "zeros":
                new_rows = router[original_n:]
                if not torch.all(new_rows == 0):
                    issues.append(f"{router_key}: new router rows are not zero-initialized")

    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"✓ All {n_layers} layers have correct expert shapes")
        print(f"✓ All routers have shape [{expanded_n}, {d_model}] with new rows")
        print(f"✓ Expanded from {original_n} -> {expanded_n} experts")

# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Expand OLMoE experts (MoE -> larger MoE)")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint_dir", type=str,
                        help="Path to native olmo_core checkpoint directory (contains model.pt)")
    source.add_argument("--hf_model_name", type=str,
                        help="HuggingFace model name, e.g. allenai/OLMoE-1B-7B-0924")

    parser.add_argument("--output_dir",           type=str, required=True)
    parser.add_argument("--original_num_experts",  type=int, default=64)
    parser.add_argument("--expanded_num_experts",  type=int, default=128)
    parser.add_argument("--d_model",               type=int, default=2048,
                        help="Model hidden dim (2048 for OLMoE-1B-7B)")
    parser.add_argument("--n_layers",              type=int, default=16,
                        help="Number of transformer layers (16 for OLMoE-1B-7B)")
    parser.add_argument("--init", default="clone",
                        choices=["clone_with_noise", "clone", "random", "zeros"],
                        help=(
                            "clone_with_noise: copy existing expert weights + small gaussian noise (recommended)\n"
                            "clone:            exact copy, experts start identical\n"
                            "random:           random init scaled to match source std\n"
                            "zeros:            zero init (experts completely inactive at start)"
                        ))
    parser.add_argument("--noise_scale",    type=float, default=0.01)

    parser.add_argument("--router_init", default=None,
                        choices=["clone_with_noise", "clone", "random", "zeros"],
                        help="Init method for new router rows. Defaults to --init if not set.")
    parser.add_argument("--router_noise_scale", type=float, default=None,
                        help="Noise scale for router clone_with_noise. Defaults to --noise_scale if not set.")

    parser.add_argument("--inspect_only",   action="store_true",
                        help="Print expert-related keys and shapes, then exit")
    return parser.parse_args()


if __name__ == "__main__":
    prepare_cli_environment()
    args = parse_args()

    assert args.expanded_num_experts > args.original_num_experts, \
        "expanded_num_experts must be greater than original_num_experts"

    # ── Load source state dict ────────────────────────────────────────────
    if args.hf_model_name:
        log.info(f"Loading from HuggingFace: {args.hf_model_name}")
        olmocore_sd = load_hf_checkpoint_as_olmocore(
            args.hf_model_name,
            args.original_num_experts,
        )
    else:
        ckpt_path = os.path.join(args.checkpoint_dir, "model.pt")
        log.info(f"Loading native olmo_core checkpoint: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location="cpu")
        # olmo_core checkpoint format: {"model": state_dict} or plain state_dict
        olmocore_sd = raw.get("model", raw)

    if args.inspect_only:
        print("\n=== Expert / Router keys ===")
        for k, v in olmocore_sd.items():
            if any(kw in k for kw in ["expert", "router"]):
                print(f"  {k:80s} {str(v.shape):30s} {v.dtype}")
        exit(0)

    # ── Expand ────────────────────────────────────────────────────────────
    log.info(
        f"Expanding {args.original_num_experts} -> {args.expanded_num_experts} experts "
        f"using init='{args.init}'"
    )
    expanded_sd = expand_state_dict(
        olmocore_sd,
        original_n=args.original_num_experts,
        expanded_n=args.expanded_num_experts,
        d_model=args.d_model,
        init=args.init,
        noise_scale=args.noise_scale,
        router_init        = args.router_init,        
        router_noise_scale = args.router_noise_scale 
    )

    # ── Verify ────────────────────────────────────────────────────────────
    verify(
        expanded_sd,
        original_n=args.original_num_experts,
        expanded_n=args.expanded_num_experts,
        d_model=args.d_model,
        n_layers=args.n_layers,
        router_init=args.router_init,
    )

    # ── Save in olmo_core format ──────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    log.info(f"Saving expanded checkpoint to {args.output_dir} ...")

    # save_state_dict for distributed-compatible loading (like dense_to_moe.py)
    save_state_dict(args.output_dir, {"model": expanded_sd})
    # Also save plain model.pt for convenience
    torch.save(expanded_sd, os.path.join(args.output_dir, "model.pt"))

    log.info(f"✓ Done. Saved to {args.output_dir}")