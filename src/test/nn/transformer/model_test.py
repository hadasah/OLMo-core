import logging
from dataclasses import replace
from test.nn.attention.attention_test import BF16_ATOL, BF16_RTOL
from typing import Optional, cast

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Shard, init_device_mesh

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.parallel import (
    DataParallelConfig,
    DataParallelType,
    build_world_mesh,
)
from olmo_core.distributed.utils import get_full_tensor, get_world_size
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    RingAttentionLoadBalancerType,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNorm, LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe import MoEConfig, MoERouterConfig, MoEType
from olmo_core.nn.rope import RoPEConfig
from olmo_core.nn.transformer import (
    MoEHybridTransformerBlockBase,
    MoEReorderedNormTransformerBlock,
    MoETransformer,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.nn.transformer.config import SharedBlockConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.testing import (
    BACKENDS,
    FLASH_2_MARKS,
    FLASH_3_MARKS,
    GPU_MARKS,
    TE_MARKS,
    requires_flash_attn_2,
    requires_multi_gpu,
    run_distributed_test,
)
from olmo_core.utils import get_default_device, seed_all

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cuda", id="cpu->cuda", marks=GPU_MARKS),
        pytest.param("cpu", "cpu", id="cpu->cpu"),
    ],
)
def test_small_llama2_builder_config(init_device, device):
    config = TransformerConfig.llama2_271M(vocab_size=50257)
    log.info(config)
    model = config.build(init_device=init_device)
    model.init_weights(device=torch.device(device))

    # Make sure num params estimate is correct.
    num_actual_params = 0
    for _, p in model.named_parameters():
        num_actual_params += p.numel()
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params

    for module in model.modules():
        # Make sure there are no biases anywhere and layer norm weights are all 1.
        if isinstance(module, (nn.Linear, LayerNorm)):
            assert module.bias is None
        if isinstance(module, LayerNorm):
            assert module.weight is not None
            assert (module.weight == 1).all()

    # Make sure block_idx is set correctly.
    assert model.blocks["0"].block_idx == 0
    assert model.blocks[str(len(model.blocks) - 1)].block_idx == len(model.blocks) - 1


def check_ngpt_matrices(model: nn.Module, d_model: int):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert module.bias is None

            w = module.weight
            if isinstance(w, DTensor):
                w = w.full_tensor()

            if w.shape[1] == d_model and "attention.w_out" not in name:
                pass
            elif w.shape[0] == d_model:
                w = w.transpose(0, 1)
            else:
                continue

            log.info(f"Checking norm for '{name}'")
            norm = torch.linalg.vector_norm(w, dim=1)
            torch.testing.assert_close(norm, torch.ones_like(norm))


@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cuda", id="cpu->cuda", marks=GPU_MARKS),
        pytest.param("cpu", "cpu", id="cpu->cpu"),
    ],
)
def test_small_ngpt_builder_config(init_device, device):
    config = TransformerConfig.ngpt_271M(vocab_size=50257)
    model = config.build(init_device=init_device)
    model.init_weights(device=torch.device(device))

    # Make sure num params estimate is correct.
    num_actual_params = 0
    for _, p in model.named_parameters():
        num_actual_params += p.numel()
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params

    # Make sure block_idx is set correctly.
    assert model.blocks["0"].block_idx == 0
    assert model.blocks[str(len(model.blocks) - 1)].block_idx == len(model.blocks) - 1

    # Make sure all weights are normalized in the embedding dimension.
    check_ngpt_matrices(model, config.d_model)


def run_ngpt_with_fsdp2():
    config = TransformerConfig.ngpt_271M(vocab_size=50257)
    model = config.build(
        init_device="meta",
    )
    model.apply_fsdp()
    model.init_weights(max_seq_len=1024, device=get_default_device())
    optim = torch.optim.Adam(model.parameters())

    # Take an optimizer step.
    model(input_ids=torch.randint(0, 50257, (2, 128))).sum().backward()
    optim.step()

    # Re-normalize weights.
    model.normalize_matrices()  # type: ignore

    # Check that the re-normalization was successful.
    check_ngpt_matrices(model, config.d_model)


@requires_multi_gpu
def test_ngpt_with_fsdp2():
    run_distributed_test(run_ngpt_with_fsdp2, backend="nccl", start_method="spawn")


def get_transformer_config(
    architecture: str,
    dtype: torch.dtype = torch.float32,
    attn_backend: Optional[AttentionBackendName] = None,
    swa: Optional[SlidingWindowAttentionConfig] = None,
) -> TransformerConfig:
    config: TransformerConfig
    if architecture == "olmo2":
        config = TransformerConfig.olmo2_190M(
            vocab_size=16_000,
            n_layers=2,
            fused_ops=False,
            attn_backend=attn_backend,
            dtype=DType.from_pt(dtype),
            sliding_window=swa,
        )
    elif architecture == "llama":
        config = TransformerConfig.llama2_271M(
            vocab_size=16_000,
            n_layers=2,
            fused_ops=False,
            attn_backend=attn_backend,
            dtype=DType.from_pt(dtype),
            sliding_window=swa,
        )
    else:
        raise NotImplementedError(architecture)

    return config


def get_transformer_inputs() -> torch.Tensor:
    return torch.arange(0, 128).unsqueeze(0)


def run_tensor_parallel_transformer(checkpoint_dir, outputs_path, architecture: str):
    device = get_default_device()
    config = get_transformer_config(architecture)
    input_ids = get_transformer_inputs().to(device)

    mesh = init_device_mesh(
        device.type,
        (get_world_size(),),
        mesh_dim_names=("tp",),
    )

    model = config.build()
    model.apply_tp(mesh["tp"])
    model.init_weights(device=device, max_seq_len=512)
    load_model_and_optim_state(checkpoint_dir, model)

    logits = model(input_ids=input_ids)

    loss = logits.sum()
    loss.backward()

    og_logits = torch.load(outputs_path, map_location=device)
    torch.testing.assert_close(og_logits, get_full_tensor(logits))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("architecture", ["olmo2", "llama"])
def test_tensor_parallel_transformer(backend: str, architecture: str, tmp_path):
    device = torch.device("cuda") if "nccl" in backend else torch.device("cpu")
    config = get_transformer_config(architecture)
    model = config.build()
    model.init_weights(device=device, max_seq_len=512)
    input_ids = get_transformer_inputs().to(device)
    logits = model(input_ids=input_ids)

    outputs_path = tmp_path / "logits.pt"
    torch.save(logits, outputs_path)

    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, model)

    run_distributed_test(
        run_tensor_parallel_transformer,
        backend=backend,
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            outputs_path,
            architecture,
        ),
    )


def run_context_parallel_transformer_ring(checkpoint_dir, outputs_path, architecture: str):
    device = get_default_device()
    config = get_transformer_config(
        architecture, dtype=torch.bfloat16, attn_backend=AttentionBackendName.flash_2
    )

    mesh = init_device_mesh(
        device.type,
        (get_world_size(),),
        mesh_dim_names=("cp",),
    )

    model = config.build()
    ring_style = RingContextParallelStyle(load_balancer=RingAttentionLoadBalancerType.zig_zag)
    model.apply_cp(mesh["cp"], ring=ring_style)
    model.init_weights(device=device, max_seq_len=512)
    load_model_and_optim_state(checkpoint_dir, model)

    input_ids = get_transformer_inputs().to(device)
    local_logits = model(input_ids=input_ids)
    logits = DTensor.from_local(local_logits, mesh, (Shard(1),))

    og_logits = torch.load(outputs_path, map_location=device)
    torch.testing.assert_close(og_logits, get_full_tensor(logits), rtol=BF16_RTOL, atol=BF16_ATOL)


@requires_multi_gpu
@requires_flash_attn_2
@pytest.mark.parametrize("architecture", ["olmo2"])
@pytest.mark.skip("known precision issues with ring-flash-attn")
def test_context_parallel_transformer_ring(architecture: str, tmp_path):
    seed_all(0)
    device = torch.device("cuda")
    config = get_transformer_config(
        architecture, dtype=torch.bfloat16, attn_backend=AttentionBackendName.flash_2
    )

    model = config.build()
    model.init_weights(device=device, max_seq_len=512)
    input_ids = get_transformer_inputs().to(device)
    logits = model(input_ids=input_ids)

    outputs_path = tmp_path / "logits.pt"
    torch.save(logits, outputs_path)

    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, model)

    run_distributed_test(
        run_context_parallel_transformer_ring,
        backend="nccl",
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            outputs_path,
            architecture,
        ),
    )


def run_context_parallel_transformer_ulysses(
    checkpoint_dir, outputs_path, architecture: str, backend_name: AttentionBackendName
):
    device = get_default_device()
    config = get_transformer_config(architecture, dtype=torch.bfloat16, attn_backend=backend_name)

    mesh = init_device_mesh(
        device.type,
        (get_world_size(),),
        mesh_dim_names=("cp",),
    )

    model = config.build()
    model.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())
    model.init_weights(device=device, max_seq_len=512)
    load_model_and_optim_state(checkpoint_dir, model)

    input_ids = get_transformer_inputs().to(device)
    local_logits = model(input_ids=input_ids)
    logits = DTensor.from_local(local_logits, mesh, (Shard(1),))

    og_logits = torch.load(outputs_path, map_location=device)
    tol_scale = 2.0  # requires slightly more tolerance than default
    torch.testing.assert_close(
        og_logits, get_full_tensor(logits), rtol=BF16_RTOL * tol_scale, atol=BF16_ATOL * tol_scale
    )


@requires_multi_gpu
@pytest.mark.parametrize("architecture", ["olmo2"])
@pytest.mark.parametrize(
    "backend_name",
    [
        pytest.param(AttentionBackendName.flash_2, id="flash-attn-2", marks=FLASH_2_MARKS),
        pytest.param(AttentionBackendName.flash_3, id="flash-attn-3", marks=FLASH_3_MARKS),
        pytest.param(AttentionBackendName.te, id="te-attn", marks=TE_MARKS),
    ],
)
def test_context_parallel_transformer_ulysses(
    architecture: str, backend_name: AttentionBackendName, tmp_path
):
    seed_all(0)
    device = torch.device("cuda")
    config = get_transformer_config(architecture, dtype=torch.bfloat16, attn_backend=backend_name)

    model = config.build()
    model.init_weights(device=device, max_seq_len=512)
    input_ids = get_transformer_inputs().to(device)
    logits = model(input_ids=input_ids)

    outputs_path = tmp_path / "logits.pt"
    torch.save(logits, outputs_path)

    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, model)

    run_distributed_test(
        run_context_parallel_transformer_ulysses,
        backend="nccl",
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            outputs_path,
            architecture,
            backend_name,
        ),
    )


def run_init_with_hsdp():
    assert dist.get_world_size() == 4
    mesh = build_world_mesh(
        dp=DataParallelConfig(name=DataParallelType.hsdp, shard_degree=2, num_replicas=2)
    )
    config = get_transformer_config("olmo2")
    model = config.build(init_device="meta")
    model.apply_fsdp(mesh)
    model.init_weights(max_seq_len=512, device=get_default_device())

    # Check that params across all replica groups are exactly the same.
    for name, param in model.named_parameters():
        full_param = get_full_tensor(param).detach()
        full_param_avg = full_param / 4
        dist.all_reduce(full_param_avg)
        torch.testing.assert_close(
            full_param_avg,
            full_param,
            msg=f"parameter '{name}' is inconsistent across the process group",
        )


@requires_multi_gpu
def test_init_with_hsdp():
    if torch.cuda.device_count() < 4:
        pytest.skip("Requires 4 GPUs")

    run_distributed_test(
        run_init_with_hsdp,
        backend="nccl",
        start_method="spawn",
        world_size=4,
    )


def run_moe_hybrid_combined_forward(
    dropless: bool, shared_experts: bool, reordered_norm: bool, tp: bool
):
    layer_norm = LayerNormConfig(name=LayerNormType.rms, bias=False)
    config = TransformerConfig(
        name=TransformerType.moe,
        d_model=512,
        vocab_size=16_000,
        n_layers=2,
        block=TransformerBlockConfig(
            name=(
                TransformerBlockType.moe_hybrid_reordered_norm
                if reordered_norm
                else TransformerBlockType.moe_hybrid
            ),
            attention=AttentionConfig(n_heads=8, rope=RoPEConfig(), qk_norm=layer_norm),
            layer_norm=layer_norm,
            feed_forward=FeedForwardConfig(hidden_size=1024, bias=False),
            feed_forward_moe=MoEConfig(
                name=MoEType.dropless if dropless else MoEType.default,
                num_experts_list=[4],
                hidden_sizes_list=[256],
                shared_mlp=(
                    FeedForwardConfig(hidden_size=512, bias=False) if shared_experts else None
                ),
                routers_list=[MoERouterConfig(uniform_expert_assignment=True)],
            ),
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False),
    )

    device = get_default_device()
    model = config.build(init_device=device.type)
    assert isinstance(model, MoETransformer)
    mesh = init_device_mesh(
        device.type,
        (get_world_size(),),
        mesh_dim_names=("tp" if tp else "ep",),
    )

    if tp:
        model.apply_tp(mesh["tp"])
    else:
        model.apply_ep(mesh["ep"])

    input_ids = get_transformer_inputs().to(device)
    model.init_weights(device=device, max_seq_len=512, max_local_microbatch_size=input_ids.numel())

    for block in model.blocks.values():
        cast(MoEHybridTransformerBlockBase, block).use_combined_forward = False
    output1 = model(input_ids)

    for block in model.blocks.values():
        cast(MoEHybridTransformerBlockBase, block).use_combined_forward = True
    output2 = model(input_ids)

    torch.testing.assert_close(output1, output2)


@requires_multi_gpu
@pytest.mark.parametrize(
    "dropless", [pytest.param(True, id="dropless"), pytest.param(False, id="default-router")]
)
@pytest.mark.parametrize(
    "shared_experts", [pytest.param(True, id="shared-experts"), pytest.param(False, id="no-shared")]
)
@pytest.mark.parametrize(
    "reordered_norm",
    [pytest.param(True, id="reordered-norm"), pytest.param(False, id="default-block")],
)
@pytest.mark.parametrize("tp", [pytest.param(True, id="TP"), pytest.param(False, id="EP")])
def test_moe_hybrid_combined_forward(
    dropless: bool, shared_experts: bool, reordered_norm: bool, tp: bool
):
    run_distributed_test(
        run_moe_hybrid_combined_forward,
        backend="nccl",
        start_method="spawn",
        func_args=(
            dropless,
            shared_experts,
            reordered_norm,
            tp,
        ),
    )


def test_build_with_block_overrides():
    d_model = 512
    config = TransformerConfig.llama_like_moe(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=8,
        n_heads=8,
        num_experts=32,
        top_k=4,
        expert_hidden_size=int(0.5 * d_model),
        capacity_factor=1.2,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        reordered_norm=True,
        hybrid=True,
        qk_norm=True,
        rope_theta=10_000,
        layer_norm_eps=1e-6,
        feed_forward=FeedForwardConfig(hidden_size=d_model * 2, bias=False),
    )
    assert config.block.feed_forward_moe is not None
    moe_config = replace(config.block.feed_forward_moe, shared_mlp=config.block.feed_forward)
    config.block_overrides = {
        0: replace(
            config.block,
            name=TransformerBlockType(str(config.block.name).replace("_hybrid_", "_")),
            feed_forward=None,
            feed_forward_moe=moe_config,
        )
    }

    model = config.build(init_device="cpu")
    assert isinstance(model.blocks["0"], MoEReorderedNormTransformerBlock)
    assert isinstance(model.blocks["1"], MoEHybridTransformerBlockBase)

    assert config.num_params == model.num_params


def test_transformer_num_flops_per_token():
    seed_all(0)

    d_model = 128
    seq_len = 256
    n_heads = 8
    n_kv_heads = 4
    vocab_size = 1024

    def _flops_per_token(*, n_layers: int, swa_pattern: list[int]) -> int:
        config = TransformerConfig.llama_like(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            sliding_window=SlidingWindowAttentionConfig(pattern=swa_pattern),
        )
        model = config.build(init_device="cpu")
        return model.num_flops_per_token(seq_len)

    base = _flops_per_token(n_layers=4, swa_pattern=[16, 16, 16, 16])
    assert base > 0

    # adding layers should strictly increase FLOPs/token.
    more_blocks = _flops_per_token(n_layers=8, swa_pattern=[16, 16, 16, 16])
    assert more_blocks > base

    # Relative checks: increasing sliding window size should increase FLOPs/token.
    bigger_window = _flops_per_token(n_layers=4, swa_pattern=[128, 128, 128, 128])
    assert bigger_window > base


@pytest.mark.parametrize(
    "config_builder,expected_d_model",
    [
        pytest.param(TransformerConfig.gemma3_1B, 2304, id="gemma3_1B"),
        pytest.param(TransformerConfig.gemma3_4B, 2560, id="gemma3_4B"),
        pytest.param(TransformerConfig.gemma3_12B, 3840, id="gemma3_12B"),
        pytest.param(TransformerConfig.gemma3_27B, 5376, id="gemma3_27B"),
    ],
)
def test_gemma3_builder_configs(config_builder, expected_d_model):
    config = config_builder(n_layers=2)
    assert config.d_model == expected_d_model
    assert config.n_layers == 2

    assert config.block.feed_forward is not None
    assert config.block.feed_forward.activation == ActivationFunction.gelu_tanh

    sequence_mixer = config.block.sequence_mixer
    assert isinstance(sequence_mixer, AttentionConfig)
    assert sequence_mixer.qk_norm is not None
    assert sequence_mixer.rope is not None
    assert sequence_mixer.rope.theta == 10_000

    # Use meta device to avoid allocating large amounts of memory for big models.
    model = config.build(init_device="meta")

    num_actual_params = sum(p.numel() for p in model.parameters())
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params


def test_gemma3_block_overrides_rope_theta():
    config = TransformerConfig.gemma3_1B(n_layers=12)

    assert config.block_overrides is not None

    local_count = 0
    global_count = 0
    for layer_idx in range(config.n_layers):
        if layer_idx in config.block_overrides:
            global_block = config.block_overrides[layer_idx]
            attention = global_block.sequence_mixer
            assert isinstance(attention, AttentionConfig)
            assert attention.rope is not None
            assert attention.rope.theta == 1_000_000
            assert attention.sliding_window is None
            global_count += 1
        else:
            attention = config.block.sequence_mixer
            assert isinstance(attention, AttentionConfig)
            assert attention.rope is not None
            assert attention.rope.theta == 10_000
            local_count += 1

    assert global_count == 2
    assert local_count == 10


def test_gemma3_sliding_window_pattern():
    config = TransformerConfig.gemma3_1B(n_layers=12)

    attention = config.block.sequence_mixer
    assert isinstance(attention, AttentionConfig)

    swa = attention.sliding_window
    assert swa is not None
    assert swa.pattern == [1024, 1024, 1024, 1024, 1024, -1]
    assert swa.force_full_attention_on_first_layer is False
    assert swa.force_full_attention_on_last_layer is False


@pytest.mark.parametrize(
    "config_builder,expected_d_model",
    [
        pytest.param(TransformerConfig.qwen3_0_6B, 1024, id="qwen3_0_6B"),
        pytest.param(TransformerConfig.qwen3_1_7B, 2048, id="qwen3_1_7B"),
        pytest.param(TransformerConfig.qwen3_4B, 2560, id="qwen3_4B"),
        pytest.param(TransformerConfig.qwen3_8B, 4096, id="qwen3_8B"),
        pytest.param(TransformerConfig.qwen3_14B, 5120, id="qwen3_14B"),
        pytest.param(TransformerConfig.qwen3_32B, 5120, id="qwen3_32B"),
    ],
)
def test_qwen3_builder_configs(config_builder, expected_d_model):
    config = config_builder(vocab_size=151936, n_layers=2)
    assert config.d_model == expected_d_model
    assert config.n_layers == 2
    attention = config.block.sequence_mixer
    assert isinstance(attention, AttentionConfig)
    assert attention.n_kv_heads == 8
    assert attention.rope is not None
    assert attention.rope.theta == 1_000_000

    # Use meta device to avoid allocating large amounts of memory for big models.
    model = config.build(init_device="meta")

    num_actual_params = sum(p.numel() for p in model.parameters())
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params


def _small_shared_moe_config(
    *,
    n_layers: int,
    shared_blocks: Optional[SharedBlockConfig],
    d_model: int = 64,
) -> TransformerConfig:
    layer_norm = LayerNormConfig(name=LayerNormType.rms, bias=False)
    block = TransformerBlockConfig(
        name=TransformerBlockType.moe_reordered_norm,
        sequence_mixer=AttentionConfig(
            n_heads=4,
            rope=RoPEConfig(),
            bias=False,
        ),
        layer_norm=layer_norm,
        feed_forward_moe=MoEConfig(
            name=MoEType.dropless,
            num_experts_list=[4],
            hidden_sizes_list=[d_model],
            routers_list=[MoERouterConfig(top_k=2)],
            lb_loss_weight=0.01,
            z_loss_weight=0.001,
        ),
    )
    return TransformerConfig(
        name=TransformerType.moe,
        d_model=d_model,
        vocab_size=256,
        n_layers=n_layers,
        block=block,
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False),
        shared_blocks=shared_blocks,
    )


def test_shared_blocks_module_identity():
    n_layers = 6
    sb = SharedBlockConfig(
        start_layer=1, n_layers=4, share_routers=True, share_experts=True
    )
    config = _small_shared_moe_config(n_layers=n_layers, shared_blocks=sb)
    model = config.build(init_device="cpu")
    assert isinstance(model, MoETransformer)

    canonical_moe = model.blocks[str(sb.start_layer)].feed_forward_moe
    for idx in sb.layer_indices():
        assert model.blocks[str(idx)].feed_forward_moe is canonical_moe, (
            f"block {idx} should share feed_forward_moe with block {sb.start_layer}"
        )

    for idx in range(n_layers):
        if idx in sb.layer_indices():
            continue
        assert model.blocks[str(idx)].feed_forward_moe is not canonical_moe, (
            f"block {idx} should not share feed_forward_moe (outside shared range)"
        )


def test_shared_blocks_param_count():
    sb = SharedBlockConfig(
        start_layer=1, n_layers=4, share_routers=True, share_experts=True
    )
    config = _small_shared_moe_config(n_layers=6, shared_blocks=sb)
    model = config.build(init_device="cpu")

    seen_ids: set[int] = set()
    actual_params = 0
    for p in model.parameters():
        if id(p) in seen_ids:
            continue
        seen_ids.add(id(p))
        actual_params += p.numel()

    assert config.num_params == actual_params
    assert model.num_params == actual_params

    unshared_config = _small_shared_moe_config(n_layers=6, shared_blocks=None)
    assert config.num_params < unshared_config.num_params


def test_shared_blocks_partial_routers_only():
    sb = SharedBlockConfig(
        start_layer=0, n_layers=3, share_routers=True, share_experts=False
    )
    config = _small_shared_moe_config(n_layers=4, shared_blocks=sb)
    model = config.build(init_device="cpu")

    canonical_block = model.blocks["0"]
    for idx in (1, 2):
        block = model.blocks[str(idx)]
        assert block.feed_forward_moe.routers_list is canonical_block.feed_forward_moe.routers_list
        assert block.feed_forward_moe.experts_list is not canonical_block.feed_forward_moe.experts_list


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MoE kernels require CUDA")
def test_shared_blocks_forward_backward_grad_accumulates():
    sb = SharedBlockConfig(
        start_layer=1, n_layers=3, share_routers=True, share_experts=True
    )
    config = _small_shared_moe_config(n_layers=5, shared_blocks=sb)
    device = torch.device("cuda")
    model = config.build(init_device=device.type)
    model.init_weights(device=device, max_seq_len=32)

    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
    logits = model(input_ids=input_ids)
    logits.sum().backward()

    shared_router = model.blocks[str(sb.start_layer)].feed_forward_moe.routers_list[0]
    assert shared_router.weight.grad is not None
    assert torch.isfinite(shared_router.weight.grad).all()
    assert shared_router.weight.grad.abs().sum() > 0


def test_shared_blocks_num_param_uses():
    sb = SharedBlockConfig(
        start_layer=1, n_layers=4, share_routers=True, share_experts=True
    )
    config = _small_shared_moe_config(n_layers=6, shared_blocks=sb)
    model = config.build(init_device="cpu")

    # Usage-weighted count should match a manual sum walking every (module, param) pair.
    expected_uses = sum(
        p.numel() for _, p in model.named_parameters(remove_duplicate=False)
    )
    assert model.num_param_uses == expected_uses
    assert config.num_param_uses == expected_uses

    # With sharing, uses > unique-params; without sharing they're equal.
    assert config.num_param_uses > config.num_params
    assert model.num_param_uses > model.num_params

    unshared_config = _small_shared_moe_config(n_layers=6, shared_blocks=None)
    unshared_model = unshared_config.build(init_device="cpu")
    assert unshared_config.num_param_uses == unshared_config.num_params
    assert unshared_model.num_param_uses == unshared_model.num_params

    # And the shared model's uses should match the unshared model's total params
    # (since sharing only changes physical storage, not the per-layer parameter "usage").
    assert config.num_param_uses == unshared_config.num_params
    assert model.num_param_uses == unshared_model.num_params


def test_shared_blocks_init_uses_canonical_block_idx():
    """
    Under InitMethod.llama_depth the init std scales as 1/sqrt(2*(block_idx+1)). If
    init_weights walks every block and re-initializes shared submodules, the highest
    block_idx in the shared range silently wins. After the dedup fix in init_weights,
    the canonical (lowest) block_idx should drive the init.
    """
    import math

    init_std = 0.02
    start_layer = 1
    n_shared = 3  # shared range: [1, 2, 3]
    sb = SharedBlockConfig(
        start_layer=start_layer,
        n_layers=n_shared,
        share_routers=True,
        share_experts=True,
    )
    config = _small_shared_moe_config(n_layers=5, shared_blocks=sb)
    config.init_method = InitMethod.llama_depth
    config.init_std = init_std

    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))

    # Inspect the shared expert weight matrix (large enough for tight sample-std).
    shared_w1 = model.blocks[str(start_layer)].feed_forward_moe.experts_list[0].mlp.w1
    empirical_std = shared_w1.detach().float().std().item()

    # llama_depth: std = init_std / sqrt(2*(block_idx+1)).
    canonical_std = init_std / math.sqrt(2 * (start_layer + 1))
    last_in_range_std = init_std / math.sqrt(2 * (start_layer + n_shared - 1 + 1))
    # trunc_normal_ at ±3*std has variance ~0.991*std^2, so empirical std is close
    # to but slightly below the analytical std. Compare on relative error.
    assert abs(empirical_std - canonical_std) / canonical_std < 0.05, (
        f"empirical_std={empirical_std:.6f} should be near canonical_std={canonical_std:.6f}, "
        f"NOT last-in-range_std={last_in_range_std:.6f}"
    )
    # Sanity: the two candidates are far enough apart that this comparison is meaningful.
    assert abs(empirical_std - last_in_range_std) / last_in_range_std > 0.1


def test_shared_blocks_init_called_once_per_shared_submodule():
    """
    Directly count how many times init_method.init_feed_forward_moe runs on the shared
    submodule. Should be exactly 1, not n_shared.
    """
    sb = SharedBlockConfig(
        start_layer=1, n_layers=4, share_routers=True, share_experts=True
    )
    config = _small_shared_moe_config(n_layers=6, shared_blocks=sb)
    config.init_method = InitMethod.llama_depth

    model = config.build(init_device="cpu")

    call_log: list[tuple[int, int]] = []  # (block_idx, id(submodule))
    original = type(model.init_method).init_feed_forward_moe

    def recording(self, m, *, block_idx, **kw):
        call_log.append((block_idx, id(m)))
        return original(self, m, block_idx=block_idx, **kw)

    type(model.init_method).init_feed_forward_moe = recording  # type: ignore[assignment]
    try:
        model.init_weights(device=torch.device("cpu"))
    finally:
        type(model.init_method).init_feed_forward_moe = original  # type: ignore[assignment]

    shared_id = id(model.blocks[str(sb.start_layer)].feed_forward_moe)
    shared_calls = [(idx, mid) for idx, mid in call_log if mid == shared_id]
    assert len(shared_calls) == 1, (
        f"shared MoE should be inited exactly once, got {len(shared_calls)} calls: {shared_calls}"
    )
    assert shared_calls[0][0] == sb.start_layer, (
        f"canonical (lowest) block_idx={sb.start_layer} should drive init, "
        f"got block_idx={shared_calls[0][0]}"
    )


def test_shared_blocks_apply_tp_dedup():
    """
    With shared_blocks, inner submodules (attention, norms, feed_forward_moe) are
    aliased across multiple blocks. parallelize_module/apply_tp must only run once
    per unique submodule — calling it twice corrupts DTensor placement state.
    Patch out the actual TP calls so we can run on CPU and count invocations by id().
    """
    import unittest.mock as mock
    import olmo_core.nn.transformer.block as block_mod
    import olmo_core.nn.transformer.model as model_mod

    sb = SharedBlockConfig(
        start_layer=1,
        n_layers=3,
        share_routers=True,
        share_experts=True,
        share_attention=True,
        share_norms=True,
    )
    config = _small_shared_moe_config(n_layers=5, shared_blocks=sb)
    model = config.build(init_device="cpu")

    parallelize_targets: list[int] = []
    inner_apply_tp_targets: list[int] = []

    def record_parallelize(module, **kw):  # noqa: ARG001
        parallelize_targets.append(id(module))
        return module

    def record_apply_tp(self, *_a, **_kw):
        inner_apply_tp_targets.append(id(self))

    # Capture id()s before patching (after patching, the model still references the
    # same objects, but we want to record what the live model points to right now).
    shared_attn_id = id(model.blocks[str(sb.start_layer)].attention)
    shared_moe_id = id(model.blocks[str(sb.start_layer)].feed_forward_moe)
    shared_attn_norm_id = id(model.blocks[str(sb.start_layer)].attention_norm)
    shared_ff_norm_id = id(model.blocks[str(sb.start_layer)].feed_forward_norm)

    attention_cls = type(model.blocks["0"].attention)
    moe_cls = type(model.blocks["0"].feed_forward_moe)
    lm_head_cls = type(model.lm_head)

    with mock.patch.object(block_mod, "parallelize_module", side_effect=record_parallelize), \
         mock.patch.object(model_mod, "parallelize_module", side_effect=record_parallelize), \
         mock.patch.object(attention_cls, "apply_tp", record_apply_tp), \
         mock.patch.object(moe_cls, "apply_tp", record_apply_tp), \
         mock.patch.object(lm_head_cls, "apply_tp", lambda *a, **kw: None):
        fake_mesh = mock.MagicMock()
        model.apply_tp(fake_mesh)

    # Each shared submodule should be touched exactly once across all blocks.
    assert inner_apply_tp_targets.count(shared_attn_id) == 1, (
        f"shared attention parallelized {inner_apply_tp_targets.count(shared_attn_id)} times "
        f"(expected 1)"
    )
    assert inner_apply_tp_targets.count(shared_moe_id) == 1
    assert parallelize_targets.count(shared_attn_norm_id) == 1
    assert parallelize_targets.count(shared_ff_norm_id) == 1

    # Sanity: the n_layers non-shared blocks (block 0 and block 4) should each have
    # their own unique attention/norm — those get parallelized exactly once too.
    non_shared_attn_id = id(model.blocks["0"].attention)
    assert non_shared_attn_id != shared_attn_id
    assert inner_apply_tp_targets.count(non_shared_attn_id) == 1


def test_shared_blocks_apply_cp_dedup():
    """
    Same idea as apply_tp dedup but for context parallelism: shared attention /
    feed_forward_moe should have apply_cp invoked exactly once.
    """
    import unittest.mock as mock

    sb = SharedBlockConfig(
        start_layer=1,
        n_layers=3,
        share_routers=True,
        share_experts=True,
        share_attention=True,
    )
    config = _small_shared_moe_config(n_layers=5, shared_blocks=sb)
    model = config.build(init_device="cpu")

    inner_apply_cp_targets: list[int] = []

    def record_apply_cp(self, *_a, **_kw):
        inner_apply_cp_targets.append(id(self))

    shared_attn_id = id(model.blocks[str(sb.start_layer)].attention)
    shared_moe_id = id(model.blocks[str(sb.start_layer)].feed_forward_moe)

    attention_cls = type(model.blocks["0"].attention)
    moe_cls = type(model.blocks["0"].feed_forward_moe)
    lm_head_cls = type(model.lm_head)

    with mock.patch.object(attention_cls, "apply_cp", record_apply_cp), \
         mock.patch.object(moe_cls, "apply_cp", record_apply_cp), \
         mock.patch.object(lm_head_cls, "apply_cp", lambda *a, **kw: None):
        fake_mesh = mock.MagicMock()
        model.apply_cp(fake_mesh)

    assert inner_apply_cp_targets.count(shared_attn_id) == 1
    assert inner_apply_cp_targets.count(shared_moe_id) == 1

    non_shared_attn_id = id(model.blocks["0"].attention)
    assert non_shared_attn_id != shared_attn_id
    assert inner_apply_cp_targets.count(non_shared_attn_id) == 1


def test_shared_blocks_state_dict_dedups_alias_paths():
    """
    state_dict() should emit only the canonical (lowest-block-index) path for each
    Parameter aliased via shared_blocks, not N copies. The total key count drops
    by (n_layers_shared - 1) * (number of submodule paths shared per layer).
    """
    sb = SharedBlockConfig(
        start_layer=1, n_layers=4, share_routers=True, share_experts=True
    )
    config = _small_shared_moe_config(n_layers=6, shared_blocks=sb)
    model = config.build(init_device="cpu")

    sd = model.state_dict()

    # No keys for the aliased blocks' MoE submodule should remain.
    for alias_idx in (2, 3, 4):
        alias_prefix = f"blocks.{alias_idx}.feed_forward_moe."
        leaked = [k for k in sd if k.startswith(alias_prefix)]
        assert not leaked, (
            f"alias block {alias_idx} should have no feed_forward_moe keys in state_dict, "
            f"got: {leaked[:3]}..."
        )

    # The canonical block's MoE keys MUST be present.
    canonical_prefix = f"blocks.{sb.start_layer}.feed_forward_moe."
    canonical_keys = [k for k in sd if k.startswith(canonical_prefix)]
    assert canonical_keys, "canonical block's MoE keys should be present in state_dict"

    # Sanity: an unshared model has all the alias keys.
    unshared_config = _small_shared_moe_config(n_layers=6, shared_blocks=None)
    unshared_sd = unshared_config.build(init_device="cpu").state_dict()
    assert any(k.startswith("blocks.2.feed_forward_moe.") for k in unshared_sd)
    assert len(sd) < len(unshared_sd)


def test_shared_blocks_state_dict_roundtrip():
    """
    Save a shared-blocks model's state_dict, load it back into a fresh shared-blocks
    model with the same config, and assert weights are bitwise identical.
    """
    sb = SharedBlockConfig(
        start_layer=1, n_layers=3, share_routers=True, share_experts=True
    )
    config = _small_shared_moe_config(n_layers=5, shared_blocks=sb)
    model_a = config.build(init_device="cpu")
    model_a.init_weights(device=torch.device("cpu"))

    sd = model_a.state_dict()
    # Fresh model with same config.
    model_b = config.build(init_device="cpu")
    model_b.init_weights(device=torch.device("cpu"))

    result = model_b.load_state_dict(sd, strict=True)
    assert not result.missing_keys, f"unexpected missing keys: {result.missing_keys[:5]}"
    assert not result.unexpected_keys, f"unexpected extra keys: {result.unexpected_keys[:5]}"

    # All parameters identical after round-trip.
    for (name, pa), (_, pb) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert torch.equal(pa, pb), f"mismatch at {name}"


def test_shared_blocks_load_raises_on_disagreeing_unshared_source():
    """
    Loading an unshared checkpoint into a shared-blocks model is ambiguous: the
    source has distinct per-layer weights for what's now a single Parameter. We
    should raise with a clear message rather than silently picking last-write-wins.
    """
    sb = SharedBlockConfig(
        start_layer=1, n_layers=3, share_routers=True, share_experts=True
    )
    # Source: unshared model with the same overall shape.
    unshared_config = _small_shared_moe_config(n_layers=5, shared_blocks=None)
    source_model = unshared_config.build(init_device="cpu")
    source_model.init_weights(device=torch.device("cpu"))
    source_sd = source_model.state_dict()

    # Target: shared-blocks model.
    target_config = _small_shared_moe_config(n_layers=5, shared_blocks=sb)
    target_model = target_config.build(init_device="cpu")
    target_model.init_weights(device=torch.device("cpu"))

    # The source has distinct per-layer values for blocks 1, 2, 3's MoE (because it
    # was init'd as an unshared model with depth-scaled init independent per layer).
    # Target's load_state_dict should detect the disagreement and raise.
    with pytest.raises(RuntimeError, match="differing per-layer values"):
        target_model.load_state_dict(source_sd)


def test_shared_blocks_load_accepts_agreeing_alias_keys():
    """
    If the source state_dict has multiple alias keys but their values agree (e.g.
    saved before the dedup fix landed), loading should succeed without raising.
    """
    sb = SharedBlockConfig(
        start_layer=1, n_layers=3, share_routers=True, share_experts=True
    )
    config = _small_shared_moe_config(n_layers=5, shared_blocks=sb)
    model_a = config.build(init_device="cpu")
    model_a.init_weights(device=torch.device("cpu"))

    # Produce a pre-dedup state_dict by walking with remove_duplicate=False.
    legacy_sd = {
        name: p.detach().clone()
        for name, p in model_a.named_parameters(remove_duplicate=False)
    }
    legacy_sd.update(
        {name: b.detach().clone() for name, b in model_a.named_buffers(remove_duplicate=False)}
    )

    model_b = config.build(init_device="cpu")
    model_b.init_weights(device=torch.device("cpu"))
    # Should not raise — all alias values agree.
    model_b.load_state_dict(legacy_sd, strict=True)

    for (name, pa), (_, pb) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert torch.equal(pa, pb), f"mismatch at {name}"


def test_shared_blocks_none_state_dict_is_passthrough():
    """
    When shared_blocks is None, state_dict and load_state_dict should behave
    identically to base nn.Module behavior.
    """
    config = _small_shared_moe_config(n_layers=4, shared_blocks=None)
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))

    sd = model.state_dict()
    # No dedup happened — every block has its own MoE keys.
    for i in range(4):
        assert any(k.startswith(f"blocks.{i}.feed_forward_moe.") for k in sd)

    # Round-trip still works.
    fresh = config.build(init_device="cpu")
    fresh.init_weights(device=torch.device("cpu"))
    fresh.load_state_dict(sd, strict=True)
    for (name, pa), (_, pb) in zip(model.named_parameters(), fresh.named_parameters()):
        assert torch.equal(pa, pb), f"mismatch at {name}"


def test_shared_blocks_rejects_block_overrides_in_shared_range():
    """
    block_overrides at indices that fall inside shared_blocks.layer_indices() would be
    silently clobbered by _apply_shared_blocks. Reject the combination at construction
    time with a clear error rather than producing a wrong model.
    """
    sb = SharedBlockConfig(
        start_layer=1, n_layers=3, share_routers=True, share_experts=True
    )
    config = _small_shared_moe_config(n_layers=5, shared_blocks=sb)
    # Pretend the user added an override on block 2, which is inside [1, 2, 3].
    config.block_overrides = {2: config.block}
    with pytest.raises(Exception, match="fall inside the shared range"):
        config.build(init_device="cpu")


def test_shared_blocks_allows_block_overrides_outside_shared_range():
    """
    block_overrides at indices outside the shared range are fine and should build
    normally. Sanity check that the validation isn't over-eager.
    """
    sb = SharedBlockConfig(
        start_layer=1, n_layers=3, share_routers=True, share_experts=True
    )
    config = _small_shared_moe_config(n_layers=5, shared_blocks=sb)
    # Block 0 and block 4 are outside [1, 2, 3] — overrides there are legal.
    config.block_overrides = {0: config.block, 4: config.block}
    model = config.build(init_device="cpu")
    assert isinstance(model, MoETransformer)


def test_shared_blocks_validates_range():
    with pytest.raises(Exception):
        SharedBlockConfig(start_layer=5, n_layers=4, share_routers=True).validate_against(6)
    with pytest.raises(Exception):
        SharedBlockConfig(start_layer=0, n_layers=2)
    with pytest.raises(Exception):
        SharedBlockConfig(start_layer=0, n_layers=1, share_routers=True)
