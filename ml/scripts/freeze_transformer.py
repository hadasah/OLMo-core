"""
Freeze train module for continued training with first-half expert parameters frozen.

Provides FreezeTransformerTrainModuleConfig and FreezeTransformerTrainModule, which
subclass the standard olmo_core train module classes to zero out gradients for the
first half of expert (and router) parameters after each backward pass.

Usage in train.py:
    from freeze_train_module import FreezeTransformerTrainModuleConfig
    train_module_config = FreezeTransformerTrainModuleConfig(
        freeze_experts="first_half",
        ...  # same kwargs as TransformerTrainModuleConfig
    )
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.config import DType
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.utils import get_full_tensor, get_local_tensor
from olmo_core.nn.transformer import Transformer
from olmo_core.optim import SkipStepOptimizer
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module.transformer import (
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import move_to_device
from torch.distributed.tensor import DTensor, distribute_tensor

log = logging.getLogger(__name__)


def distribute_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Redistribute `target` to match the DTensor layout of `source`, or return
    the full tensor if `source` is not a DTensor.
    """
    if not isinstance(source, DTensor):
        return get_full_tensor(target)
    if isinstance(target, DTensor):
        if target.device_mesh == source.device_mesh and target.placements == source.placements:
            return target
        else:
            return target.redistribute(device_mesh=source.device_mesh, placements=source.placements)
    return distribute_tensor(target, device_mesh=source.device_mesh, placements=source.placements)


@dataclass
class FreezeTransformerTrainModuleConfig(TransformerTrainModuleConfig):
    """
    Extension of TransformerTrainModuleConfig that adds expert-freezing behaviour.

    After each backward pass, gradients for the first half of expert (and corresponding
    router) parameters are zeroed out so that only the second half of experts is updated.

    Extra parameters
    ----------------
    freeze_experts : str
        Which half of experts to freeze. Currently only ``"first_half"`` is supported.
    """

    def __init__(self, *args, freeze_experts: str = "first_half", **kwargs):
        self.freeze_experts = freeze_experts
        super().__init__(*args, **kwargs)

    def build(
        self,
        model: Transformer,
        device: Optional[torch.device] = None,
    ) -> "FreezeTransformerTrainModule":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)
        # freeze_experts is not a field on the parent dataclass, so pop it before forwarding.
        freeze_experts = kwargs.pop("freeze_experts", self.freeze_experts)
        return FreezeTransformerTrainModule(
            model=model,
            device=device,
            freeze_experts=freeze_experts,
            **kwargs,
        )


class FreezeTransformerTrainModule(TransformerTrainModule):
    """
    Custom transformer train module that zeros out gradients for the first half of
    expert (and router) parameters after each backward pass.

    Inherits all standard training behaviour from TransformerTrainModule and only
    overrides ``train_batch`` to inject the gradient-masking step.

    Parameters
    ----------
    freeze_experts : str
        Which experts to freeze. Supported value: ``"first_half"``.
    *args, **kwargs
        Forwarded to :class:`TransformerTrainModule`.
    """

    def __init__(self, *args, freeze_experts: str = "first_half", **kwargs):
        self.freeze_experts = freeze_experts
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Gradient masking helpers
    # ------------------------------------------------------------------

    def _zero_first_half_grad(self, param: torch.Tensor, name: str) -> None:
        """Zero the gradient for the first half of `param` along dim-0."""
        if param.grad is None:
            return

        full_grad = get_full_tensor(param.grad)

        if "experts" in name:
            # Expert weight tensors are [num_experts, ...]; zero the first half.
            mask = torch.zeros_like(full_grad, dtype=torch.bool)
            mask[: full_grad.shape[0] // 2, :] = True
        elif "router" in name:
            # Router weight tensors are [num_experts, d_model]; zero the first half.
            mask = torch.zeros_like(full_grad, dtype=torch.bool)
            mask[: full_grad.shape[0] // 2, :] = True
        else:
            return

        local_mask = get_local_tensor(distribute_like(param, mask))
        get_local_tensor(param.grad).masked_fill_(local_mask, 0.0)

    def _apply_expert_grad_mask(self) -> None:
        """
        Iterate over all named parameters and apply gradient masking to
        expert / router parameters according to ``self.freeze_experts``.
        """
        if self.freeze_experts != "first_half":
            raise ValueError(
                f"Unsupported freeze_experts value: '{self.freeze_experts}'. "
                "Only 'first_half' is currently supported."
            )

        for name, param in self.model.named_parameters():
            if "experts" in name or "router" in name:
                self._zero_first_half_grad(param, name)

    # ------------------------------------------------------------------
    # Overridden training step
    # ------------------------------------------------------------------

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False) -> None:
        # ---- identical to TransformerTrainModule.train_batch up to the ----
        # ---- backward pass; the only addition is _apply_expert_grad_mask ----

        self.model.train()

        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
            self.record_metric(
                "train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean
            )

        batch_num_tokens_for_loss = move_to_device(
            (batch["labels"] != self.label_ignore_index).sum(), self.device
        )
        if self.cp_enabled:
            assert self._cp_config is not None
            batch_num_tokens_for_loss = batch_num_tokens_for_loss / self._cp_config.degree

        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = None
        if self.z_loss_multiplier is not None:
            z_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        auxiliary_batch_losses: Dict[str, torch.Tensor] = {}

        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small "
                f"relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

                _, ce_loss, z_loss = self.model_forward(
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="sum",
                    z_loss_multiplier=self.z_loss_multiplier,
                    loss_div_factor=batch_num_tokens_for_loss,
                    return_logits=False,
                    **model_kwargs,
                )

                loss = ce_loss
                if z_loss is not None:
                    loss = loss + z_loss

                ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss

                auxiliary_losses = self.model.compute_auxiliary_losses(
                    batch_num_tokens_for_loss, reset=True
                )
                for loss_name, loss_val in auxiliary_losses.items():
                    loss = loss + loss_val
                    loss_val = get_local_tensor(loss_val.detach())
                    if loss_name in auxiliary_batch_losses:
                        auxiliary_batch_losses[loss_name] += loss_val
                    else:
                        auxiliary_batch_losses[loss_name] = loss_val
                del auxiliary_losses

                loss.backward()

        # ----- NEW: zero out frozen-expert gradients after all micro-batches -----
        self._apply_expert_grad_mask()
        # -------------------------------------------------------------------------

        del batch

        if dry_run:
            self.model.reset_auxiliary_losses()
            self.model.reset_auxiliary_metrics()
            return

        self.record_ce_loss(ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            self.record_metric("Z loss", z_batch_loss, ReduceType.mean, namespace="train")
        for loss_name, loss_val in auxiliary_batch_losses.items():
            self.record_metric(loss_name, loss_val, ReduceType.mean, namespace="train")

        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(
            batch_num_tokens_for_loss,
            reset=True,
        ).items():
            self.record_metric(metric_name, metric_val, reduction, namespace="train")

        if isinstance(self.optim, SkipStepOptimizer):
            self.optim.latest_loss = ce_batch_loss