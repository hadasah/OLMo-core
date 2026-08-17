from typing import Optional

import torch
import torch.nn as nn


class ResidualStream(nn.Module):
    """
    A parameter-free module that just handles a residual stream connection, like those in a transformer
    block. The benefit of using this module instead of a direct add operation is that the flexible
    to configure hooks for logging or other purposes, like with the
    :class:`olmo_core.train.callbacks.GAPMonitorCallback`.

    :param fom_prob: If set, applies Final Output Masking (FOM) to the sub-layer output ``x``:
        during training the entire output vector is independently zeroed for each token with
        probability ``fom_prob``. Unlike :data:`dropout`, this masks whole tokens (not individual
        channels) and does not rescale the surviving tokens. This is the dense-block equivalent of
        the MoE :data:`~olmo_core.nn.moe.MoEConfig.fom_prob`.
    """

    def __init__(
        self, alpha: float = 1.0, dropout: float = 0.0, fom_prob: Optional[float] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fom_prob = fom_prob

    def forward(self, residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Final Output Masking (FOM): independently zero the sub-layer output for each token
        # with probability `fom_prob` during training. No rescaling is applied.
        if self.fom_prob is not None and self.training and torch.is_grad_enabled():
            keep_mask = torch.empty(
                (*x.shape[:-1], 1), dtype=x.dtype, device=x.device
            ).bernoulli_(1.0 - self.fom_prob)
            x = x * keep_mask
        return torch.add(residual, self.dropout(x), alpha=self.alpha)
