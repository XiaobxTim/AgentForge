#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AgentForge mHC Sinkhorn baseline."""

__version__ = "1.0.0"

import torch

try:
    from . import _C
except ImportError as exc:
    raise ImportError(
        "Cannot import cann_bench._C. "
        "Build and install the AgentForge baseline first."
    ) from exc


def mhc_sinkhorn(
    comb: torch.Tensor,
    iter_step: int = 20,
    eps: float = 1e-6,
) -> torch.Tensor:
    return torch.ops.cann_bench.mhc_sinkhorn(
        comb,
        iter_step,
        eps,
    )
