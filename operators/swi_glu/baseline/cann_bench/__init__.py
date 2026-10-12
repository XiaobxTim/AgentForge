#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AgentForge SwiGLU baseline."""

__version__ = "1.0.0"

import torch

try:
    from . import _C
except ImportError as exc:
    raise ImportError(
        "Cannot import cann_bench._C. "
        "Build and install the AgentForge baseline first."
    ) from exc


def swi_glu(
    input: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    return torch.ops.cann_bench.swi_glu(
        input,
        dim,
    )
