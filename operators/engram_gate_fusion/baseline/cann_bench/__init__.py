#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AgentForge Engram Gate Fusion baseline development package."""

__version__ = "1.0.0"

import torch

try:
    from . import _C
except ImportError as exc:
    raise ImportError(
        "Cannot import cann_bench._C. "
        "Build the AgentForge Engram baseline first."
    ) from exc

def engram_gate_fusion(
    keys,
    hidden_states,
    value,
    norm1_weight,
    norm2_weight,
    conv_norm_weight,
    conv_weight,
    conv_state=None,
    hc_mult=4,
    hidden_size=1024,
    kernel_size=4,
    dilation=3,
    norm_eps=1e-5,
):
    return torch.ops.cann_bench.engram_gate_fusion(
        keys,
        hidden_states,
        value,
        norm1_weight,
        norm2_weight,
        conv_norm_weight,
        conv_weight,
        conv_state,
        hc_mult,
        hidden_size,
        kernel_size,
        dilation,
        norm_eps,
    )
