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


def engram_gate_stage_a(
    keys,
    hidden_states,
    value,
    norm1_weight,
    norm2_weight,
    conv_norm_weight,
    hc_mult=4,
    hidden_size=1024,
    norm_eps=1e-5,
):
    return torch.ops.cann_bench.engram_gate_stage_a(
        keys,
        hidden_states,
        value,
        norm1_weight,
        norm2_weight,
        conv_norm_weight,
        hc_mult,
        hidden_size,
        norm_eps,
    )
