# Engram Gate Fusion — AgentForge Starter Baseline

This directory contains the **Starter Baseline** for the AgentForge
**Engram Gate Fusion** operator challenge.

This is the highest-complexity operator in AgentForge.

The baseline is correct and readable, but intentionally leaves substantial
optimization space in kernel fusion, memory traffic, reduction, scheduling,
and stateful ShortConv execution.

## 1. Task

CANN Bench task:

~~~text
tasks/level3/engram_gate_fusion
~~~

Operator interface:

~~~python
cann_bench.engram_gate_fusion(
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
) -> (output, conv_state_out)
~~~

Main tensor layouts:

| Tensor | Shape | Dtype |
|---|---|---|
| `keys` | `[B, L, HC, D]` | BF16 |
| `hidden_states` | `[B, L, HC, D]` | BF16 |
| `value` | `[B, L, D]` | BF16 |
| `norm1_weight` | `[HC, D]` | FP32 |
| `norm2_weight` | `[HC, D]` | FP32 |
| `conv_norm_weight` | `[HC, D]` | FP32 |
| `conv_weight` | `[HC * D, 1, K]` | FP32 |
| `conv_state` | `[B, HC * D, state_len]` | BF16 |

where:

~~~text
state_len = (kernel_size - 1) * dilation
~~~

## 2. Operator Computation

The operator performs the following major stages:

~~~text
keys -------- RMSNorm -------\
                              \
                               gate
                              /
hidden ------ RMSNorm -------/

value ---------------------- gate multiplication
                              |
                              v
                         value_gated
                              |
                              +---------------- residual ----------------+
                              |                                          |
                           RMSNorm                                       |
                              |                                          |
                              v                                          |
                    depthwise ShortConv                                 |
                              |                                          |
                            SiLU                                         |
                              |                                          |
                              +---------------- Add <--------------------+
                                               |
                                               v
                                             output
~~~

The gate is computed conceptually as:

~~~text
raw_gate =
    sum(normed_keys * normed_hidden) / sqrt(D)

gate =
    sigmoid(
        sqrt(clamp(abs(raw_gate), 1e-6))
        * sign(raw_gate)
    )

value_gated =
    gate * value
~~~

The gated value is normalized again before entering the depthwise
causal dilated convolution.

## 3. Stateful ShortConv

For prefill without an input state:

~~~text
x_cat = concat(zeros(state_len), x)
~~~

For stateful decode:

~~~text
x_cat = concat(conv_state, x)
~~~

The returned state is always:

~~~text
conv_state_out = x_cat[..., -state_len:]
~~~

The Starter Baseline contains separate Stage-B paths for prefill and
stateful decode.

## 4. Starter Baseline Architecture

The implementation intentionally uses two major kernel stages.

~~~text
Stage A
  |
  |-- RMSNorm(keys)
  |-- RMSNorm(hidden_states)
  |-- gate
  |-- value_gated
  `-- RMSNorm(value_gated)
          |
          |  FP32 Global Memory workspaces
          v
Stage B
  |
  |-- depthwise ShortConv
  |-- Swish / SiLU
  |-- residual add
  |-- output
  `-- conv_state_out
~~~

Two FP32 intermediate tensors are materialized in Global Memory:

~~~text
value_gated
normed_value_gated
~~~

This simplifies the baseline but creates substantial HBM traffic and
therefore exposes meaningful fusion and data-reuse opportunities.

## 5. Baseline Characteristics

The Starter Baseline intentionally contains:

- two separate kernel stages;
- FP32 Global Memory intermediate workspaces;
- scalar-heavy reduction logic;
- straightforward depthwise convolution;
- simple work-item scheduling;
- limited shape specialization;
- separate prefill and decode handling;
- limited reuse across normalization, gating, and convolution.

The implementation does **not** use artificial sleeps or deliberate
performance sabotage.

Its low performance comes from the actual baseline architecture.

## 6. Baseline Reference Result

Validated environment:

| Item | Value |
|---|---|
| Device | Ascend 910B2 / Ascend A2 |
| CANN | 9.1.0 |
| Correctness | 20 / 20 |
| Pass rate | 100% |
| Overall score | 50.70 |
| Average speedup | 0.02x |

The **AgentForge Starter Baseline** is separate from the
**CANN Bench reference implementation / hardware anchor**.

## 7. Potential Optimization Directions

Potential directions include:

- fusing Stage A and Stage B;
- eliminating FP32 Global Memory workspaces;
- keeping more intermediate values in Unified Buffer;
- reducing repeated HBM traffic;
- vectorizing RMSNorm and gate reductions;
- improving depthwise ShortConv scheduling;
- increasing weight and activation reuse;
- specializing prefill and decode paths;
- specializing for different `D`, `HC`, `K`, dilation, and sequence lengths;
- improving multicore decomposition.

Participants are not restricted to these approaches.

## 8. Build

Recommended environment:

~~~bash
source /workspace/agentforge-env/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
~~~

Build:

~~~bash
cd /workspace/AgentForge/operators/engram_gate_fusion/baseline

rm -rf build dist
rm -f cann_bench/_C.abi3.so

bash build.sh --soc=ascend910b
~~~

For the complete environment description, see:

~~~text
/workspace/AgentForge/environment.md
~~~

## 9. Evaluation

Correctness only:

~~~bash
cd /workspace/cann-bench

./scripts/run_evaluation.sh \
  --source-dir /workspace/AgentForge/operators/engram_gate_fusion/baseline \
  --task-dir tasks/level3/engram_gate_fusion \
  --device-id 0 \
  --no-perf
~~~

Full evaluation:

~~~bash
cd /workspace/cann-bench

./scripts/run_evaluation.sh \
  --source-dir /workspace/AgentForge/operators/engram_gate_fusion/baseline \
  --task-dir tasks/level3/engram_gate_fusion \
  --device-id 0
~~~

## 10. AgentForge Rules

The Starter Baseline is the starting point for autonomous optimization.

The final candidate must remain a valid Ascend C implementation.

The core computation must not be replaced by CPU fallback, cached or
hard-coded outputs, delegation to an existing implementation of the target
operator, fake tensors, or timing/profiler manipulation.

The frozen CANN Bench evaluator is authoritative for compilation,
correctness, performance, and final operator score.
