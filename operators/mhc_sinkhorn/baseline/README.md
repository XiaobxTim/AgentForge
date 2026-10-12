# mHC Sinkhorn — AgentForge Starter Baseline

This directory contains the **Starter Baseline** for the AgentForge
**mHC Sinkhorn** operator challenge.

The baseline is designed to be correct, readable, and easy to optimize.
It is intentionally simple and is **not** intended to represent a
high-performance implementation.

## 1. Task

CANN Bench task:

~~~text
tasks/level3/mhc_sinkhorn
~~~

Operator interface:

~~~python
cann_bench.mhc_sinkhorn(
    comb,
    iter_step=20,
    eps=1e-6,
) -> output
~~~

Input:

~~~text
comb: Tensor[B, M, M], FP32
~~~

The frozen CANN Bench task covers different batch sizes, matrix sizes,
and iteration counts.

## 2. Computation

For each small matrix, the operator performs iterative Sinkhorn-style
normalization.

The first iteration performs numerically stable row softmax followed by
column normalization:

~~~text
row_max = max(x, dim=row)
x = exp(x - row_max)
x = x / (row_sum + eps)
x = x / (col_sum + eps)
~~~

Subsequent iterations repeatedly perform:

~~~text
x = x / (row_sum + eps)
x = x / (col_sum + eps)
~~~

The number of iterations is controlled by `iter_step`.

## 3. Starter Baseline Design

The baseline processes one small matrix at a time and keeps its working
data in Unified Buffer where possible.

Its implementation is intentionally straightforward:

1. Load one `[M, M]` matrix.
2. Perform row reductions.
3. Normalize rows.
4. Perform column reductions.
5. Normalize columns.
6. Repeat for `iter_step`.
7. Write the result back to Global Memory.

The baseline contains several deliberately simple implementation choices:

- scalar-heavy `GetValue` / `SetValue` reductions;
- limited parallelism inside each small matrix;
- limited batching of independent matrices;
- no specialized scheduling for different values of `M`;
- no aggressive vectorized reduction strategy;
- no advanced data-layout transformation.

These are real implementation limitations rather than artificial delays.

## 4. Baseline Reference Result

Validated environment:

| Item | Value |
|---|---|
| Device | Ascend 910B2 / Ascend A2 |
| CANN | 9.1.0 |
| Correctness | 20 / 20 |
| Pass rate | 100% |
| Overall score | 53.33 |
| Average speedup | 0.11x |

The low performance is expected for this intentionally simple baseline.

The **AgentForge Starter Baseline** is separate from the
**CANN Bench reference implementation / hardware anchor**.

## 5. Potential Optimization Directions

Possible optimization directions include:

- vectorizing row and column reductions;
- processing multiple small matrices concurrently;
- improving AI Vector Core utilization;
- specializing kernels for common values of `M`;
- reducing scalar `GetValue` / `SetValue` operations;
- improving data residency in Unified Buffer;
- reducing synchronization overhead;
- improving multicore scheduling;
- selecting different strategies for different batch sizes.

Participants are not restricted to these approaches.

## 6. Build

Recommended environment:

~~~bash
source /workspace/agentforge-env/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
~~~

Build:

~~~bash
cd /workspace/AgentForge/operators/mhc_sinkhorn/baseline

rm -rf build dist
rm -f cann_bench/_C.abi3.so

bash build.sh --soc=ascend910b
~~~

For the complete environment description, see:

~~~text
/workspace/AgentForge/environment.md
~~~

## 7. Evaluation

Correctness only:

~~~bash
cd /workspace/cann-bench

./scripts/run_evaluation.sh \
  --source-dir /workspace/AgentForge/operators/mhc_sinkhorn/baseline \
  --task-dir tasks/level3/mhc_sinkhorn \
  --device-id 0 \
  --no-perf
~~~

Full evaluation:

~~~bash
cd /workspace/cann-bench

./scripts/run_evaluation.sh \
  --source-dir /workspace/AgentForge/operators/mhc_sinkhorn/baseline \
  --task-dir tasks/level3/mhc_sinkhorn \
  --device-id 0
~~~

## 8. AgentForge Rules

The Starter Baseline is the starting point for autonomous optimization.

The final candidate must remain a valid Ascend C implementation.

The core computation must not be replaced by CPU fallback, cached or
hard-coded outputs, delegation to an existing implementation of the
target operator, fake tensors, or timing/profiler manipulation.

The frozen CANN Bench evaluator is authoritative for compilation,
correctness, performance, and final operator score.
