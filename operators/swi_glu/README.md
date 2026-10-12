# SwiGLU

SwiGLU is the first operator optimization task in AgentForge.

This task focuses on fundamental Ascend C optimization techniques, including vector computation, memory access, tiling, data movement, and operator fusion.

## 1. Operator Definition

The operator computes:

    output = SiLU(x0) * x1

where:

    x0, x1 = chunk(input, 2, dim)

The split dimension must have an even size.

Official interface:

    cann_bench.swi_glu(Tensor input, int dim=-1) -> Tensor output

Supported input dtypes:

- float16
- float32
- bfloat16

The output uses the same dtype as the input.

FP16 and BF16 inputs are evaluated against a reference implementation that performs the main computation in FP32 and casts the result back to the original dtype.

## 2. Official Baseline

The organizer-provided baseline is located at:

    operators/swi_glu/baseline/

The baseline is designed to be:

- compilable
- functionally correct
- easy for an Agent to understand and modify
- intentionally not fully optimized

The official baseline should be treated as the immutable starting point of an optimization run.

A Harness should create a working candidate from the clean baseline before modifying source code.

## 3. Reference Baseline Result

Reference environment:

- Accelerator: Ascend 910B2
- Product family: Ascend A2
- NPU count: 1
- CANN: 9.1.0
- Python: 3.12.13
- PyTorch: 2.10.0
- torch_npu: 2.10.0.post4
- CANN Bench hardware key: 910b2
- CANN Bench revision: d212f691486ffa1ae0e4f7dfe42bc3cdafea883e

Reference CANN Bench result:

| Metric | Result |
|---|---:|
| Correctness cases | 20 / 20 |
| Correctness rate | 100% |
| Compile / run score | 20.00 / 20 |
| Correctness score | 30.00 / 30 |
| Performance score | 24.34 / 50 |
| Overall CANN Bench score | 74.34 / 100 |
| Reported average speedup | 1.19x |

These numbers are provided only as a reference.

Final competition results are determined by the official evaluation environment and may vary slightly because of runtime noise and system conditions.

## 4. Optimization Scope

The primary optimization target is:

    csrc/ops/swi_glu/op_kernel/swi_glu_kernel.cpp

Within a candidate copy of the baseline, participants and autonomous Agents may modify:

    csrc/ops/swi_glu/op_kernel/*
    csrc/ops/swi_glu/op_plugin/*
    csrc/ops/swi_glu/CMakeLists.txt

Typical optimization topics include:

- tiling strategy
- block scheduling
- vector utilization
- UB utilization
- data-copy strategy
- copy / compute overlap
- double buffering
- dtype-specific paths
- shape-specific fast paths
- split-dimension-aware scheduling
- reduction of unnecessary data movement

The organizer does not prescribe a specific optimization strategy.

## 5. Wrapper Restrictions

The C++ / Python wrapper layer may be used for:

- reading tensor shape
- reading dtype
- reading operator attributes
- calculating tiling parameters
- allocating output tensors
- launching the submitted Ascend C kernel

The wrapper layer must not perform the target computation using PyTorch, torch_npu, ACLNN, or an existing CANN implementation of SwiGLU.

Tensor transformations that are part of the operator semantics must be implemented inside the submitted kernel rather than delegated to existing framework operators.

## 6. Frozen Evaluation Components

Participants must not modify the official evaluation definition, including:

- proto.yaml
- golden.py
- cases.yaml
- CANN Bench evaluator
- scoring logic
- 910B2 performance metadata
- official task definitions

The official CANN Bench repository used by AgentForge is kept separately from participant candidate code.

## 7. Build

Environment creation and dependency installation are documented in the
project-level `environment.md`.

The commands below assume the recommended virtual environment path:

    /workspace/agentforge-env

If a different virtual environment path is used, replace it accordingly.

Initialize the environment:

    source /workspace/agentforge-env/bin/activate
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

Build the baseline:

    cd /workspace/AgentForge/operators/swi_glu/baseline
    bash build.sh --soc=ascend910b

A successful build should generate a wheel under:

    dist/

## 8. Correctness Evaluation

Run correctness only:

    cd /workspace/cann-bench

    ./scripts/run_evaluation.sh \
      --source-dir /workspace/AgentForge/operators/swi_glu/baseline \
      --task-dir tasks/level1/swi_glu \
      --device-id 0 \
      --no-perf

The organizer baseline has been verified to pass all 20 official correctness cases.

## 9. Performance Evaluation

Run full evaluation:

    cd /workspace/cann-bench

    ./scripts/run_evaluation.sh \
      --source-dir /workspace/AgentForge/operators/swi_glu/baseline \
      --task-dir tasks/level1/swi_glu \
      --device-id 0 \
      --warmup 3 \
      --repeat 5

The exact final competition parameters will be frozen and published before the contest.

## 10. AgentForge Workflow

The intended optimization workflow is:

    Clean Baseline
          |
          v
    Create Candidate
          |
          v
    Analyze
          |
          v
    Modify Kernel
          |
          v
    Compile
          |
          v
    Correctness
          |
          v
    Benchmark / Profile
          |
          v
    Keep or Roll Back
          |
          v
    Iterate

During the official autonomous optimization phase, manual kernel modification is not allowed.

The submitted Harness is expected to perform the optimization process autonomously under the competition resource and time budget.
