# AgentForge

AgentForge is an **Agentic Operator Optimization Hackathon** for Ascend C.

The goal of the challenge is to build an autonomous optimization system that can iteratively analyze, modify, compile, evaluate, and optimize Ascend C kernels.

Participants start from organizer-provided baseline implementations and use an Agent / Harness workflow to improve kernel performance while preserving correctness.

## Challenge Tasks

AgentForge contains three operator optimization tasks:

1. **SwiGLU**
2. **mHC Sinkhorn**
3. **Engram Gate Fusion**

The three tasks form a progressive optimization path:

- SwiGLU: vector computation, memory access, fusion, and tiling
- mHC Sinkhorn: reduction, iteration, tiny-matrix scheduling, and data residency
- Engram Gate Fusion: multi-stage fusion, RMSNorm, gating, depthwise convolution, residual computation, and data reuse

## Target Platform

The current development and evaluation platform is:

- Accelerator: Ascend A2 / Ascend 910B2
- NPU Count: 1
- HBM: 64 GB
- OS: Ubuntu 22.04.5 LTS
- Architecture: aarch64
- CANN: 9.1.0
- Python: 3.12
- PyTorch: 2.10.0
- torch_npu: 2.10.0.post4

Detailed environment information is available in:

`environment.md`

## Evaluation

AgentForge uses the open-source **CANN Bench** evaluator on the contest hardware.

The currently frozen CANN Bench revision is:

`d212f691486ffa1ae0e4f7dfe42bc3cdafea883e`

Target hardware key:

`910b2`

Performance metadata:

`tasks/metadata/910b2.json`

The following challenge operators are covered by the official 910B2 metadata:

- `swi_glu`
- `mhc_sinkhorn`
- `engram_gate_fusion`

The public `cannbench.com` service is not used as the AgentForge contest evaluation backend.

## Optimization Workflow

The intended autonomous optimization workflow is:

    Analyze
      ↓
    Plan
      ↓
    Modify Kernel
      ↓
    Compile
      ↓
    Correctness Check
      ↓
    Benchmark / Profile
      ↓
    Candidate Selection / Rollback
      ↓
    Iterate

The final evaluation uses the kernel produced by the autonomous Harness.

During the final autonomous optimization phase, manual kernel modification is not allowed.

## Repository Structure

    AgentForge/
    ├── operators/
    │   ├── swi_glu/
    │   │   └── baseline/
    │   ├── mhc_sinkhorn/
    │   │   └── baseline/
    │   └── engram_gate_fusion/
    │       └── baseline/
    │
    ├── harness/
    │   └── starter/
    │
    ├── tools/
    ├── scripts/
    ├── docs/
    │
    ├── requirements.txt
    ├── environment.md
    ├── .gitignore
    └── README.md

## Baselines

The organizer will provide functional Ascend C baseline implementations for all three tasks.

Each baseline must:

- compile successfully on the target Ascend environment
- pass the corresponding CANN Bench correctness cases
- provide meaningful optimization headroom
- remain understandable and editable by an autonomous Agent

The baselines are intentionally not intended to represent highly optimized final implementations.

## Harness

AgentForge does not prescribe a single Agent framework or Skill implementation.

A minimal Starter Harness will be provided to demonstrate the basic optimization loop.

Participants may extend the Starter Harness or build their own legal Agent / Skill workflow.

The Harness is expected to coordinate operations such as:

- source-code analysis
- optimization planning
- source modification
- compilation
- correctness validation
- benchmarking
- profiling
- candidate comparison
- rollback
- iterative optimization

## Reproducibility

The contest environment, CANN Bench revision, operator specifications, evaluation cases, and performance metadata will be frozen for the official competition.

The repository will also provide unified scripts for:

- environment initialization
- compilation
- correctness testing
- benchmarking
- profiling
- final reproduction

These scripts will be added after the three baseline implementations are validated on the target Ascend 910B2 environment.

## Development Status

Current progress:

- [x] Ascend 910B2 environment verified
- [x] CANN 9.1.0 verified
- [x] PyTorch + torch_npu NPU execution verified
- [x] CANN Bench hardware resolver verified
- [x] CANN Bench 910B2 metadata verified
- [x] CANN Bench compilation pipeline verified
- [x] CANN Bench correctness evaluation verified
- [x] CANN Bench performance profiling verified
- [ ] SwiGLU baseline
- [ ] mHC Sinkhorn baseline
- [ ] Engram Gate Fusion baseline
- [ ] Unified evaluation tools
- [ ] Starter Harness

## License

License information will be added before the public contest release.
