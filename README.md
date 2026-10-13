# AgentForge

AgentForge is an **Agentic Operator Optimization** framework for Ascend C.

This repository provides three validated Ascend C starter baselines, a unified CANN Bench evaluation interface, and a minimal Starter Harness for autonomous kernel optimization.

The Starter Harness is intentionally lightweight. It provides the basic optimization infrastructure while leaving substantial room for participants to design their own Agent architecture, context strategy, Skills, search strategy, memory, profiling workflow, and optimization logic.

## Challenge Tasks

AgentForge contains three operator optimization tasks:

1. **SwiGLU**
2. **mHC Sinkhorn**
3. **Engram Gate Fusion**

The operators cover progressively more complex optimization patterns:

- **SwiGLU**: vector computation, memory access, fusion, and tiling
- **mHC Sinkhorn**: reduction, iteration, tiny-matrix scheduling, and data residency
- **Engram Gate Fusion**: multi-stage computation, RMSNorm, gating, depthwise convolution, state handling, residual computation, and data reuse

## Target Platform

The current validated environment is:

- Accelerator: Ascend A2 / Ascend 910B2
- NPU Count: 1
- HBM: 64 GB
- OS: Ubuntu 22.04.5 LTS
- Architecture: aarch64
- CANN: 9.1.0
- Python: 3.12
- PyTorch: 2.10.0
- torch_npu: 2.10.0.post4


### Reference Model Runtime

The competition environment provides the Qwen3.5-4B BF16 model weights for autonomous optimization. AgentForge includes a verified Transformers-based reference runtime for loading and running the model on Ascend 910B2.

The reference runtime is provided for reproducibility and development convenience. Teams may implement their own compatible model runtime; the model weights are standardized, while the internal inference and Harness implementation remain part of the design space.

The model path is provided through `AGENTFORGE_MODEL_PATH`. See `environment.md` for the validated model runtime and dependency versions.

Detailed environment and dependency information is available in `environment.md` and `requirements.txt`.

## Repository Structure

    AgentForge/
    ├── README.md
    ├── environment.md
    ├── requirements.txt
    ├── run_agent.sh
    ├── .gitignore
    │
    ├── operators/
    │   ├── swi_glu/
    │   │   └── baseline/
    │   ├── mhc_sinkhorn/
    │   │   └── baseline/
    │   └── engram_gate_fusion/
    │       └── baseline/
    │
    └── harness/
        └── starter/
            ├── evaluator.py
            ├── workspace.py
            ├── iteration.py
            ├── round.py
            ├── inspector.py
            ├── source_context.py
            ├── runner.py
            ├── agent_entrypoint.sh
            ├── operators.yaml
            ├── AGENT_PROTOCOL.md
            ├── CONTESTANT_INTERFACE.md
            └── skills/

## Starter Baselines

The repository includes validated Ascend C baseline implementations for all three operators.

Each baseline:

- compiles successfully on the validated Ascend environment;
- passes all corresponding frozen CANN Bench correctness cases;
- is intentionally understandable and editable;
- leaves meaningful room for further optimization.

Representative baseline measurements on the current Ascend 910B2 environment are:

| Operator | Correctness | Overall Score | Average Speedup |
|---|---:|---:|---:|
| SwiGLU | 20/20 | ~74.2 | ~1.18x |
| mHC Sinkhorn | 20/20 | ~53.3 | ~0.11x |
| Engram Gate Fusion | 20/20 | 50.70 | 0.02x |

Performance measurements may vary slightly between runs.

The frozen baseline tags are:

- `swi-glu-baseline-v1`
- `mhc-sinkhorn-baseline-v1`
- `engram-gate-fusion-baseline-v1`

## Starter Harness

The repository includes a minimal model-independent Starter Harness for autonomous optimization.

Its main components are:

- `evaluator.py`: invokes CANN Bench and parses correctness and performance results;
- `workspace.py`: creates isolated candidate workspaces and snapshots;
- `iteration.py`: manages candidate comparison, acceptance, rejection, and rollback;
- `round.py`: manages one optimization round and its artifacts;
- `inspector.py`: summarizes the current candidate source tree;
- `source_context.py`: builds a compact source context for an Agent;
- `runner.py`: coordinates the full multi-round optimization workflow;
- `agent_entrypoint.sh`: model-independent placeholder interface for a contestant Agent;
- `skills/`: minimal interface and templates for contestant-designed or externally sourced Skills.

The Starter Harness is not intended to be an optimal autonomous optimizer. It is a working starting point that can be redesigned or extended.

## Autonomous Workflow

The default workflow is:

    Frozen Baseline
          ↓
    Candidate Workspace
          ↓
        Analyze
          ↓
         Plan
          ↓
     Modify Candidate
          ↓
      Correctness
          ↓
      Performance
          ↓
    Accept / Rollback
          ↓
        Iterate
          ↓
    Export Best Candidate

The Harness preserves the current best snapshot and restores it when a candidate fails correctness, regresses in performance, the Agent fails, or the Agent times out.

## Quick Start

Activate the prepared environment first, or follow `environment.md` to configure an equivalent environment.

The standard repository entry point is:

    ./run_agent.sh \
      --operator engram_gate_fusion \
      --budget 10 \
      --run-id example_run

Supported operators are:

- `swi_glu`
- `mhc_sinkhorn`
- `engram_gate_fusion`

Useful options can be inspected with:

    ./run_agent.sh --help

The default `agent_entrypoint.sh` is intentionally a stub. A real Agent implementation should replace or extend this interface.

## Skills

The repository intentionally does not provide operator-specific optimization Skills.

The `harness/starter/skills/` directory contains only the Skill interface and templates. Users may design their own Skills or adapt suitable Skills from public repositories.

See `harness/starter/skills/README.md` for the repository-level Skill interface.

## Evaluation

AgentForge uses the open-source **CANN Bench** evaluator on the target hardware.

The frozen CANN Bench revision used during current development is:

`d212f691486ffa1ae0e4f7dfe42bc3cdafea883e`

Target hardware key:

`910b2`

Performance metadata:

`tasks/metadata/910b2.json`

The three AgentForge operators covered by the current evaluator are:

- `swi_glu`
- `mhc_sinkhorn`
- `engram_gate_fusion`

The public `cannbench.com` service is not used as the repository evaluation backend.

## Harness Artifacts

Runtime outputs are stored under the Starter Harness workspace and are excluded from Git.

A run may contain:

    workspaces/<operator>/<run-id>/
    ├── candidate/
    ├── snapshots/
    ├── rounds/
    │   └── round_XXXX/
    │       ├── context.json
    │       ├── SOURCE_CONTEXT.md
    │       ├── analysis.md
    │       ├── plan.json
    │       ├── agent_status.json
    │       └── result.json
    └── final/
        ├── best_candidate/
        └── final_summary.json

## Additional Documentation

- `environment.md`: validated software and hardware environment
- `harness/starter/AGENT_PROTOCOL.md`: Starter Harness round protocol
- `harness/starter/CONTESTANT_INTERFACE.md`: standard Harness / Agent interface
- operator-level README files: baseline implementation and build details

Competition rules, scoring, submission requirements, and other event-specific information are maintained separately from this code repository.

## License

License information will be provided with the public release.
