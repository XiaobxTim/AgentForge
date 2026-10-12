# AgentForge Environment

AgentForge is developed and evaluated on a fixed Ascend A2 environment.

## Hardware

- Accelerator: Ascend 910B2
- Product family: Ascend A2
- NPU count: 1
- HBM: 64 GB

## Operating System

- Ubuntu 22.04.5 LTS
- Architecture: aarch64

## Ascend Software Stack

- CANN: 9.1.0
- Bisheng: 15.0.5
- npu-smi: 26.1.1

## Python Stack

- Python: 3.12.13
- PyTorch: 2.10.0
- torch_npu: 2.10.0.post4
- NumPy: 2.2.6
- PyYAML: 6.0.3

## CANN Bench

Repository:

https://gitcode.com/cann/cann-bench.git

Frozen commit:

d212f691486ffa1ae0e4f7dfe42bc3cdafea883e

Hardware resolver:

Ascend910B2 -> 910b2

Performance metadata:

tasks/metadata/910b2.json

The following AgentForge operators are all covered by the official
910B2 metadata:

- swi_glu
- mhc_sinkhorn
- engram_gate_fusion

## Environment Initialization

Before development or evaluation:

```
bash
source /workspace/agentforge-env/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```