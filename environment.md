# AgentForge Environment

AgentForge is developed and evaluated on a fixed Ascend A2 environment.

This document describes both the reference environment and the steps required
to create the Python environment used by AgentForge from scratch.

---

## 1. Reference Hardware

- Accelerator: Ascend 910B2
- Product family: Ascend A2
- NPU count: 1
- HBM: 64 GB
- Architecture: aarch64

## 2. Operating System

- Ubuntu 22.04.5 LTS

## 3. Ascend Software Stack

- CANN: 9.1.0
- Bisheng: 15.0.5
- npu-smi: 26.1.1

Reference CANN installation:

    /usr/local/Ascend/cann-9.1.0

The standard environment setup script is:

    /usr/local/Ascend/ascend-toolkit/set_env.sh

## 4. Python

Reference Python:

    Python 3.12.13

Reference interpreter:

    /usr/local/python3.12.13/bin/python3

AgentForge uses a Python virtual environment.

The reference environment used during development is located at:

    /workspace/agentforge-env

`agentforge-env` is only the recommended environment name. It is not a
pre-installed system environment. A different virtual environment path may be
used, but the activation commands must be changed accordingly.

---

## 5. Create the Python Environment

From a clean machine with the required Python and CANN installation available:

    /usr/local/python3.12.13/bin/python3 -m venv /workspace/agentforge-env

Activate the environment:

    source /workspace/agentforge-env/bin/activate

Upgrade the basic Python packaging tools:

    python -m pip install --upgrade pip setuptools wheel

The shell prompt may now show something similar to:

    (agentforge-env)

## 6. Install AgentForge Python Dependencies

Move to the AgentForge repository:

    cd /workspace/AgentForge

Install the dependencies in `requirements.txt`.

On the reference aarch64 Ascend environment, `torch_npu` is obtained from the
Huawei Cloud PyPI mirror:

    python -m pip install \
      --index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

The core reference versions are:

- torch: 2.10.0
- torch_npu: 2.10.0.post4
- numpy: 2.2.6
- PyYAML: 6.0.3

The exact environment used for the official competition will be frozen before
the contest.

Do not use a full system-wide `pip freeze` as the project dependency list.
`requirements.txt` contains AgentForge project dependencies rather than every
package pre-installed in the Ascend image.

---

## 7. Load the CANN Environment

The Python virtual environment and the CANN environment are separate.

After activating the Python environment, load CANN:

    source /workspace/agentforge-env/bin/activate
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

Verify that the CANN installation is visible:

    echo $ASCEND_HOME_PATH

The reference environment should resolve to:

    /usr/local/Ascend/cann-9.1.0

Verify Bisheng:

    which bisheng
    bisheng --version

---

## 8. Verify PyTorch and torch_npu

Run:

    python - <<'PY'
    import torch
    import torch_npu

    print("torch:", torch.__version__)
    print("torch_npu:", torch_npu.__version__)
    print("NPU available:", torch.npu.is_available())
    print("NPU count:", torch.npu.device_count())

    if torch.npu.is_available():
        print("NPU name:", torch.npu.get_device_name(0))

        x = torch.tensor(
            [1.0, 2.0, 3.0],
            dtype=torch.float32,
            device="npu:0",
        )

        print("NPU tensor:", x)
    PY

The reference system should report:

    torch: 2.10.0+cpu
    torch_npu: 2.10.0.post4
    NPU available: True
    NPU count: 1
    NPU name: Ascend910B2

The `+cpu` suffix in the PyTorch version is expected. `torch_npu` provides the
Ascend NPU backend.

---


## 9. Official Model and Reference Runtime

AgentForge Final uses the organizer-provided Qwen3.5-4B BF16 model weights.

Reference model location used during development:

    /workspace/shared_assets/models/Qwen/Qwen3.5-4B

During official evaluation, participants should not hard-code this path.
The organizer provides the model location through:

    AGENTFORGE_MODEL_PATH

The Harness should read the model path from this environment variable.

The verified reference model runtime is:

- transformers: 5.19.0
- accelerate: 1.15.0
- safetensors: 0.8.0
- pillow: 12.3.0

This reference runtime has been verified with Qwen3.5-4B BF16 on Ascend 910B2.

The reference runtime is provided for reproducibility and baseline development.
It is not the only permitted inference runtime in Final evaluation. Teams may
design their own compatible model loading and inference runtime, subject to the
competition rules. The organizer standardizes the model weights rather than
the internal runtime implementation.

The organizer provides the following environment variables during Final:

    AGENTFORGE_MODEL_PATH
    AGENTFORGE_DEVICE
    AGENTFORGE_TIME_LIMIT_SECONDS

These variables are inherited by `run_agent.sh` and the child processes it
starts. A team may read them from any component of its Harness. Use of
`team_agent.sh` is optional and is not required for accessing these variables.

For example, a Python Harness may obtain the model location with:

    import os
    model_path = os.environ["AGENTFORGE_MODEL_PATH"]

Participants must not replace or override the organizer-provided model weights
during official evaluation.

---
## 10. CANN Bench

AgentForge uses the open-source CANN Bench evaluator.

Reference repository:

    https://gitcode.com/cann/cann-bench.git

Reference location used during development:

    /workspace/cann-bench

Clone the repository:

    cd /workspace
    git clone https://gitcode.com/cann/cann-bench.git
    cd cann-bench

The frozen AgentForge revision is:

    d212f691486ffa1ae0e4f7dfe42bc3cdafea883e

Checkout the frozen revision:

    git checkout d212f691486ffa1ae0e4f7dfe42bc3cdafea883e

Verify:

    git rev-parse HEAD
    git status

The expected commit is:

    d212f691486ffa1ae0e4f7dfe42bc3cdafea883e

The working tree should remain clean during official evaluation.

### Hardware Resolver

The reference device reports:

    Ascend910B2

CANN Bench resolves this device to:

    910b2

Performance metadata:

    tasks/metadata/910b2.json

The three AgentForge operators are covered by the official 910B2 metadata:

- swi_glu
- mhc_sinkhorn
- engram_gate_fusion

---

## 11. Daily Environment Initialization

After the environment has been created once, a new shell only needs:

    source /workspace/agentforge-env/bin/activate
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

Then verify the NPU if necessary:

    npu-smi info

Typical AgentForge development starts with:

    cd /workspace/AgentForge

CANN Bench evaluation starts with:

    cd /workspace/cann-bench

---

## 12. Reference Toolchain Summary

Reference development stack:

    Ascend 910B2
    Ubuntu 22.04.5 LTS
    aarch64
    CANN 9.1.0
    Bisheng 15.0.5
    Python 3.12.13
    torch 2.10.0
    torch_npu 2.10.0.post4
    NumPy 2.2.6
    PyYAML 6.0.3
    transformers 5.19.0
    accelerate 1.15.0
    safetensors 0.8.0
    pillow 12.3.0
    Qwen3.5-4B BF16 (reference model weights)

The official contest environment takes precedence over local environments if
there is any difference.
