# AgentForge Contestant Interface

AgentForge evaluates an autonomous operator-optimization system rather
than a manually optimized final kernel.

Contestants may redesign the Harness, create their own Skills, or use
Skills from public repositories subject to the competition rules.

## 1. Standard Entry Point

The submitted system must provide:

~~~bash
./run_agent.sh \
  --operator <operator> \
  --budget <rounds> \
  --run-id <run_id>
~~~

Supported operators are:

~~~text
swi_glu
mhc_sinkhorn
engram_gate_fusion
~~~

`budget` is the maximum number of autonomous optimization rounds.

## 2. Frozen Starting Point

Each final run starts from the organizer-provided Starter Baseline:

~~~text
operators/<operator>/baseline
~~~

The Agent must not begin from a manually optimized final kernel.

The Harness should copy the baseline into an isolated candidate workspace
before optimization.

## 3. Contestant-Modifiable Components

Contestants may modify or replace:

- Harness architecture;
- prompting strategy;
- Agent orchestration;
- context construction;
- memory;
- reflection;
- planning;
- patching strategy;
- profiler integration;
- Skill design;
- Skill retrieval;
- Skill routing;
- public Skills, subject to their licenses.

## 4. Skills

Contestants may:

- write Skills themselves;
- adapt public Skills;
- combine multiple Skills;
- dynamically select among frozen local Skills.

All Skills used in the final evaluation must be present in the frozen
submission package before the run begins.

The final run must not download new Skills from the Internet.

External Skills must record:

- source repository;
- revision or commit;
- license;
- whether they were modified.

## 5. Prohibited Behavior

The autonomous system must not:

- modify the frozen CANN Bench evaluator;
- modify the organizer Starter Baseline in place;
- use CPU fallback for target computation;
- cache or hard-code evaluation outputs;
- embed manually optimized final kernels selected by operator identity;
- bypass correctness evaluation;
- manipulate benchmark or profiler results;
- retrieve a prewritten final answer during the final run.

## 6. Required Agent Round Artifacts

For each optimization round, the submission should preserve enough
information to reconstruct the decision process.

At minimum:

~~~text
context.json
analysis.md
plan.json
result.json
~~~

Equivalent structured formats are acceptable if clearly documented.

## 7. Required Final Artifacts

At the end of a run, the system must expose:

~~~text
best candidate source
best score / performance result
run history
Harness configuration
Skill package / manifest
~~~

The optimized candidate will be independently evaluated by the frozen
CANN Bench evaluator.

## 8. Reproducibility

The final submitted Harness and Skill package will be frozen before
official evaluation.

No manual kernel modification is allowed during the final autonomous run.
