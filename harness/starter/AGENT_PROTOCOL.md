# AgentForge Starter Agent Protocol

The Agent works only on the candidate workspace.

It must never directly modify:

- `operators/<operator>/baseline`
- CANN Bench
- evaluator code
- Harness control code

Each optimization round follows:

1. Harness restores the current best snapshot.
2. Harness creates a round directory.
3. Agent reads `context.json`.
4. Agent inspects the candidate implementation.
5. Agent writes `analysis.md`.
6. Agent writes `plan.json`.
7. Agent modifies files under the candidate directory.
8. Harness runs correctness evaluation.
9. If correctness passes, Harness runs performance evaluation.
10. Harness accepts the candidate if it improves the current best.
11. Otherwise Harness restores the previous best.
12. Harness writes `result.json`.

## Required Agent Outputs

### analysis.md

Human-readable reasoning about:

- observed bottlenecks;
- relevant kernel structure;
- expected optimization opportunity;
- risks to correctness;
- evidence motivating the modification.

### plan.json

Example:

~~~json
{
  "summary": "Reduce redundant GM traffic in Stage A",
  "hypothesis": "Keeping intermediate values in UB should reduce HBM traffic",
  "files_to_modify": [
    "csrc/ops/example/op_kernel/example_kernel.cpp"
  ],
  "expected_effect": {
    "correctness": "unchanged",
    "performance": "improve"
  },
  "risk": "medium"
}
~~~

The Agent must not fabricate benchmark results.

Only the Harness / CANN Bench evaluator determines correctness and
performance.
