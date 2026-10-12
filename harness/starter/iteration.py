#!/usr/bin/env python3

import argparse
import json
from datetime import datetime
from pathlib import Path

from evaluator import run_evaluation
from workspace import (
    candidate_dir,
    copy_source,
    metadata_path,
    snapshots_dir,
    workspace_dir,
    write_metadata,
)


def load_metadata(operator, run_id):
    path = metadata_path(
        operator,
        run_id,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Workspace metadata not found: {path}\n"
            "Run workspace.py init first."
        )

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def metric_pair(result):
    metrics = result["metrics"]

    score = metrics.get(
        "overall_score"
    )

    speedup = metrics.get(
        "avg_speedup"
    )

    if score is None:
        raise RuntimeError(
            "Evaluation result does not contain "
            "overall_score."
        )

    if speedup is None:
        speedup = 0.0

    return (
        float(score),
        float(speedup),
    )


def is_better(
    candidate_score,
    candidate_speedup,
    best_score,
    best_speedup,
):
    eps = 1e-12

    if candidate_score > best_score + eps:
        return True

    if abs(
        candidate_score - best_score
    ) <= eps:
        if (
            candidate_speedup
            > best_speedup + eps
        ):
            return True

    return False


def restore_snapshot(
    operator,
    run_id,
    snapshot_name,
):
    source = (
        snapshots_dir(
            operator,
            run_id,
        )
        / snapshot_name
    )

    if not source.exists():
        raise FileNotFoundError(
            f"Snapshot does not exist: {source}"
        )

    destination = candidate_dir(
        operator,
        run_id,
    )

    copy_source(
        source,
        destination,
    )

    write_metadata(
        operator,
        run_id,
        {
            "current_snapshot": snapshot_name,
            "last_restore_time": (
                datetime.now()
                .astimezone()
                .isoformat()
            ),
        },
    )


def save_snapshot(
    operator,
    run_id,
    snapshot_name,
):
    source = candidate_dir(
        operator,
        run_id,
    )

    destination = (
        snapshots_dir(
            operator,
            run_id,
        )
        / snapshot_name
    )

    copy_source(
        source,
        destination,
    )

    return destination


def write_iteration_record(
    operator,
    run_id,
    iteration,
    record,
):
    directory = (
        workspace_dir(
            operator,
            run_id,
        )
        / "iterations"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        directory
        / f"iter_{iteration:04d}.json"
    )

    path.write_text(
        json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path


def initialize_best(
    operator,
    run_id,
):
    metadata = load_metadata(
        operator,
        run_id,
    )

    baseline_snapshot = (
        snapshots_dir(
            operator,
            run_id,
        )
        / "baseline"
    )

    if not baseline_snapshot.exists():
        raise FileNotFoundError(
            "Baseline snapshot is missing."
        )

    print("=" * 72)
    print("INITIALIZE BEST CANDIDATE")
    print("=" * 72)
    print(f"operator : {operator}")
    print(f"run_id   : {run_id}")
    print()
    print(
        "Restoring candidate from "
        "baseline snapshot..."
    )

    restore_snapshot(
        operator,
        run_id,
        "baseline",
    )

    print()
    print("Step 1/2: correctness")
    print()

    correctness = run_evaluation(
        operator,
        "correctness",
        run_id,
    )

    if (
        not correctness["success"]
        or not correctness[
            "correctness_passed"
        ]
    ):
        raise RuntimeError(
            "Baseline snapshot failed correctness. "
            "Cannot initialize best candidate."
        )

    print()
    print("Step 2/2: performance")
    print()

    performance = run_evaluation(
        operator,
        "performance",
        run_id,
    )

    if (
        not performance["success"]
        or not performance[
            "correctness_passed"
        ]
    ):
        raise RuntimeError(
            "Baseline snapshot failed full "
            "performance evaluation."
        )

    score, speedup = metric_pair(
        performance
    )

    now = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    write_metadata(
        operator,
        run_id,
        {
            "iteration": 0,
            "best_snapshot": "baseline",
            "current_snapshot": "baseline",
            "best_overall_score": score,
            "best_avg_speedup": speedup,
            "best_metrics": performance[
                "metrics"
            ],
            "baseline_metrics": performance[
                "metrics"
            ],
            "best_updated_at": now,
        },
    )

    record = {
        "type": "initialize_best",
        "operator": operator,
        "run_id": run_id,
        "timestamp": now,
        "correctness": correctness,
        "performance": performance,
        "best_snapshot": "baseline",
        "best_overall_score": score,
        "best_avg_speedup": speedup,
    }

    directory = (
        workspace_dir(
            operator,
            run_id,
        )
        / "iterations"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    init_path = (
        directory
        / "initial_best.json"
    )

    init_path.write_text(
        json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("BEST INITIALIZED")
    print("=" * 72)
    print(
        f"snapshot      : baseline"
    )
    print(
        f"overall score : {score}"
    )
    print(
        f"avg speedup   : {speedup}x"
    )


def judge_candidate(
    operator,
    run_id,
):
    metadata = load_metadata(
        operator,
        run_id,
    )

    if "best_snapshot" not in metadata:
        raise RuntimeError(
            "Best candidate is not initialized.\n"
            "Run:\n"
            "  iteration.py init-best ..."
        )

    iteration = (
        int(
            metadata.get(
                "iteration",
                0,
            )
        )
        + 1
    )

    best_snapshot = metadata[
        "best_snapshot"
    ]

    best_score = float(
        metadata[
            "best_overall_score"
        ]
    )

    best_speedup = float(
        metadata.get(
            "best_avg_speedup",
            0.0,
        )
    )

    timestamp = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    print("=" * 72)
    print(
        f"AGENTFORGE ITERATION {iteration}"
    )
    print("=" * 72)
    print(f"operator       : {operator}")
    print(f"run_id         : {run_id}")
    print(f"current best   : {best_snapshot}")
    print(f"best score     : {best_score}")
    print(
        f"best speedup   : {best_speedup}x"
    )
    print()

    # --------------------------------------------------------
    # Stage 1: correctness gate
    # --------------------------------------------------------

    print("Stage 1: correctness gate")
    print()

    correctness = run_evaluation(
        operator,
        "correctness",
        run_id,
    )

    correctness_ok = (
        correctness["success"]
        and correctness[
            "correctness_passed"
        ]
    )

    if not correctness_ok:
        print()
        print(
            "REJECT: correctness failed."
        )
        print(
            f"Rolling back to {best_snapshot}..."
        )

        restore_snapshot(
            operator,
            run_id,
            best_snapshot,
        )

        record = {
            "iteration": iteration,
            "timestamp": timestamp,
            "decision": (
                "reject_correctness"
            ),
            "best_before": {
                "snapshot": best_snapshot,
                "overall_score": best_score,
                "avg_speedup": best_speedup,
            },
            "correctness": correctness,
            "performance": None,
            "restored_snapshot": (
                best_snapshot
            ),
        }

        path = write_iteration_record(
            operator,
            run_id,
            iteration,
            record,
        )

        write_metadata(
            operator,
            run_id,
            {
                "iteration": iteration,
                "last_decision": (
                    "reject_correctness"
                ),
                "last_iteration_record": (
                    str(path)
                ),
            },
        )

        print()
        print(
            "Candidate rejected and "
            "rollback completed."
        )

        return

    # --------------------------------------------------------
    # Stage 2: performance gate
    # --------------------------------------------------------

    print()
    print(
        "Correctness passed."
    )
    print()
    print(
        "Stage 2: performance evaluation"
    )
    print()

    performance = run_evaluation(
        operator,
        "performance",
        run_id,
    )

    performance_ok = (
        performance["success"]
        and performance[
            "correctness_passed"
        ]
    )

    if not performance_ok:
        print()
        print(
            "REJECT: full evaluation failed."
        )
        print(
            f"Rolling back to {best_snapshot}..."
        )

        restore_snapshot(
            operator,
            run_id,
            best_snapshot,
        )

        record = {
            "iteration": iteration,
            "timestamp": timestamp,
            "decision": (
                "reject_performance_run"
            ),
            "best_before": {
                "snapshot": best_snapshot,
                "overall_score": best_score,
                "avg_speedup": best_speedup,
            },
            "correctness": correctness,
            "performance": performance,
            "restored_snapshot": (
                best_snapshot
            ),
        }

        path = write_iteration_record(
            operator,
            run_id,
            iteration,
            record,
        )

        write_metadata(
            operator,
            run_id,
            {
                "iteration": iteration,
                "last_decision": (
                    "reject_performance_run"
                ),
                "last_iteration_record": (
                    str(path)
                ),
            },
        )

        return

    candidate_score, candidate_speedup = (
        metric_pair(
            performance
        )
    )

    better = is_better(
        candidate_score,
        candidate_speedup,
        best_score,
        best_speedup,
    )

    if better:
        new_snapshot = (
            f"best_{iteration:04d}"
        )

        snapshot_path = save_snapshot(
            operator,
            run_id,
            new_snapshot,
        )

        write_metadata(
            operator,
            run_id,
            {
                "iteration": iteration,
                "best_snapshot": new_snapshot,
                "current_snapshot": new_snapshot,
                "best_overall_score": (
                    candidate_score
                ),
                "best_avg_speedup": (
                    candidate_speedup
                ),
                "best_metrics": performance[
                    "metrics"
                ],
                "best_updated_at": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "last_decision": "accept",
            },
        )

        decision = "accept"

        print()
        print("=" * 72)
        print("ACCEPT NEW BEST")
        print("=" * 72)
        print(
            f"old score : {best_score}"
        )
        print(
            f"new score : {candidate_score}"
        )
        print(
            f"old speed : {best_speedup}x"
        )
        print(
            f"new speed : {candidate_speedup}x"
        )
        print(
            f"snapshot  : {snapshot_path}"
        )

        restored_snapshot = None

    else:
        decision = "reject_not_better"

        print()
        print("=" * 72)
        print("REJECT: NOT BETTER")
        print("=" * 72)
        print(
            f"best score      : {best_score}"
        )
        print(
            f"candidate score : {candidate_score}"
        )
        print(
            f"best speed      : {best_speedup}x"
        )
        print(
            f"candidate speed : {candidate_speedup}x"
        )
        print()
        print(
            f"Rolling back to {best_snapshot}..."
        )

        restore_snapshot(
            operator,
            run_id,
            best_snapshot,
        )

        restored_snapshot = (
            best_snapshot
        )

        write_metadata(
            operator,
            run_id,
            {
                "iteration": iteration,
                "last_decision": (
                    decision
                ),
            },
        )

    record = {
        "iteration": iteration,
        "timestamp": timestamp,
        "decision": decision,
        "best_before": {
            "snapshot": best_snapshot,
            "overall_score": best_score,
            "avg_speedup": best_speedup,
        },
        "candidate": {
            "overall_score": (
                candidate_score
            ),
            "avg_speedup": (
                candidate_speedup
            ),
        },
        "correctness": correctness,
        "performance": performance,
        "restored_snapshot": (
            restored_snapshot
        ),
    }

    path = write_iteration_record(
        operator,
        run_id,
        iteration,
        record,
    )

    write_metadata(
        operator,
        run_id,
        {
            "iteration": iteration,
            "last_iteration_record": (
                str(path)
            ),
        },
    )

    print()
    print(
        f"Iteration record: {path}"
    )


def show_best(
    operator,
    run_id,
):
    metadata = load_metadata(
        operator,
        run_id,
    )

    print("=" * 72)
    print("CURRENT BEST")
    print("=" * 72)
    print(
        f"operator       : {operator}"
    )
    print(
        f"run_id         : {run_id}"
    )
    print(
        "iteration      : "
        f"{metadata.get('iteration')}"
    )
    print(
        "best snapshot  : "
        f"{metadata.get('best_snapshot')}"
    )
    print(
        "overall score  : "
        f"{metadata.get('best_overall_score')}"
    )
    print(
        "average speedup: "
        f"{metadata.get('best_avg_speedup')}x"
    )
    print(
        "last decision  : "
        f"{metadata.get('last_decision')}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "AgentForge candidate selection "
            "and rollback controller"
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    for command in [
        "init-best",
        "judge",
        "show-best",
    ]:
        sub = subparsers.add_parser(
            command
        )

        sub.add_argument(
            "--operator",
            required=True,
            choices=[
                "swi_glu",
                "mhc_sinkhorn",
                "engram_gate_fusion",
            ],
        )

        sub.add_argument(
            "--run-id",
            required=True,
        )

    args = parser.parse_args()

    if args.command == "init-best":
        initialize_best(
            args.operator,
            args.run_id,
        )

    elif args.command == "judge":
        judge_candidate(
            args.operator,
            args.run_id,
        )

    elif args.command == "show-best":
        show_best(
            args.operator,
            args.run_id,
        )


if __name__ == "__main__":
    main()
