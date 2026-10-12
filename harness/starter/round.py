#!/usr/bin/env python3

import argparse
import json
from datetime import datetime
from pathlib import Path

from iteration import (
    judge_candidate,
    load_metadata,
    restore_snapshot,
)
from workspace import (
    candidate_dir,
    workspace_dir,
)


def rounds_dir(operator, run_id):
    return (
        workspace_dir(operator, run_id)
        / "rounds"
    )


def round_dir(operator, run_id, round_id):
    return (
        rounds_dir(operator, run_id)
        / f"round_{round_id:04d}"
    )


def next_round_id(operator, run_id):
    root = rounds_dir(operator, run_id)

    if not root.exists():
        return 1

    ids = []

    for path in root.glob("round_*"):
        if not path.is_dir():
            continue

        try:
            ids.append(
                int(path.name.split("_")[1])
            )
        except (IndexError, ValueError):
            pass

    return max(ids, default=0) + 1


def build_context(operator, run_id, round_id):
    metadata = load_metadata(
        operator,
        run_id,
    )

    candidate = candidate_dir(
        operator,
        run_id,
    )

    return {
        "protocol_version": "1.0",
        "operator": operator,
        "run_id": run_id,
        "round_id": round_id,
        "candidate_dir": str(candidate),
        "best_snapshot": metadata.get(
            "best_snapshot"
        ),
        "best_overall_score": metadata.get(
            "best_overall_score"
        ),
        "best_avg_speedup": metadata.get(
            "best_avg_speedup"
        ),
        "best_metrics": metadata.get(
            "best_metrics"
        ),
        "constraints": {
            "modify_candidate_only": True,
            "must_preserve_correctness": True,
            "cpu_fallback_forbidden": True,
            "cached_outputs_forbidden": True,
            "hardcoded_eval_outputs_forbidden": True,
            "profiler_manipulation_forbidden": True,
        },
        "required_agent_outputs": [
            "analysis.md",
            "plan.json",
        ],
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
    }


def prepare_round(operator, run_id):
    metadata = load_metadata(
        operator,
        run_id,
    )

    best_snapshot = metadata.get(
        "best_snapshot"
    )

    if not best_snapshot:
        raise RuntimeError(
            "No best snapshot is initialized. "
            "Run iteration.py init-best first."
        )

    rid = next_round_id(
        operator,
        run_id,
    )

    target = round_dir(
        operator,
        run_id,
        rid,
    )

    target.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        f"Restoring candidate from "
        f"{best_snapshot}..."
    )

    restore_snapshot(
        operator,
        run_id,
        best_snapshot,
    )

    context = build_context(
        operator,
        run_id,
        rid,
    )

    context_path = (
        target / "context.json"
    )

    context_path.write_text(
        json.dumps(
            context,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    analysis_path = (
        target / "analysis.md"
    )

    analysis_path.write_text(
        "# Analysis\n\n"
        "<Agent writes analysis here.>\n",
        encoding="utf-8",
    )

    plan_path = (
        target / "plan.json"
    )

    plan_path.write_text(
        json.dumps(
            {
                "summary": "",
                "hypothesis": "",
                "files_to_modify": [],
                "expected_effect": {
                    "correctness": "",
                    "performance": "",
                },
                "risk": "",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("ROUND PREPARED")
    print("=" * 72)
    print(f"round      : {rid}")
    print(
        f"candidate  : "
        f"{context['candidate_dir']}"
    )
    print(
        f"context    : {context_path}"
    )
    print(
        f"analysis   : {analysis_path}"
    )
    print(
        f"plan       : {plan_path}"
    )
    print()
    print(
        "Agent may now inspect and modify "
        "the candidate."
    )


def validate_agent_outputs(
    operator,
    run_id,
    rid,
):
    target = round_dir(
        operator,
        run_id,
        rid,
    )

    if not target.exists():
        raise FileNotFoundError(
            f"Round does not exist: {target}"
        )

    analysis_path = (
        target / "analysis.md"
    )

    plan_path = (
        target / "plan.json"
    )

    if not analysis_path.exists():
        raise RuntimeError(
            "Agent did not produce analysis.md"
        )

    if not plan_path.exists():
        raise RuntimeError(
            "Agent did not produce plan.json"
        )

    plan = json.loads(
        plan_path.read_text(
            encoding="utf-8",
        )
    )

    required = {
        "summary",
        "hypothesis",
        "files_to_modify",
        "expected_effect",
        "risk",
    }

    missing = (
        required
        - set(plan.keys())
    )

    if missing:
        raise RuntimeError(
            "plan.json missing keys: "
            + ", ".join(
                sorted(missing)
            )
        )


def finalize_round(
    operator,
    run_id,
    rid,
):
    target = round_dir(
        operator,
        run_id,
        rid,
    )

    validate_agent_outputs(
        operator,
        run_id,
        rid,
    )

    metadata_before = load_metadata(
        operator,
        run_id,
    )

    iteration_before = int(
        metadata_before.get(
            "iteration",
            0,
        )
    )

    print("=" * 72)
    print(
        f"FINALIZE ROUND {rid}"
    )
    print("=" * 72)

    judge_candidate(
        operator,
        run_id,
    )

    metadata_after = load_metadata(
        operator,
        run_id,
    )

    iteration_after = int(
        metadata_after.get(
            "iteration",
            iteration_before,
        )
    )

    record_path = metadata_after.get(
        "last_iteration_record"
    )

    iteration_record = None

    if record_path:
        path = Path(record_path)

        if path.exists():
            iteration_record = (
                json.loads(
                    path.read_text(
                        encoding="utf-8",
                    )
                )
            )

    result = {
        "operator": operator,
        "run_id": run_id,
        "round_id": rid,
        "iteration_before": (
            iteration_before
        ),
        "iteration_after": (
            iteration_after
        ),
        "decision": metadata_after.get(
            "last_decision"
        ),
        "best_snapshot": metadata_after.get(
            "best_snapshot"
        ),
        "best_overall_score": (
            metadata_after.get(
                "best_overall_score"
            )
        ),
        "best_avg_speedup": (
            metadata_after.get(
                "best_avg_speedup"
            )
        ),
        "iteration_record": (
            iteration_record
        ),
        "completed_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
    }

    result_path = (
        target / "result.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("ROUND RESULT")
    print("=" * 72)
    print(
        f"decision      : "
        f"{result['decision']}"
    )
    print(
        f"best snapshot : "
        f"{result['best_snapshot']}"
    )
    print(
        f"best score    : "
        f"{result['best_overall_score']}"
    )
    print(
        f"best speedup  : "
        f"{result['best_avg_speedup']}x"
    )
    print(
        f"result        : "
        f"{result_path}"
    )


def show_round(
    operator,
    run_id,
    rid,
):
    target = round_dir(
        operator,
        run_id,
        rid,
    )

    if not target.exists():
        raise FileNotFoundError(
            target
        )

    print(f"round: {target}")
    print()

    for name in [
        "context.json",
        "analysis.md",
        "plan.json",
        "result.json",
    ]:
        path = target / name

        print(
            f"{name}: "
            f"{'present' if path.exists() else 'missing'}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "AgentForge optimization round controller"
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    prepare = subparsers.add_parser(
        "prepare"
    )

    prepare.add_argument(
        "--operator",
        required=True,
    )

    prepare.add_argument(
        "--run-id",
        required=True,
    )

    finalize = subparsers.add_parser(
        "finalize"
    )

    finalize.add_argument(
        "--operator",
        required=True,
    )

    finalize.add_argument(
        "--run-id",
        required=True,
    )

    finalize.add_argument(
        "--round",
        type=int,
        required=True,
    )

    show = subparsers.add_parser(
        "show"
    )

    show.add_argument(
        "--operator",
        required=True,
    )

    show.add_argument(
        "--run-id",
        required=True,
    )

    show.add_argument(
        "--round",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_round(
            args.operator,
            args.run_id,
        )

    elif args.command == "finalize":
        finalize_round(
            args.operator,
            args.run_id,
            args.round,
        )

    elif args.command == "show":
        show_round(
            args.operator,
            args.run_id,
            args.round,
        )


if __name__ == "__main__":
    main()
