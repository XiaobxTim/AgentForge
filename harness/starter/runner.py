#!/usr/bin/env python3

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from iteration import (
    load_metadata,
    restore_snapshot,
)
from round import (
    finalize_round,
    next_round_id,
    prepare_round,
    round_dir,
)
from source_context import (
    build_source_context,
)
from workspace import (
    candidate_dir,
    copy_source,
    snapshots_dir,
    workspace_dir,
)


ROOT = Path(__file__).resolve().parents[2]
STARTER_DIR = Path(__file__).resolve().parent

DEFAULT_AGENT = (
    STARTER_DIR
    / "agent_entrypoint.sh"
)

DEFAULT_SKILLS = (
    STARTER_DIR
    / "skills"
)


def run_control_command(
    command,
):
    print()
    print("$ " + " ".join(
        shlex.quote(str(x))
        for x in command
    ))
    print()

    result = subprocess.run(
        command,
        cwd=ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Harness control command failed "
            f"with exit code {result.returncode}"
        )


def initialize_run(
    operator,
    run_id,
    force=False,
):
    command = [
        sys.executable,
        str(
            STARTER_DIR
            / "workspace.py"
        ),
        "init",
        "--operator",
        operator,
        "--run-id",
        run_id,
    ]

    if force:
        command.append(
            "--force"
        )

    run_control_command(
        command
    )

    run_control_command(
        [
            sys.executable,
            str(
                STARTER_DIR
                / "iteration.py"
            ),
            "init-best",
            "--operator",
            operator,
            "--run-id",
            run_id,
        ]
    )


def run_agent(
    operator,
    run_id,
    rid,
    agent_cmd,
    skills_dir,
    timeout_seconds,
):
    target = round_dir(
        operator,
        run_id,
        rid,
    )

    candidate = candidate_dir(
        operator,
        run_id,
    )

    context_path = (
        target
        / "context.json"
    )

    source_context_path = (
        target
        / "SOURCE_CONTEXT.md"
    )

    analysis_path = (
        target
        / "analysis.md"
    )

    plan_path = (
        target
        / "plan.json"
    )

    command = (
        shlex.split(
            agent_cmd
        )
        + [
            "--operator",
            operator,
            "--candidate",
            str(candidate),
            "--context",
            str(context_path),
            "--source-context",
            str(source_context_path),
            "--analysis",
            str(analysis_path),
            "--plan",
            str(plan_path),
            "--skills-dir",
            str(skills_dir),
        ]
    )

    environment = os.environ.copy()

    environment.update(
        {
            "AGENTFORGE_OPERATOR": (
                operator
            ),
            "AGENTFORGE_RUN_ID": (
                run_id
            ),
            "AGENTFORGE_ROUND_ID": (
                str(rid)
            ),
            "AGENTFORGE_CANDIDATE": (
                str(candidate)
            ),
            "AGENTFORGE_CONTEXT": (
                str(context_path)
            ),
            "AGENTFORGE_SOURCE_CONTEXT": (
                str(source_context_path)
            ),
            "AGENTFORGE_ANALYSIS": (
                str(analysis_path)
            ),
            "AGENTFORGE_PLAN": (
                str(plan_path)
            ),
            "AGENTFORGE_SKILLS_DIR": (
                str(skills_dir)
            ),
        }
    )

    print()
    print("=" * 72)
    print(
        f"INVOKE AGENT — ROUND {rid}"
    )
    print("=" * 72)
    print(
        f"command   : {agent_cmd}"
    )
    print(
        f"candidate : {candidate}"
    )
    print(
        f"context   : {context_path}"
    )
    print(
        "source ctx: "
        f"{source_context_path}"
    )
    print(
        f"skills    : {skills_dir}"
    )
    print(
        "timeout   : "
        f"{timeout_seconds}s"
    )
    print()

    started_at = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    timed_out = False
    returncode = None

    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            timeout=timeout_seconds,
        )

        returncode = (
            process.returncode
        )

    except subprocess.TimeoutExpired:
        timed_out = True

    completed_at = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    status = {
        "operator": operator,
        "run_id": run_id,
        "round_id": rid,
        "agent_command": (
            agent_cmd
        ),
        "started_at": started_at,
        "completed_at": (
            completed_at
        ),
        "timeout_seconds": (
            timeout_seconds
        ),
        "timed_out": timed_out,
        "returncode": returncode,
        "success": (
            not timed_out
            and returncode == 0
        ),
    }

    status_path = (
        target
        / "agent_status.json"
    )

    status_path.write_text(
        json.dumps(
            status,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return status


def record_agent_failure(
    operator,
    run_id,
    rid,
    status,
):
    metadata = load_metadata(
        operator,
        run_id,
    )

    best_snapshot = (
        metadata.get(
            "best_snapshot"
        )
    )

    if best_snapshot:
        restore_snapshot(
            operator,
            run_id,
            best_snapshot,
        )

    target = round_dir(
        operator,
        run_id,
        rid,
    )

    if status["timed_out"]:
        decision = (
            "agent_timeout"
        )
    else:
        decision = (
            "agent_failed"
        )

    result = {
        "operator": operator,
        "run_id": run_id,
        "round_id": rid,
        "decision": decision,
        "agent_status": status,
        "best_snapshot": (
            best_snapshot
        ),
        "best_overall_score": (
            metadata.get(
                "best_overall_score"
            )
        ),
        "best_avg_speedup": (
            metadata.get(
                "best_avg_speedup"
            )
        ),
        "completed_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
    }

    result_path = (
        target
        / "result.json"
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
    print("AGENT ROUND FAILED")
    print("=" * 72)
    print(
        f"decision : {decision}"
    )
    print(
        "candidate restored to: "
        f"{best_snapshot}"
    )
    print(
        f"result   : {result_path}"
    )


def export_best(
    operator,
    run_id,
    budget,
    completed_rounds,
):
    metadata = load_metadata(
        operator,
        run_id,
    )

    best_snapshot = (
        metadata.get(
            "best_snapshot"
        )
    )

    if not best_snapshot:
        raise RuntimeError(
            "No best snapshot available."
        )

    source = (
        snapshots_dir(
            operator,
            run_id,
        )
        / best_snapshot
    )

    if not source.exists():
        raise FileNotFoundError(
            f"Best snapshot missing: {source}"
        )

    final_dir = (
        workspace_dir(
            operator,
            run_id,
        )
        / "final"
    )

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        final_dir
        / "best_candidate"
    )

    copy_source(
        source,
        destination,
    )

    summary = {
        "operator": operator,
        "run_id": run_id,
        "budget": budget,
        "completed_rounds": (
            completed_rounds
        ),
        "best_snapshot": (
            best_snapshot
        ),
        "best_overall_score": (
            metadata.get(
                "best_overall_score"
            )
        ),
        "best_avg_speedup": (
            metadata.get(
                "best_avg_speedup"
            )
        ),
        "best_metrics": (
            metadata.get(
                "best_metrics"
            )
        ),
        "best_candidate": (
            str(destination)
        ),
        "exported_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
    }

    summary_path = (
        final_dir
        / "final_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return (
        destination,
        summary_path,
        summary,
    )


def run_optimization(args):
    if (
        args.budget is not None
        and args.budget < 0
    ):
        raise ValueError(
            "budget must be >= 0"
        )

    if args.agent_timeout <= 0:
        raise ValueError(
            "agent-timeout must be > 0"
        )

    skills_dir = Path(
        args.skills_dir
    ).resolve()

    if not skills_dir.exists():
        raise FileNotFoundError(
            f"Skills directory not found: "
            f"{skills_dir}"
        )

    print("=" * 72)
    print("AGENTFORGE AUTONOMOUS RUN")
    print("=" * 72)
    print(
        f"operator      : {args.operator}"
    )
    print(
        f"run_id        : {args.run_id}"
    )

    if args.budget is None:
        print(
            "budget        : unlimited rounds "
            "(external wall-clock applies)"
        )
    else:
        print(
            f"budget        : {args.budget} rounds"
        )

    print(
        f"agent         : {args.agent_cmd}"
    )
    print(
        f"agent timeout : "
        f"{args.agent_timeout}s"
    )
    print(
        f"skills        : {skills_dir}"
    )

    initialize_run(
        args.operator,
        args.run_id,
        force=args.force,
    )

    completed_rounds = 0

    while (
        args.budget is None
        or completed_rounds < args.budget
    ):
        rid = next_round_id(
            args.operator,
            args.run_id,
        )

        print()
        print("#" * 72)
        print(
            f"# OPTIMIZATION ROUND {rid}"
        )
        print("#" * 72)
        print()

        prepare_round(
            args.operator,
            args.run_id,
        )

        target = round_dir(
            args.operator,
            args.run_id,
            rid,
        )

        build_source_context(
            args.operator,
            args.run_id,
            output_dir=target,
        )

        status = run_agent(
            args.operator,
            args.run_id,
            rid,
            args.agent_cmd,
            skills_dir,
            args.agent_timeout,
        )

        if not status["success"]:
            record_agent_failure(
                args.operator,
                args.run_id,
                rid,
                status,
            )

            print()
            print(
                "Stopping autonomous run "
                "because the Agent failed."
            )
            break

        finalize_round(
            args.operator,
            args.run_id,
            rid,
        )

        completed_rounds += 1

        # Keep a valid best candidate checkpoint after every
        # completed round. This preserves the latest accepted
        # result if an external Final watchdog later terminates
        # the autonomous run.
        export_best(
            args.operator,
            args.run_id,
            args.budget,
            completed_rounds,
        )

    (
        best_candidate,
        summary_path,
        summary,
    ) = export_best(
        args.operator,
        args.run_id,
        args.budget,
        completed_rounds,
    )

    print()
    print("=" * 72)
    print("AUTONOMOUS RUN COMPLETE")
    print("=" * 72)
    print(
        "completed rounds : "
        f"{completed_rounds}"
    )
    print(
        "best snapshot    : "
        f"{summary['best_snapshot']}"
    )
    print(
        "best score       : "
        f"{summary['best_overall_score']}"
    )
    print(
        "best speedup     : "
        f"{summary['best_avg_speedup']}x"
    )
    print(
        "best candidate   : "
        f"{best_candidate}"
    )
    print(
        "summary          : "
        f"{summary_path}"
    )
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run an AgentForge autonomous "
            "operator optimization session"
        )
    )

    parser.add_argument(
        "--operator",
        required=True,
        choices=[
            "swi_glu",
            "mhc_sinkhorn",
            "engram_gate_fusion",
        ],
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help=(
            "Optional maximum number of autonomous "
            "optimization rounds. If omitted, the "
            "Starter Harness has no round limit and "
            "the Final wall-clock limit is expected "
            "to be enforced externally."
        ),
    )

    parser.add_argument(
        "--run-id",
        required=True,
    )

    parser.add_argument(
        "--agent-cmd",
        default=str(
            DEFAULT_AGENT
        ),
        help=(
            "Command used to invoke the "
            "contestant Agent."
        ),
    )

    parser.add_argument(
        "--skills-dir",
        default=str(
            DEFAULT_SKILLS
        ),
    )

    parser.add_argument(
        "--agent-timeout",
        type=int,
        default=1800,
        help=(
            "Maximum Agent execution time "
            "per round in seconds."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recreate an existing run "
            "workspace."
        ),
    )

    args = parser.parse_args()

    run_optimization(
        args
    )


if __name__ == "__main__":
    main()
