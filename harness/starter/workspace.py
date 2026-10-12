#!/usr/bin/env python3

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("operators.yaml")
WORKSPACE_ROOT = Path(__file__).with_name("workspaces")


IGNORE_PATTERNS = shutil.ignore_patterns(
    "build",
    "dist",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.egg-info",
)


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def operator_source(operator):
    config = load_config()

    if operator not in config["operators"]:
        raise ValueError(
            f"Unknown operator: {operator}"
        )

    return (
        ROOT
        / config["operators"][operator]["source_dir"]
    ).resolve()


def workspace_dir(operator, run_id):
    return (
        WORKSPACE_ROOT
        / operator
        / run_id
    ).resolve()


def candidate_dir(operator, run_id):
    return workspace_dir(
        operator,
        run_id,
    ) / "candidate"


def snapshots_dir(operator, run_id):
    return workspace_dir(
        operator,
        run_id,
    ) / "snapshots"


def metadata_path(operator, run_id):
    return workspace_dir(
        operator,
        run_id,
    ) / "metadata.json"


def copy_source(src, dst):
    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(
        src,
        dst,
        ignore=IGNORE_PATTERNS,
    )


def write_metadata(operator, run_id, data):
    path = metadata_path(
        operator,
        run_id,
    )

    existing = {}

    if path.exists():
        existing = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    existing.update(data)

    path.write_text(
        json.dumps(
            existing,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def cmd_init(args):
    baseline = operator_source(
        args.operator
    )

    workdir = workspace_dir(
        args.operator,
        args.run_id,
    )

    candidate = candidate_dir(
        args.operator,
        args.run_id,
    )

    snapshots = snapshots_dir(
        args.operator,
        args.run_id,
    )

    if workdir.exists():
        if not args.force:
            raise RuntimeError(
                f"Workspace already exists: {workdir}\n"
                "Use --force only if you intentionally "
                "want to recreate it."
            )

        shutil.rmtree(workdir)

    workdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshots.mkdir(
        parents=True,
        exist_ok=True,
    )

    copy_source(
        baseline,
        candidate,
    )

    baseline_snapshot = (
        snapshots / "baseline"
    )

    copy_source(
        candidate,
        baseline_snapshot,
    )

    created_at = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    write_metadata(
        args.operator,
        args.run_id,
        {
            "operator": args.operator,
            "run_id": args.run_id,
            "created_at": created_at,
            "baseline_source": str(baseline),
            "candidate_dir": str(candidate),
            "baseline_snapshot": str(
                baseline_snapshot
            ),
            "current_snapshot": "baseline",
        },
    )

    print("Workspace initialized")
    print(f"operator  : {args.operator}")
    print(f"run_id    : {args.run_id}")
    print(f"baseline  : {baseline}")
    print(f"candidate : {candidate}")
    print(
        f"snapshot  : {baseline_snapshot}"
    )


def cmd_snapshot(args):
    candidate = candidate_dir(
        args.operator,
        args.run_id,
    )

    if not candidate.exists():
        raise FileNotFoundError(
            f"Candidate does not exist: {candidate}"
        )

    snapshots = snapshots_dir(
        args.operator,
        args.run_id,
    )

    snapshots.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        snapshots
        / args.name
    )

    if destination.exists():
        if not args.force:
            raise RuntimeError(
                f"Snapshot already exists: {destination}"
            )

        shutil.rmtree(destination)

    copy_source(
        candidate,
        destination,
    )

    write_metadata(
        args.operator,
        args.run_id,
        {
            "last_snapshot": args.name,
            "last_snapshot_time": (
                datetime.now()
                .astimezone()
                .isoformat()
            ),
        },
    )

    print("Snapshot created")
    print(f"name : {args.name}")
    print(f"path : {destination}")


def cmd_restore(args):
    candidate = candidate_dir(
        args.operator,
        args.run_id,
    )

    source = (
        snapshots_dir(
            args.operator,
            args.run_id,
        )
        / args.name
    )

    if not source.exists():
        raise FileNotFoundError(
            f"Snapshot does not exist: {source}"
        )

    copy_source(
        source,
        candidate,
    )

    write_metadata(
        args.operator,
        args.run_id,
        {
            "current_snapshot": args.name,
            "last_restore_time": (
                datetime.now()
                .astimezone()
                .isoformat()
            ),
        },
    )

    print("Candidate restored")
    print(f"snapshot  : {args.name}")
    print(f"candidate : {candidate}")


def cmd_show(args):
    workdir = workspace_dir(
        args.operator,
        args.run_id,
    )

    candidate = candidate_dir(
        args.operator,
        args.run_id,
    )

    snapshots = snapshots_dir(
        args.operator,
        args.run_id,
    )

    print(f"workspace : {workdir}")
    print(f"candidate : {candidate}")

    if metadata_path(
        args.operator,
        args.run_id,
    ).exists():
        print()
        print("metadata:")
        print(
            metadata_path(
                args.operator,
                args.run_id,
            ).read_text(
                encoding="utf-8"
            )
        )

    print()
    print("snapshots:")

    if snapshots.exists():
        for item in sorted(
            snapshots.iterdir()
        ):
            if item.is_dir():
                print(
                    f"  - {item.name}"
                )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "AgentForge candidate workspace manager"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    init_parser = subparsers.add_parser(
        "init"
    )

    init_parser.add_argument(
        "--operator",
        required=True,
        choices=[
            "swi_glu",
            "mhc_sinkhorn",
            "engram_gate_fusion",
        ],
    )

    init_parser.add_argument(
        "--run-id",
        required=True,
    )

    init_parser.add_argument(
        "--force",
        action="store_true",
    )

    init_parser.set_defaults(
        func=cmd_init
    )

    snapshot_parser = (
        subparsers.add_parser(
            "snapshot"
        )
    )

    snapshot_parser.add_argument(
        "--operator",
        required=True,
        choices=[
            "swi_glu",
            "mhc_sinkhorn",
            "engram_gate_fusion",
        ],
    )

    snapshot_parser.add_argument(
        "--run-id",
        required=True,
    )

    snapshot_parser.add_argument(
        "--name",
        required=True,
    )

    snapshot_parser.add_argument(
        "--force",
        action="store_true",
    )

    snapshot_parser.set_defaults(
        func=cmd_snapshot
    )

    restore_parser = (
        subparsers.add_parser(
            "restore"
        )
    )

    restore_parser.add_argument(
        "--operator",
        required=True,
        choices=[
            "swi_glu",
            "mhc_sinkhorn",
            "engram_gate_fusion",
        ],
    )

    restore_parser.add_argument(
        "--run-id",
        required=True,
    )

    restore_parser.add_argument(
        "--name",
        required=True,
    )

    restore_parser.set_defaults(
        func=cmd_restore
    )

    show_parser = subparsers.add_parser(
        "show"
    )

    show_parser.add_argument(
        "--operator",
        required=True,
        choices=[
            "swi_glu",
            "mhc_sinkhorn",
            "engram_gate_fusion",
        ],
    )

    show_parser.add_argument(
        "--run-id",
        required=True,
    )

    show_parser.set_defaults(
        func=cmd_show
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
