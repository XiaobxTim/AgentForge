#!/usr/bin/env python3

import argparse
import json
from datetime import datetime
from pathlib import Path

from workspace import (
    candidate_dir,
    workspace_dir,
)


SOURCE_SUFFIXES = {
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".py",
    ".cmake",
}

SPECIAL_FILES = {
    "CMakeLists.txt",
    "README.md",
    "setup.py",
    "pyproject.toml",
}


def is_relevant_file(path: Path):
    if path.name in SPECIAL_FILES:
        return True

    return path.suffix in SOURCE_SUFFIXES


def inspect_candidate(
    operator,
    run_id,
):
    candidate = candidate_dir(
        operator,
        run_id,
    )

    if not candidate.exists():
        raise FileNotFoundError(
            f"Candidate does not exist: {candidate}"
        )

    files = []

    for path in sorted(
        candidate.rglob("*")
    ):
        if not path.is_file():
            continue

        relative = path.relative_to(
            candidate
        )

        parts = set(relative.parts)

        if parts & {
            "build",
            "dist",
            "__pycache__",
        }:
            continue

        if not is_relevant_file(path):
            continue

        try:
            text = path.read_text(
                encoding="utf-8"
            )

            lines = text.splitlines()

            line_count = len(lines)

        except UnicodeDecodeError:
            line_count = None

        files.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "lines": line_count,
            }
        )

    kernel_files = [
        item["path"]
        for item in files
        if (
            "/op_kernel/" in
            f"/{item['path']}"
        )
    ]

    plugin_files = [
        item["path"]
        for item in files
        if (
            "/op_plugin/" in
            f"/{item['path']}"
        )
    ]

    python_files = [
        item["path"]
        for item in files
        if item["path"].endswith(
            ".py"
        )
    ]

    cmake_files = [
        item["path"]
        for item in files
        if (
            item["path"].endswith(
                "CMakeLists.txt"
            )
            or item["path"].endswith(
                ".cmake"
            )
        )
    ]

    inspection = {
        "operator": operator,
        "run_id": run_id,
        "candidate_dir": str(
            candidate
        ),
        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "summary": {
            "relevant_file_count": (
                len(files)
            ),
            "kernel_file_count": (
                len(kernel_files)
            ),
            "plugin_file_count": (
                len(plugin_files)
            ),
            "python_file_count": (
                len(python_files)
            ),
        },
        "kernel_files": kernel_files,
        "plugin_files": plugin_files,
        "python_files": python_files,
        "cmake_files": cmake_files,
        "files": files,
    }

    return inspection


def write_inspection(
    operator,
    run_id,
    output_dir=None,
):
    inspection = inspect_candidate(
        operator,
        run_id,
    )

    if output_dir is None:
        output_dir = (
            workspace_dir(
                operator,
                run_id,
            )
            / "inspection"
        )
    else:
        output_dir = Path(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        / "inspection.json"
    )

    json_path.write_text(
        json.dumps(
            inspection,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    overview_path = (
        output_dir
        / "SOURCE_OVERVIEW.md"
    )

    lines = [
        "# Candidate Source Overview",
        "",
        f"- Operator: `{operator}`",
        f"- Run ID: `{run_id}`",
        (
            "- Candidate: "
            f"`{inspection['candidate_dir']}`"
        ),
        "",
        "## Summary",
        "",
        (
            "- Relevant files: "
            f"{inspection['summary']['relevant_file_count']}"
        ),
        (
            "- Kernel files: "
            f"{inspection['summary']['kernel_file_count']}"
        ),
        (
            "- Plugin files: "
            f"{inspection['summary']['plugin_file_count']}"
        ),
        (
            "- Python files: "
            f"{inspection['summary']['python_file_count']}"
        ),
        "",
        "## Kernel Files",
        "",
    ]

    for path in inspection[
        "kernel_files"
    ]:
        lines.append(
            f"- `{path}`"
        )

    lines.extend(
        [
            "",
            "## Plugin Files",
            "",
        ]
    )

    for path in inspection[
        "plugin_files"
    ]:
        lines.append(
            f"- `{path}`"
        )

    lines.extend(
        [
            "",
            "## All Relevant Files",
            "",
            "| File | Lines | Bytes |",
            "|---|---:|---:|",
        ]
    )

    for item in inspection["files"]:
        line_count = (
            item["lines"]
            if item["lines"]
            is not None
            else "-"
        )

        lines.append(
            f"| `{item['path']}` "
            f"| {line_count} "
            f"| {item['bytes']} |"
        )

    overview_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("CANDIDATE INSPECTION")
    print("=" * 72)
    print(
        f"operator       : {operator}"
    )
    print(
        f"run_id         : {run_id}"
    )
    print(
        "relevant files : "
        f"{inspection['summary']['relevant_file_count']}"
    )
    print(
        "kernel files   : "
        f"{inspection['summary']['kernel_file_count']}"
    )
    print(
        f"inspection     : {json_path}"
    )
    print(
        f"overview       : {overview_path}"
    )

    return (
        inspection,
        json_path,
        overview_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an AgentForge candidate "
            "workspace"
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
        "--run-id",
        required=True,
    )

    args = parser.parse_args()

    write_inspection(
        args.operator,
        args.run_id,
    )


if __name__ == "__main__":
    main()
