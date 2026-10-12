#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

from inspector import inspect_candidate
from iteration import load_metadata
from workspace import (
    candidate_dir,
    workspace_dir,
)


# ----------------------------------------------------------------------
# Context policy
# ----------------------------------------------------------------------

FULL_FILE_LINE_LIMIT = 220

HEAD_LINES = 70
TAIL_LINES = 50

ANCHOR_RADIUS = 18
MAX_ANCHORS_PER_FILE = 10


ANCHOR_PATTERNS = [
    r"__global__",
    r"__aicore__",
    r"\bInit\s*\(",
    r"\bProcess\s*\(",
    r"\bProcessOneItem\s*\(",
    r"\blaunch_",
    r"\bcalc_.*tiling",
    r"TORCH_LIBRARY",
    r"TORCH_LIBRARY_IMPL",
    r"register_direct_launch_op",
    r"DataCopy",
    r"Reduce",
    r"Swish",
    r"Exp\s*\(",
    r"Mul\s*\(",
    r"Add\s*\(",
    r"GetValue",
    r"SetValue",
]


def merge_intervals(intervals, total_lines):
    if not intervals:
        return []

    normalized = []

    for start, end in intervals:
        start = max(1, start)
        end = min(total_lines, end)

        if start <= end:
            normalized.append(
                (start, end)
            )

    normalized.sort()

    merged = []

    for start, end in normalized:
        if not merged:
            merged.append(
                [start, end]
            )
            continue

        prev_start, prev_end = (
            merged[-1]
        )

        if start <= prev_end + 1:
            merged[-1][1] = max(
                prev_end,
                end,
            )
        else:
            merged.append(
                [start, end]
            )

    return [
        (start, end)
        for start, end in merged
    ]


def select_intervals(lines):
    total = len(lines)

    if total <= FULL_FILE_LINE_LIMIT:
        return [
            (1, total)
        ]

    intervals = [
        (1, min(HEAD_LINES, total)),
        (
            max(
                1,
                total - TAIL_LINES + 1,
            ),
            total,
        ),
    ]

    anchors = []

    combined = re.compile(
        "|".join(
            f"(?:{pattern})"
            for pattern in ANCHOR_PATTERNS
        )
    )

    for index, line in enumerate(
        lines,
        start=1,
    ):
        if combined.search(line):
            anchors.append(index)

    # Spread selected anchors through the file rather than
    # taking only the first matches.
    if len(anchors) > MAX_ANCHORS_PER_FILE:
        if MAX_ANCHORS_PER_FILE == 1:
            anchors = [
                anchors[0]
            ]
        else:
            selected = []

            last_index = (
                len(anchors) - 1
            )

            for i in range(
                MAX_ANCHORS_PER_FILE
            ):
                pos = round(
                    i
                    * last_index
                    / (
                        MAX_ANCHORS_PER_FILE
                        - 1
                    )
                )

                selected.append(
                    anchors[pos]
                )

            anchors = sorted(
                set(selected)
            )

    for line_number in anchors:
        intervals.append(
            (
                line_number
                - ANCHOR_RADIUS,
                line_number
                + ANCHOR_RADIUS,
            )
        )

    return merge_intervals(
        intervals,
        total,
    )


def render_excerpt(
    relative_path,
    candidate,
):
    path = (
        candidate
        / relative_path
    )

    try:
        text = path.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        return (
            f"## `{relative_path}`\n\n"
            "_Binary or non-UTF-8 file omitted._\n"
        )

    lines = text.splitlines()

    intervals = select_intervals(
        lines
    )

    suffix = path.suffix.lower()

    if suffix in {
        ".py",
    }:
        language = "python"

    elif suffix in {
        ".cpp",
        ".cc",
        ".c",
        ".h",
        ".hpp",
    }:
        language = "cpp"

    elif suffix == ".cmake":
        language = "cmake"

    elif path.name == "CMakeLists.txt":
        language = "cmake"

    else:
        language = "text"

    output = [
        f"## `{relative_path}`",
        "",
        (
            f"Total lines: {len(lines)}"
        ),
        "",
    ]

    for start, end in intervals:
        output.append(
            f"### Lines {start}-{end}"
        )
        output.append("")
        output.append(
            f"~~~{language}"
        )

        for line_number in range(
            start,
            end + 1,
        ):
            line = lines[
                line_number - 1
            ]

            output.append(
                f"{line_number:5d} | {line}"
            )

        output.append("~~~")
        output.append("")

    return "\n".join(output)


def find_previous_result(
    operator,
    run_id,
):
    rounds_root = (
        workspace_dir(
            operator,
            run_id,
        )
        / "rounds"
    )

    if not rounds_root.exists():
        return None

    round_dirs = sorted(
        [
            path
            for path in rounds_root.glob(
                "round_*"
            )
            if path.is_dir()
        ],
        reverse=True,
    )

    for directory in round_dirs:
        result_path = (
            directory
            / "result.json"
        )

        if not result_path.exists():
            continue

        try:
            return json.loads(
                result_path.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError:
            continue

    return None


def choose_priority_files(
    inspection,
):
    result = []

    def add(path):
        if path and path not in result:
            result.append(path)

    # Kernel code is the primary optimization surface.
    for path in inspection[
        "kernel_files"
    ]:
        add(path)

    # Plugin / launch path is useful for workspace,
    # tiling, and dispatch changes.
    for path in inspection[
        "plugin_files"
    ]:
        add(path)

    # Operator-local CMake is useful when adding kernels.
    for item in inspection[
        "cmake_files"
    ]:
        if "/ops/" in item:
            add(item)

    # Public Python interface.
    for path in inspection[
        "python_files"
    ]:
        if (
            "cann_bench/" in path
            or path == "setup.py"
        ):
            add(path)

    # README contains task-specific baseline rationale.
    for item in inspection[
        "files"
    ]:
        if item["path"] == "README.md":
            add("README.md")

    return result


def build_source_context(
    operator,
    run_id,
    output_dir=None,
):
    candidate = candidate_dir(
        operator,
        run_id,
    )

    if not candidate.exists():
        raise FileNotFoundError(
            f"Candidate does not exist: {candidate}"
        )

    metadata = load_metadata(
        operator,
        run_id,
    )

    inspection = inspect_candidate(
        operator,
        run_id,
    )

    previous_result = (
        find_previous_result(
            operator,
            run_id,
        )
    )

    priority_files = (
        choose_priority_files(
            inspection
        )
    )

    if output_dir is None:
        output_dir = (
            workspace_dir(
                operator,
                run_id,
            )
            / "source_context"
        )
    else:
        output_dir = Path(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    context_metadata = {
        "operator": operator,
        "run_id": run_id,
        "candidate_dir": str(
            candidate
        ),
        "best_snapshot": (
            metadata.get(
                "best_snapshot"
            )
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
        "last_decision": (
            metadata.get(
                "last_decision"
            )
        ),
        "priority_files": (
            priority_files
        ),
        "previous_result": (
            previous_result
        ),
        "inspection_summary": (
            inspection["summary"]
        ),
    }

    json_path = (
        output_dir
        / "source_context.json"
    )

    json_path.write_text(
        json.dumps(
            context_metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    md = [
        "# AgentForge Source Context",
        "",
        "This file is generated by the Harness.",
        "",
        "The Agent should use it as evidence for",
        "analysis and optimization planning.",
        "",
        "## Current Best",
        "",
        f"- Operator: `{operator}`",
        f"- Run ID: `{run_id}`",
        (
            "- Best snapshot: "
            f"`{metadata.get('best_snapshot')}`"
        ),
        (
            "- Overall score: "
            f"`{metadata.get('best_overall_score')}`"
        ),
        (
            "- Average speedup: "
            f"`{metadata.get('best_avg_speedup')}x`"
        ),
        "",
        "### Best Metrics",
        "",
        "~~~json",
        json.dumps(
            metadata.get(
                "best_metrics"
            ),
            indent=2,
            ensure_ascii=False,
        ),
        "~~~",
        "",
    ]

    if previous_result is not None:
        md.extend(
            [
                "## Previous Round Feedback",
                "",
                "~~~json",
                json.dumps(
                    {
                        "round_id": (
                            previous_result.get(
                                "round_id"
                            )
                        ),
                        "decision": (
                            previous_result.get(
                                "decision"
                            )
                        ),
                        "best_overall_score": (
                            previous_result.get(
                                "best_overall_score"
                            )
                        ),
                        "best_avg_speedup": (
                            previous_result.get(
                                "best_avg_speedup"
                            )
                        ),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "~~~",
                "",
            ]
        )

    md.extend(
        [
            "## Candidate Structure",
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
            "",
            "### Priority Files",
            "",
        ]
    )

    for path in priority_files:
        md.append(
            f"- `{path}`"
        )

    md.extend(
        [
            "",
            "## Optimization Guidance",
            "",
            "The Agent should:",
            "",
            "1. Preserve operator semantics and correctness.",
            "2. Prefer one focused optimization hypothesis per round.",
            "3. Modify only the candidate workspace.",
            "4. Use CANN Bench results as the only performance evidence.",
            "5. Avoid hard-coded evaluation outputs or CPU fallback.",
            "6. Keep changes explainable in `analysis.md` and `plan.json`.",
            "",
            "# Source Excerpts",
            "",
        ]
    )

    for relative_path in priority_files:
        md.append(
            render_excerpt(
                relative_path,
                candidate,
            )
        )

    markdown_path = (
        output_dir
        / "SOURCE_CONTEXT.md"
    )

    markdown_path.write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("SOURCE CONTEXT BUILT")
    print("=" * 72)
    print(
        f"operator       : {operator}"
    )
    print(
        f"run_id         : {run_id}"
    )
    print(
        f"priority files : {len(priority_files)}"
    )
    print(
        f"context json   : {json_path}"
    )
    print(
        f"context md     : {markdown_path}"
    )

    return (
        context_metadata,
        json_path,
        markdown_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build source context for an "
            "AgentForge optimization Agent"
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

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    args = parser.parse_args()

    build_source_context(
        args.operator,
        args.run_id,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
