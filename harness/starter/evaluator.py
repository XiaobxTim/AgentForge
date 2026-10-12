#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("operators.yaml")
LOG_DIR = Path(__file__).with_name("logs")


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_report(text: str):
    result = {
        "total_cases": None,
        "passed_cases": None,
        "failed_cases": None,
        "pass_rate": None,
        "overall_score": None,
        "avg_speedup": None,
    }

    patterns = {
        "total_cases": r"\|\s*总用例数\s*\|\s*(\d+)\s*\|",
        "passed_cases": r"\|\s*通过用例数\s*\|\s*(\d+)\s*\|",
        "failed_cases": r"\|\s*失败用例数\s*\|\s*(\d+)\s*\|",
        "pass_rate": r"\|\s*通过率\s*\|\s*([0-9.]+)%",
        "overall_score": r"\|\s*综合得分\s*\|\s*([0-9.]+)\s*\|",
        "avg_speedup": r"\|\s*平均加速比\s*\|\s*([0-9.]+)x\s*\|",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if not m:
            continue

        value = m.group(1)

        if key in {
            "total_cases",
            "passed_cases",
            "failed_cases",
        }:
            result[key] = int(value)
        else:
            result[key] = float(value)

    return result


def run_evaluation(operator: str, mode: str):
    config = load_config()

    operators = config["operators"]

    if operator not in operators:
        raise ValueError(
            f"Unknown operator: {operator}. "
            f"Available: {', '.join(operators.keys())}"
        )

    op_cfg = operators[operator]
    bench_cfg = config["cann_bench"]

    cann_root = Path(bench_cfg["root"])
    source_dir = ROOT / op_cfg["source_dir"]
    task_dir = cann_root / op_cfg["task_dir"]

    script = cann_root / "scripts/run_evaluation.sh"

    if not script.exists():
        raise FileNotFoundError(
            f"CANN Bench evaluator not found: {script}"
        )

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Operator source directory not found: {source_dir}"
        )

    if not task_dir.exists():
        raise FileNotFoundError(
            f"CANN Bench task directory not found: {task_dir}"
        )

    command = [
        str(script),
        "--source-dir",
        str(source_dir),
        "--task-dir",
        str(task_dir),
        "--device-id",
        str(bench_cfg.get("device_id", 0)),
    ]

    if mode == "correctness":
        command.append("--no-perf")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = (
        LOG_DIR
        / f"{operator}_{mode}_{timestamp}.log"
    )

    print("=" * 72)
    print("AgentForge Starter Evaluator")
    print("=" * 72)
    print(f"operator : {operator}")
    print(f"mode     : {mode}")
    print(f"source   : {source_dir}")
    print(f"task     : {task_dir}")
    print(f"log      : {log_path}")
    print("=" * 72)
    print()

    process = subprocess.run(
        command,
        cwd=cann_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output = process.stdout

    print(output)

    log_path.write_text(
        output,
        encoding="utf-8",
    )

    metrics = parse_report(output)

    result = {
        "operator": operator,
        "mode": mode,
        "returncode": process.returncode,
        "success": process.returncode == 0,
        "source_dir": str(source_dir),
        "task_dir": str(task_dir),
        "log_path": str(log_path),
        "metrics": metrics,
    }

    result_path = log_path.with_suffix(".json")

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
    print("STRUCTURED RESULT")
    print("=" * 72)
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AgentForge unified CANN Bench evaluator"
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
        "--mode",
        choices=[
            "correctness",
            "performance",
        ],
        default="correctness",
    )

    args = parser.parse_args()

    result = run_evaluation(
        args.operator,
        args.mode,
    )

    if not result["success"]:
        sys.exit(result["returncode"] or 1)

    metrics = result["metrics"]

    if (
        args.mode == "correctness"
        and metrics["total_cases"] is not None
        and metrics["passed_cases"]
        != metrics["total_cases"]
    ):
        sys.exit(2)


if __name__ == "__main__":
    main()
