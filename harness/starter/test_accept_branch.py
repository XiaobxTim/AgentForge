#!/usr/bin/env python3

import iteration


OPERATOR = "engram_gate_fusion"
RUN_ID = "smoke_accept"


def fake_run_evaluation(
    operator,
    mode,
    run_id=None,
):
    assert operator == OPERATOR
    assert run_id == RUN_ID

    if mode == "correctness":
        return {
            "operator": operator,
            "mode": mode,
            "run_id": run_id,
            "success": True,
            "correctness_passed": True,
            "metrics": {
                "total_cases": 20,
                "passed_cases": 20,
                "failed_cases": 0,
                "pass_rate": 100.0,
                "overall_score": 50.0,
                "avg_speedup": 0.0,
            },
        }

    if mode == "performance":
        return {
            "operator": operator,
            "mode": mode,
            "run_id": run_id,
            "success": True,
            "correctness_passed": True,
            "metrics": {
                "total_cases": 20,
                "passed_cases": 20,
                "failed_cases": 0,
                "pass_rate": 100.0,

                # Deliberately better than 50.70.
                "overall_score": 51.00,
                "avg_speedup": 0.03,
            },
        }

    raise ValueError(mode)


def main():
    iteration.run_evaluation = (
        fake_run_evaluation
    )

    iteration.judge_candidate(
        OPERATOR,
        RUN_ID,
    )


if __name__ == "__main__":
    main()
