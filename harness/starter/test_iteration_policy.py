#!/usr/bin/env python3

from iteration import is_better


def check(
    name,
    candidate_score,
    candidate_speedup,
    best_score,
    best_speedup,
    expected,
):
    actual = is_better(
        candidate_score,
        candidate_speedup,
        best_score,
        best_speedup,
    )

    status = (
        "PASS"
        if actual == expected
        else "FAIL"
    )

    print(
        f"{status}: {name}"
        f" -> actual={actual}, expected={expected}"
    )

    assert actual == expected


def main():
    # Higher score -> accept
    check(
        "higher score",
        51.0,
        0.02,
        50.7,
        0.02,
        True,
    )

    # Same score but faster -> accept
    check(
        "same score, faster",
        50.7,
        0.03,
        50.7,
        0.02,
        True,
    )

    # Exactly equal -> reject
    check(
        "exactly equal",
        50.7,
        0.02,
        50.7,
        0.02,
        False,
    )

    # Worse score -> reject even if speedup is larger
    check(
        "lower score",
        50.6,
        0.50,
        50.7,
        0.02,
        False,
    )

    print()
    print("ALL SELECTION POLICY TESTS PASSED")


if __name__ == "__main__":
    main()
