"""1/0 reward for CoF-RL trajectories.

The model emits tool calls in Apertus's native format:
    <|tools_prefix|>[{"display_answers": {"answer": "<X>"}}]<|tools_suffix|>

We pull the *last* display_answers call out of the rollout's solution string
and string-match its `answer` argument against the ground truth.

Wired into verl via:
    reward.custom_reward_function:
      path: <abs>/rewards/cof_rl_reward.py
      name: compute_score
"""

import json
import re

TOOLS_BLOCK = re.compile(r"<\|tools_prefix\|>\[(.*?)\]<\|tools_suffix\|>", re.DOTALL)


def _extract_display_answer(solution_str: str) -> str | None:
    """Return the `answer` field of the last display_answers call, or None."""
    if not solution_str:
        return None
    for inner in reversed(TOOLS_BLOCK.findall(solution_str)):
        try:
            calls = json.loads(f"[{inner}]")
        except json.JSONDecodeError:
            continue
        if not isinstance(calls, list):
            continue
        for call in reversed(calls):
            if not isinstance(call, dict):
                continue
            if "display_answers" not in call:
                continue
            args = call["display_answers"]
            if isinstance(args, dict) and "answer" in args:
                return str(args["answer"])
    return None


def _normalize(s: str) -> str:
    return s.strip().lower().rstrip(".,!?;: ")


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> float:
    pred = _extract_display_answer(solution_str)
    if pred is None:
        return 0.0
    return 1.0 if _normalize(pred) == _normalize(str(ground_truth)) else 0.0


# ---------------------------------------------------------------------------
# Self-tests: run with `python rewards/cof_rl_reward.py`
# ---------------------------------------------------------------------------


def _run_self_tests():
    cases = [
        # (label, solution_str, ground_truth, expected)
        (
            "happy path",
            'sure thing <|tools_prefix|>[{"display_answers": {"answer": "B"}}]<|tools_suffix|>',
            "B",
            1.0,
        ),
        (
            "case + trailing punct normalization",
            '<|tools_prefix|>[{"display_answers": {"answer": "yes."}}]<|tools_suffix|>',
            "Yes",
            1.0,
        ),
        (
            "wrong answer",
            '<|tools_prefix|>[{"display_answers": {"answer": "A"}}]<|tools_suffix|>',
            "B",
            0.0,
        ),
        (
            "multiple tool blocks - take the last display_answers",
            (
                '<|tools_prefix|>[{"image_zoom_in_tool": {"bbox_2d": [0,0,10,10]}}]<|tools_suffix|>'
                "...some more thinking..."
                '<|tools_prefix|>[{"display_answers": {"answer": "C"}}]<|tools_suffix|>'
            ),
            "C",
            1.0,
        ),
        (
            "no display_answers at all -> 0.0",
            '<|tools_prefix|>[{"image_zoom_in_tool": {"bbox_2d": [0,0,10,10]}}]<|tools_suffix|>',
            "anything",
            0.0,
        ),
        (
            "malformed JSON inside tool block -> 0.0",
            '<|tools_prefix|>[{"display_answers": {"answer": "X"]<|tools_suffix|>',
            "X",
            0.0,
        ),
        (
            "empty solution -> 0.0",
            "",
            "X",
            0.0,
        ),
        (
            "whitespace + mixed case match",
            '<|tools_prefix|>[{"display_answers": {"answer": "  Hello World  "}}]<|tools_suffix|>',
            "hello world",
            1.0,
        ),
    ]
    failures = 0
    for label, sol, gt, expected in cases:
        got = compute_score("cof_rl", sol, gt)
        ok = got == expected
        if not ok:
            failures += 1
        print(f"[{'OK' if ok else 'FAIL'}] {label}: got={got} expected={expected}")
    print(f"\n{len(cases) - failures}/{len(cases)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
