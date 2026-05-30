"""1/0 reward for CoF-RL trajectories.

The model emits tool calls in Apertus's native format:
    <|tools_prefix|>[{"display_answers": {"answers": ["<X>", ...]}}]<|tools_suffix|>

We pull the *last* display_answers call out of the rollout's solution string
and string-match its `answers` list against the ground truth.

Wired into verl via:
    reward.custom_reward_function:
      path: <abs>/rewards/cof_rl_reward.py
      name: compute_score
"""

import json
import re

TOOLS_BLOCK = re.compile(r"<\|tools_prefix\|>(\[.*?\])<\|tools_suffix\|>", re.DOTALL)


def _extract_display_answers(solution_str: str) -> list[str] | None:
    """Return the `answers` list of the last display_answers call, or None."""
    if not solution_str:
        return None
    blocks = TOOLS_BLOCK.findall(solution_str)
    if not blocks:
        return None
    for block in reversed(blocks):
        try:
            calls = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(calls, list):
            calls = [calls]
        for call in reversed(calls):
            if not isinstance(call, dict):
                continue
            args = call.get("display_answers")
            if args is None and call.get("name") == "display_answers":
                args = call.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        continue
            if isinstance(args, dict) and isinstance(args.get("answers"), list):
                return [str(a) for a in args["answers"]]
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
    preds = _extract_display_answers(solution_str)
    if not preds:
        return 0.0
    if isinstance(ground_truth, (list, tuple)):
        gts = {_normalize(str(g)) for g in ground_truth}
        norm_preds = {_normalize(p) for p in preds}
        return 1.0 if gts == norm_preds else 0.0
    target = _normalize(str(ground_truth))
    return 1.0 if any(_normalize(p) == target for p in preds) else 0.0


# ---------------------------------------------------------------------------
# Self-tests: run with `python rewards/cof_rl_reward.py`
# ---------------------------------------------------------------------------


def _run_self_tests():
    cases = [
        # (label, solution_str, ground_truth, expected)
        (
            "happy path",
            'sure thing <|tools_prefix|>[{"display_answers": {"answers": ["B"]}}]<|tools_suffix|>',
            "B",
            1.0,
        ),
        (
            "case + trailing punct normalization",
            '<|tools_prefix|>[{"display_answers": {"answers": ["yes."]}}]<|tools_suffix|>',
            "Yes",
            1.0,
        ),
        (
            "wrong answer",
            '<|tools_prefix|>[{"display_answers": {"answers": ["A"]}}]<|tools_suffix|>',
            "B",
            0.0,
        ),
        (
            "multiple tool blocks - take the last display_answers",
            (
                '<|tools_prefix|>[{"image_zoom_in_tool": {"bbox_2d": [0,0,10,10]}}]<|tools_suffix|>'
                "...some more thinking..."
                '<|tools_prefix|>[{"display_answers": {"answers": ["C"]}}]<|tools_suffix|>'
            ),
            "C",
            1.0,
        ),
        (
            "openai-shaped native tool call",
            '<|tools_prefix|>[{"name": "display_answers", "arguments": {"answers": ["D"]}}]<|tools_suffix|>',
            "D",
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
            '<|tools_prefix|>[{"display_answers": {"answers": ["X"]]<|tools_suffix|>',
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
            '<|tools_prefix|>[{"display_answers": {"answers": ["  Hello World  "]}}]<|tools_suffix|>',
            "hello world",
            1.0,
        ),
        (
            "string gt matches one of multiple answers",
            '<|tools_prefix|>[{"display_answers": {"answers": ["A", "B", "C"]}}]<|tools_suffix|>',
            "B",
            1.0,
        ),
        (
            "list gt - exact set match",
            '<|tools_prefix|>[{"display_answers": {"answers": ["a", "B"]}}]<|tools_suffix|>',
            ["A", "b"],
            1.0,
        ),
        (
            "list gt - missing element -> 0.0",
            '<|tools_prefix|>[{"display_answers": {"answers": ["A"]}}]<|tools_suffix|>',
            ["A", "B"],
            0.0,
        ),
        (
            "empty answers list -> 0.0",
            '<|tools_prefix|>[{"display_answers": {"answers": []}}]<|tools_suffix|>',
            "X",
            0.0,
        ),
        (
            "plain decoded answer without display_answers -> 0.0",
            "Yes",
            "yes",
            0.0,
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
