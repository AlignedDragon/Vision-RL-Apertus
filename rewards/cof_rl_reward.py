"""Format-shaped reward for CoF-RL trajectories.

The model emits tool calls in Apertus's native format:
    <|tools_prefix|>[{"image_zoom_in_tool": {"bbox_2d": [x1, y1, x2, y2]}}]<|tools_suffix|>
    ...
    <|tools_prefix|>[{"display_answers": {"answers": ["<X>", ...]}}]<|tools_suffix|>

Score (max 1.0) is the sum of three parts:
    + 0.1  FORMAT: the rollout makes >=1 correctly-formatted display_answers call
    + 0.1  FORMAT: the rollout makes >=1 correctly-formatted image_zoom_in_tool call
    + 0.9  ANSWER: the last display_answers call string-matches the ground truth

The two 0.1 format bonuses are one-time: a single correct call earns them and
calling a tool repeatedly does not stack (only the first instance counts). They
are awarded even when the final answer is wrong, to credit the model for
learning the tool-call protocol.

Wired into verl via:
    reward.custom_reward_function:
      path: <abs>/rewards/cof_rl_reward.py
      name: compute_score
"""

import json
import re

TOOLS_BLOCK = re.compile(r"<\|tools_prefix\|>(\[.*?\])<\|tools_suffix\|>", re.DOTALL)

ZOOM_TOOL_NAME = "image_zoom_in_tool"

DISPLAY_FORMAT_WEIGHT = 0.1
ZOOM_FORMAT_WEIGHT = 0.1
ANSWER_WEIGHT = 0.9


# --------------------------------------------------------------------------- #
# Tool-call extraction
# --------------------------------------------------------------------------- #
def _iter_calls(solution_str: str):
    """Yield (name, args_dict) for every parseable native tool call, in order."""
    if not solution_str:
        return
    for block in TOOLS_BLOCK.findall(solution_str):
        try:
            calls = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(calls, list):
            calls = [calls]
        for call in calls:
            if not isinstance(call, dict):
                continue
            # OpenAI-shaped: {"name": ..., "arguments": {...}}
            if "name" in call and "arguments" in call:
                name = call.get("name")
                args = call.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        continue
                if isinstance(args, dict):
                    yield name, args
                continue
            # Native-shaped: {"<tool_name>": {...args...}}
            for name, args in call.items():
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        continue
                if isinstance(args, dict):
                    yield name, args


def _extract_display_answers(solution_str: str) -> list[str] | None:
    """Return the `answers` list of the LAST display_answers call, or None."""
    found = None
    for name, args in _iter_calls(solution_str):
        if name == "display_answers" and isinstance(args.get("answers"), list):
            found = [str(a) for a in args["answers"]]
    return found


def _is_valid_bbox(box) -> bool:
    """True iff `box` is 4 numeric values forming a canonical, non-degenerate box."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    return x1 < x2 and y1 < y2


def _has_display_call(solution_str: str) -> bool:
    """True iff >=1 display_answers call carries a non-empty list of answers."""
    for name, args in _iter_calls(solution_str):
        if name == "display_answers":
            answers = args.get("answers")
            if isinstance(answers, list) and len(answers) > 0:
                return True
    return False


def _has_zoom_call(solution_str: str) -> bool:
    """True iff >=1 image_zoom_in_tool call carries a valid bbox_2d."""
    for name, args in _iter_calls(solution_str):
        if name == ZOOM_TOOL_NAME and _is_valid_bbox(args.get("bbox_2d")):
            return True
    return False


# --------------------------------------------------------------------------- #
# Scoring primitives
# --------------------------------------------------------------------------- #
def _normalize(s: str) -> str:
    return s.strip().lower().rstrip(".,!?;: ")


def _answer_match(preds, ground_truth) -> float:
    if not preds:
        return 0.0
    if isinstance(ground_truth, (list, tuple)):
        gts = {_normalize(str(g)) for g in ground_truth}
        return 1.0 if {_normalize(p) for p in preds} == gts else 0.0
    target = _normalize(str(ground_truth))
    return 1.0 if any(_normalize(p) == target for p in preds) else 0.0


# --------------------------------------------------------------------------- #
# verl entry point
# --------------------------------------------------------------------------- #
def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> float:
    """0.1 display-format + 0.1 zoom-format + 0.9 answer-match (max 1.0)."""
    score = 0.0
    if _has_display_call(solution_str):
        score += DISPLAY_FORMAT_WEIGHT
    if _has_zoom_call(solution_str):
        score += ZOOM_FORMAT_WEIGHT
    score += ANSWER_WEIGHT * _answer_match(
        _extract_display_answers(solution_str), ground_truth
    )
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Self-tests: run with `python rewards/cof_rl_reward.py`
# ---------------------------------------------------------------------------


def _run_self_tests():
    cases = [
        # (label, solution_str, ground_truth, expected)
        (
            "happy path: display + correct answer",
            'sure thing <|tools_prefix|>[{"display_answers": {"answers": ["B"]}}]<|tools_suffix|>',
            "B",
            1.0,  # 0.1 display-format + 0.9 answer, capped at 1.0
        ),
        (
            "case + trailing punct normalization",
            '<|tools_prefix|>[{"display_answers": {"answers": ["yes."]}}]<|tools_suffix|>',
            "Yes",
            1.0,
        ),
        (
            "wrong answer -> display-format only",
            '<|tools_prefix|>[{"display_answers": {"answers": ["A"]}}]<|tools_suffix|>',
            "B",
            0.1,
        ),
        (
            "zoom + display + correct answer -> full 1.0",
            (
                '<|tools_prefix|>[{"image_zoom_in_tool": {"bbox_2d": [0,0,10,10]}}]<|tools_suffix|>'
                "...some more thinking..."
                '<|tools_prefix|>[{"display_answers": {"answers": ["C"]}}]<|tools_suffix|>'
            ),
            "C",
            1.0,  # 0.1 display + 0.1 zoom + 0.9 answer, capped at 1.0
        ),
        (
            "openai-shaped native tool call",
            '<|tools_prefix|>[{"name": "display_answers", "arguments": {"answers": ["D"]}}]<|tools_suffix|>',
            "D",
            1.0,
        ),
        (
            "zoom only, no display -> zoom-format only",
            '<|tools_prefix|>[{"image_zoom_in_tool": {"bbox_2d": [0,0,10,10]}}]<|tools_suffix|>',
            "anything",
            0.1,
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
            "list gt - missing element -> display-format only",
            '<|tools_prefix|>[{"display_answers": {"answers": ["A"]}}]<|tools_suffix|>',
            ["A", "B"],
            0.1,
        ),
        (
            "empty answers list -> no display-format, no answer -> 0.0",
            '<|tools_prefix|>[{"display_answers": {"answers": []}}]<|tools_suffix|>',
            "X",
            0.0,
        ),
        (
            "plain decoded answer without any tool call -> 0.0",
            "Yes",
            "yes",
            0.0,
        ),
        (
            "zoom + display + WRONG answer -> both format bonuses only",
            (
                '<|tools_prefix|>[{"image_zoom_in_tool": {"bbox_2d": [0,0,10,10]}}]<|tools_suffix|>'
                '<|tools_prefix|>[{"display_answers": {"answers": ["A"]}}]<|tools_suffix|>'
            ),
            "B",
            0.2,  # 0.1 zoom + 0.1 display
        ),
        (
            "degenerate zoom bbox (x1==x2) -> no zoom-format",
            (
                '<|tools_prefix|>[{"image_zoom_in_tool": {"bbox_2d": [5,0,5,10]}}]<|tools_suffix|>'
                '<|tools_prefix|>[{"display_answers": {"answers": ["B"]}}]<|tools_suffix|>'
            ),
            "B",
            1.0,  # display-format 0.1 + answer 0.9, capped at 1.0; zoom invalid
        ),
        (
            "zoom bbox not length-4 -> no zoom-format",
            '<|tools_prefix|>[{"image_zoom_in_tool": {"bbox_2d": [0,0,10]}}]<|tools_suffix|>',
            "B",
            0.0,
        ),
        (
            "repeated zoom calls still cap zoom-format at 0.1",
            (
                '<|tools_prefix|>[{"image_zoom_in_tool": {"bbox_2d": [0,0,10,10]}}]<|tools_suffix|>'
                '<|tools_prefix|>[{"image_zoom_in_tool": {"bbox_2d": [1,1,9,9]}}]<|tools_suffix|>'
            ),
            "B",
            0.1,
        ),
    ]
    failures = 0
    for label, sol, gt, expected in cases:
        got = compute_score("cof_rl", sol, gt)
        ok = abs(got - expected) < 1e-9
        if not ok:
            failures += 1
        print(f"[{'OK' if ok else 'FAIL'}] {label}: got={got} expected={expected}")
    print(f"\n{len(cases) - failures}/{len(cases)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
