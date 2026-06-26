"""Half answer / half bbox reward for Visual-CoT RL trajectories.

The model emits tool calls in Apertus's native format, e.g.
    <|tools_prefix|>[{"draw_bbox_tool": {"bbox_2d": [x1, y1, x2, y2]}}]<|tools_suffix|>
    ...
    <|tools_prefix|>[{"display_answers": {"answers": ["<X>"]}}]<|tools_suffix|>

Score = ANSWER_WEIGHT * answer_match + BBOX_WEIGHT * IoU(pred_bbox, gold_bbox),
with ANSWER_WEIGHT = BBOX_WEIGHT = 0.5, so a perfect answer + perfect box -> 1.0.

  - answer_match: 1.0 if the last display_answers call string-matches the ground
    truth (case/trailing-punct normalized), else 0.0.
  - IoU: continuous overlap in [0, 1] between the last draw_bbox_tool call's
    bbox_2d and the gold box. Both are in ORIGINAL-image pixel space (the
    draw_bbox_tool schema and the gold `bboxs` agree), so IoU needs no rescaling.

The gold box is read from `extra_info["bbox"]` (carried by vcot_rl_parse.py).

Wired into verl via:
    reward.custom_reward_function:
      path: <abs>/rewards/vcot_rl_reward.py
      name: compute_score
"""

import json
import re

TOOLS_BLOCK = re.compile(r"<\|tools_prefix\|>(\[.*?\])<\|tools_suffix\|>", re.DOTALL)

ANSWER_WEIGHT = 0.5
BBOX_WEIGHT = 0.5


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


def _extract_pred_bbox(solution_str: str) -> list[float] | None:
    """Return the bbox_2d of the LAST draw_bbox_tool call, or None."""
    found = None
    for name, args in _iter_calls(solution_str):
        if name == "draw_bbox_tool":
            box = args.get("bbox_2d")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                try:
                    found = [float(v) for v in box]
                except (TypeError, ValueError):
                    continue
    return found


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


def _iou(pred, gold) -> float:
    """IoU of two [x1, y1, x2, y2] boxes; 0.0 if either is missing/degenerate."""
    if not pred or not gold or len(pred) != 4 or len(gold) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(v) for v in pred)
    bx1, by1, bx2, by2 = (float(v) for v in gold)
    # normalize corner order so swapped predictions still score
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _gold_bbox(ground_truth, extra_info):
    """Pull the gold [x1,y1,x2,y2] from extra_info (preferred) or ground_truth."""
    if isinstance(extra_info, dict):
        box = extra_info.get("bbox")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            return list(box)
    # fallback: ground_truth may itself be a dict carrying the box
    if isinstance(ground_truth, dict):
        box = ground_truth.get("bbox")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            return list(box)
    return None


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
    """0.5 * answer_match + 0.5 * IoU(pred_bbox, gold_bbox)."""
    # ground_truth may arrive as the raw answer string, or as a dict bundling
    # {"answer":..., "bbox":...}; support both.
    answer_gt = ground_truth
    if isinstance(ground_truth, dict) and "answer" in ground_truth:
        answer_gt = ground_truth["answer"]

    ans = _answer_match(_extract_display_answers(solution_str), answer_gt)
    iou = _iou(_extract_pred_bbox(solution_str), _gold_bbox(ground_truth, extra_info))
    return ANSWER_WEIGHT * ans + BBOX_WEIGHT * iou


# --------------------------------------------------------------------------- #
# Self-tests: run with `python rewards/vcot_rl_reward.py`
# --------------------------------------------------------------------------- #
def _run_self_tests():
    GOLD = [10, 10, 110, 110]  # 100x100 box, area 10000

    def sol(bbox=None, answer=None):
        parts = []
        if bbox is not None:
            parts.append('<|tools_prefix|>[{"draw_bbox_tool": {"bbox_2d": '
                         + json.dumps(bbox) + '}}]<|tools_suffix|>')
        if answer is not None:
            parts.append('<|tools_prefix|>[{"display_answers": {"answers": ["'
                         + answer + '"]}}]<|tools_suffix|>')
        return " thinking... ".join(parts)

    ei = {"bbox": GOLD}
    cases = [
        ("perfect answer + perfect box -> 1.0", sol(GOLD, "cat"), "cat", ei, 1.0),
        ("perfect answer, no box -> 0.5", sol(None, "cat"), "cat", ei, 0.5),
        ("wrong answer, perfect box -> 0.5", sol(GOLD, "dog"), "cat", ei, 0.5),
        ("perfect answer, half-overlap box -> 0.5 + 0.5*IoU",
         sol([10, 10, 110, 210], "cat"), "cat", ei, 0.5 + 0.5 * (10000 / 20000)),
        ("no box, no answer -> 0.0", "", "cat", ei, 0.0),
        ("missing gold bbox -> answer only",
         sol(GOLD, "cat"), "cat", {}, 0.5),
        ("openai-shaped calls",
         '<|tools_prefix|>[{"name":"draw_bbox_tool","arguments":{"bbox_2d":[10,10,110,110]}}]<|tools_suffix|>'
         '<|tools_prefix|>[{"name":"display_answers","arguments":{"answers":["cat"]}}]<|tools_suffix|>',
         "cat", ei, 1.0),
        ("swapped corners still score IoU",
         sol([110, 110, 10, 10], "cat"), "cat", ei, 1.0),
        ("ground_truth dict bundling answer+bbox",
         sol(GOLD, "cat"), {"answer": "cat", "bbox": GOLD}, None, 1.0),
        ("case/punct normalized answer",
         sol(GOLD, "Cat."), "cat", ei, 1.0),
        ("disjoint box -> answer only",
         sol([500, 500, 600, 600], "cat"), "cat", ei, 0.5),
    ]
    failures = 0
    for label, s, gt, extra, expected in cases:
        got = compute_score("vcot_rl", s, gt, extra)
        ok = abs(got - expected) < 1e-9
        if not ok:
            failures += 1
        print(f"[{'OK' if ok else 'FAIL'}] {label}: got={got:.4f} expected={expected:.4f}")
    print(f"\n{len(cases) - failures}/{len(cases)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
