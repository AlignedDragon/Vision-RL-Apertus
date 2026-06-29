"""Format-agnostic in-distribution VCoT eval metric (fair across base / SFT / RL).

Goal: compare models on WHAT they answer/ground, not on whether they speak our
native tool-call syntax. The base model was never trained on draw_bbox_tool /
display_answers, so scoring it on emitting that format is trivially 0 and
meaningless. Here every model is given the tools, answers in its own style, and is
scored leniently:

  - answer_acc: if the response makes a display_answers tool call (SFT/RL), match
    its payload against the gold (normalized exact). Otherwise (base free-text),
    accept the answer if the gold answer appears as a normalized whole word in the
    response text.
  - grounding_acc: Acc@0.5 using the draw_bbox_tool bbox if present, else the first
    [x1,y1,x2,y2]-style box parsed from the free-text response.
  - iou: continuous IoU. fmt_bbox / fmt_display: native tool-call rates (so we can
    still see that base ~never uses the tools while still being credited for
    correct free-text answers).

`compute_score` returns a dict; verl logs val-core score (= grounding Acc@0.5) and
the rest via reward_extra_info. Boxes are in smart_resize'd/perceived pixel space.
Reuses the strict extraction + IoU primitives from vcot_rl_reward.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards.vcot_rl_reward import (
    _answer_match,
    _extract_display_answers,
    _extract_pred_bbox,
    _gold_bbox,
    _has_bbox_call,
    _has_display_call,
    _iou,
)

IOU_THRESHOLD = 0.5

# A 4-number box anywhere in free text, e.g. "[12, 34, 56, 78]" or "(12,34,56,78)".
_FREETEXT_BBOX_RE = re.compile(
    r"[\[(]?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\])]?"
)


def _norm_words(s) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()


def _freetext_answer_match(text: str, gold) -> float:
    """1.0 if every (normalized) gold word appears as a whole word in the text."""
    gw = _norm_words(gold)
    if not gw:
        return 0.0
    tw = set(_norm_words(text))
    return 1.0 if all(g in tw for g in gw) else 0.0


def _freetext_bbox(text: str):
    m = _FREETEXT_BBOX_RE.search(text or "")
    return [float(x) for x in m.groups()] if m else None


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    answer_gt = ground_truth
    if isinstance(ground_truth, dict) and "answer" in ground_truth:
        answer_gt = ground_truth["answer"]

    # Answer: prefer the structured display_answers payload (SFT/RL); fall back to
    # free-text gold-word match (base) only when there is no display_answers call.
    disp = _extract_display_answers(solution_str)
    if disp:
        ans = _answer_match(disp, answer_gt)
    else:
        ans = _freetext_answer_match(solution_str, answer_gt)

    # Grounding: prefer the draw_bbox_tool box; else any free-text box.
    pred_bbox = _extract_pred_bbox(solution_str)
    if pred_bbox is None:
        pred_bbox = _freetext_bbox(solution_str)
    iou = _iou(pred_bbox, _gold_bbox(ground_truth, extra_info))
    grounding = 1.0 if iou >= IOU_THRESHOLD else 0.0

    return {
        "score": grounding,            # val-core metric == grounding Acc@0.5
        "answer_acc": float(ans),
        "grounding_acc": grounding,
        "iou": float(iou),
        "fmt_bbox": 1.0 if _has_bbox_call(solution_str) else 0.0,
        "fmt_display": 1.0 if _has_display_call(solution_str) else 0.0,
    }


# --------------------------------------------------------------------------- #
# Self-tests: python rewards/vcot_eval_reward.py
# --------------------------------------------------------------------------- #
def _run_self_tests():
    import json

    GOLD = [10, 10, 110, 110]
    ei = {"bbox": GOLD}

    def tool_sol(bbox=None, answer=None):
        parts = []
        if bbox is not None:
            parts.append('<|tools_prefix|>[{"draw_bbox_tool": {"bbox_2d": '
                         + json.dumps(bbox) + '}}]<|tools_suffix|>')
        if answer is not None:
            parts.append('<|tools_prefix|>[{"display_answers": {"answers": ["'
                         + answer + '"]}}]<|tools_suffix|>')
        return " thinking... ".join(parts)

    # SFT/RL style (tools): structured path
    r = compute_score("vcot_rl", tool_sol(GOLD, "cat"), "cat", ei)
    assert r["answer_acc"] == 1.0 and r["grounding_acc"] == 1.0, r
    r = compute_score("vcot_rl", tool_sol(GOLD, "dog"), "cat", ei)
    assert r["answer_acc"] == 0.0 and r["grounding_acc"] == 1.0, r  # right box, wrong answer

    # base style (free text, no tools): lenient answer + free-text box
    base = "The bird does have a grey crown, so the answer is yes. Region: [10, 10, 110, 110]."
    r = compute_score("vcot_rl", base, "yes", ei)
    assert r["answer_acc"] == 1.0, r            # 'yes' present in free text
    assert r["grounding_acc"] == 1.0, r          # parsed [10,10,110,110]
    assert r["fmt_display"] == 0.0 and r["fmt_bbox"] == 0.0, r  # used no tools
    r = compute_score("vcot_rl", "I think it is a dog.", "cat", ei)
    assert r["answer_acc"] == 0.0, r
    # whole-word, not substring: 'no' must not match inside 'another'
    r = compute_score("vcot_rl", "there is another object", "no", {"bbox": GOLD})
    assert r["answer_acc"] == 0.0, r
    print("all vcot_eval_reward self-tests passed")


if __name__ == "__main__":
    _run_self_tests()
