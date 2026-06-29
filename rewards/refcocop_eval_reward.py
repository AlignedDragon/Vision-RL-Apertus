"""RefCOCO+ referring-expression-grounding metric, format-agnostic (fair to base).

RefCOCO+ REC: given an image + a referring expression, localize the region. The
ANSWER is the bounding box (there is no separate text answer to score). Standard
metric = Acc@0.5: correct iff IoU(pred, gold) >= 0.5.

Format-agnostic so the base model (which can't call tools) is judged on whatever box
it writes: use the draw_bbox_tool box if present (SFT/RL), else parse the first
[x1,y1,x2,y2] from the free-text response (base). Gold box is in smart_resize'd /
perceived pixel space (carried in extra_info["bbox"]). Note: a base model that emits
coordinates in a different space (original pixels / normalized) will score low — its
boxes aren't in the perceived space the gold uses; that's an inherent limit of
evaluating a non-grounding-tuned base, not a parser bug.

verl wiring (eval-time override):
    reward.custom_reward_function.path=<abs>/rewards/refcocop_eval_reward.py
    reward.custom_reward_function.name=compute_score
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards.vcot_rl_reward import _extract_pred_bbox, _gold_bbox, _has_bbox_call, _iou
from rewards.vcot_eval_reward import _freetext_bbox

IOU_THRESHOLD = 0.5


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    pred = _extract_pred_bbox(solution_str)
    if pred is None:
        pred = _freetext_bbox(solution_str)
    # Base models often emit NORMALIZED [0,1] coords (e.g. "[0.0,0.0,1.0,1.0]");
    # rescale to the perceived pixel space the gold box lives in so base is judged
    # fairly (SFT/RL emit pixel coords already -> max>1 -> untouched).
    if (pred is not None and max(pred) <= 1.0 and extra_info
            and isinstance(extra_info.get("image_wh"), (list, tuple))
            and len(extra_info["image_wh"]) == 2):
        W, H = extra_info["image_wh"]
        pred = [pred[0] * W, pred[1] * H, pred[2] * W, pred[3] * H]
    iou = _iou(pred, _gold_bbox(ground_truth, extra_info))
    grounding = 1.0 if iou >= IOU_THRESHOLD else 0.0
    return {
        "score": grounding,          # val-core == Acc@0.5
        "grounding_acc": grounding,
        "iou": float(iou),
        "fmt_bbox": 1.0 if _has_bbox_call(solution_str) else 0.0,
    }


def _run_self_tests():
    import json
    GOLD = [10, 10, 110, 110]
    ei = {"bbox": GOLD}

    def tool(b):
        return '<|tools_prefix|>[{"draw_bbox_tool": {"bbox_2d": ' + json.dumps(b) + '}}]<|tools_suffix|>'

    assert compute_score("refcocop", tool(GOLD), "x", ei)["grounding_acc"] == 1.0
    assert compute_score("refcocop", tool([500, 500, 600, 600]), "x", ei)["grounding_acc"] == 0.0
    # base free-text box
    r = compute_score("refcocop", "The object is at [10, 10, 110, 110].", "x", ei)
    assert r["grounding_acc"] == 1.0 and r["fmt_bbox"] == 0.0, r
    assert compute_score("refcocop", "I cannot tell.", "x", ei)["grounding_acc"] == 0.0
    print("refcocop_eval_reward self-tests passed")


if __name__ == "__main__":
    _run_self_tests()
