"""RefCOCO+ grounding metric for VCoT (Chain-of-Focus) Apertus rollouts.

RefCOCO+ is a referring-expression comprehension benchmark: given an image and a
referring expression, the model must localise the referred region. The VCoT model
emits that region via a native draw_bbox_tool call, e.g.

    <|tools_prefix|>[{"draw_bbox_tool": {"bbox_2d": [x1, y1, x2, y2]}}]<|tools_suffix|>

Standard metric = Precision@0.5 (a.k.a. Acc@0.5): the prediction is correct iff
IoU(pred_bbox, gold_bbox) >= 0.5. `compute_score` returns 1.0 / 0.0 so verl's
val-core mean is exactly Acc@0.5.

Both boxes are in the resized/perceived image space (smart_resize'd pixels): the
RL/eval parse expresses the gold box in that space via scale_bbox, matching what
the model emits ("pixels of the image as shown to you"). The gold box is read from
extra_info["bbox"] (carried by refcocop_parse.py).

We reuse the tool-call extraction + IoU primitives from vcot_rl_reward so parsing
stays identical to RL training/scoring.

Wired into verl via an eval-time override:
    reward.custom_reward_function.path=<abs>/rewards/refcocop_reward.py
    reward.custom_reward_function.name=compute_score
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards.vcot_rl_reward import _extract_pred_bbox, _iou, _gold_bbox

IOU_THRESHOLD = 0.5


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> float:
    """Acc@0.5: 1.0 if IoU(pred draw_bbox, gold bbox) >= 0.5 else 0.0."""
    pred = _extract_pred_bbox(solution_str)
    gold = _gold_bbox(ground_truth, extra_info)
    iou = _iou(pred, gold)
    return 1.0 if iou >= IOU_THRESHOLD else 0.0


# --------------------------------------------------------------------------- #
# Self-tests: run with `python rewards/refcocop_reward.py`
# --------------------------------------------------------------------------- #
def _run_self_tests():
    import json

    GOLD = [10, 10, 110, 110]  # 100x100 box

    def sol(bbox):
        return ('<|tools_prefix|>[{"draw_bbox_tool": {"bbox_2d": '
                + json.dumps(bbox) + '}}]<|tools_suffix|>')

    ei = {"bbox": GOLD}
    cases = [
        ("perfect box -> 1.0", sol(GOLD), ei, 1.0),
        # pred area 100x300=30000, inter 10000, union 30000 -> IoU=1/3 < 0.5.
        ("IoU=1/3 (<0.5) -> 0.0", sol([10, 10, 110, 310]), ei, 0.0),
        # half-area overlap inside gold: pred [10,10,110,60] area 5000, inter 5000,
        # union 10000 -> IoU 0.5 -> correct.
        ("IoU=0.5 boundary -> 1.0", sol([10, 10, 110, 60]), ei, 1.0),
        ("disjoint box -> 0.0", sol([500, 500, 600, 600]), ei, 0.0),
        ("no draw_bbox call -> 0.0", "thinking only", ei, 0.0),
        ("swapped corners still score", sol([110, 110, 10, 10]), ei, 1.0),
        ("missing gold -> 0.0", sol(GOLD), {}, 0.0),
        ("gold from ground_truth dict", sol(GOLD), None, 1.0),
    ]
    failures = 0
    for label, s, extra, expected in cases:
        gt = {"bbox": GOLD} if (extra is None) else "x"
        got = compute_score("refcocop", s, gt, extra)
        ok = abs(got - expected) < 1e-9
        failures += not ok
        print(f"[{'OK' if ok else 'FAIL'}] {label}: got={got:.1f} expected={expected:.1f}")
    print(f"\n{len(cases) - failures}/{len(cases)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
