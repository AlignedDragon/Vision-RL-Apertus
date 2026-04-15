"""Compute VTC-Bench accuracy with per-category and per-tier breakdown.

Handles both multiple-choice (letter matching) and open-ended (exact match).

Usage:
    python evaluation/compute_vtc_accuracy.py --predictions results/vtc_bench/baseline/predictions.jsonl
"""

import argparse
import json
import re
import string
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison (from compute_accuracy.py)."""
    answer = answer.lower().strip()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = " ".join(answer.split())
    return answer


def extract_mc_letter(prediction: str) -> str:
    """Extract MC option letter from a prediction string.

    Handles: "B", "B.", "B) text", "The answer is B", etc.
    """
    prediction = prediction.strip()
    if not prediction:
        return ""

    # Direct single letter
    if prediction.upper() in ("A", "B", "C", "D"):
        return prediction.upper()

    # "B." or "B)" or "B:" or "B " at start
    match = re.match(r"^([A-Da-d])[.):\s]", prediction)
    if match:
        return match.group(1).upper()

    # "The answer is B" or "Answer: B"
    match = re.search(r"(?:answer\s*(?:is|:)\s*)([A-Da-d])\b", prediction, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Last resort: first standalone A/B/C/D letter
    match = re.search(r"\b([A-Da-d])\b", prediction)
    if match:
        return match.group(1).upper()

    return prediction.strip()


def score_sample(prediction, answers, is_mc):
    """Score a single prediction. Returns 1.0 or 0.0."""
    if is_mc:
        pred_letter = extract_mc_letter(prediction)
        return 1.0 if any(pred_letter == a.strip().upper() for a in answers) else 0.0
    else:
        pred_norm = normalize_answer(prediction)
        return 1.0 if any(normalize_answer(a) == pred_norm for a in answers) else 0.0


# Category → tier mapping
TIERS = {
    "attention": 1,
    "ocr": 1,
    "perceptual": 1,
    "measure": 2,
    "color": 2,
    "counting": 2,
    "chart": 3,
    "math": 3,
    "spatial": 3,
}

TIER_NAMES = {
    1: "Tier 1: Visual Perception Enhancement",
    2: "Tier 2: Quantitative Visual Estimation",
    3: "Tier 3: Compositional Visual Reasoning",
}


def main():
    parser = argparse.ArgumentParser(description="Compute VTC-Bench accuracy")
    parser.add_argument(
        "--predictions",
        default="results/vtc_bench/baseline/predictions.jsonl",
        help="Path to predictions JSONL",
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    output_path = Path(args.output) if args.output else predictions_path.parent / "vtc_accuracy.json"

    # Load predictions
    samples = []
    with open(predictions_path) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} predictions from {predictions_path}")

    # Score each sample
    per_sample = []
    cat_scores = defaultdict(list)
    tier_scores = defaultdict(list)
    mc_scores = []
    oe_scores = []

    for s in samples:
        acc = score_sample(s["prediction"], s["answers"], s["is_mc"])
        category = s["category"]
        tier = TIERS.get(category, 0)

        entry = {
            "question_id": s["question_id"],
            "category": category,
            "is_mc": s["is_mc"],
            "prediction": s["prediction"][:200],
            "ground_truth": s["answers"][0] if s["answers"] else "",
            "accuracy": acc,
        }
        per_sample.append(entry)
        cat_scores[category].append(acc)
        tier_scores[tier].append(acc)

        if s["is_mc"]:
            mc_scores.append(acc)
        else:
            oe_scores.append(acc)

    # Compute aggregates
    overall = sum(e["accuracy"] for e in per_sample) / len(per_sample) if per_sample else 0.0

    per_category = {}
    for cat, scores in sorted(cat_scores.items()):
        per_category[cat] = {
            "accuracy": round(sum(scores) / len(scores), 4),
            "correct": sum(1 for s in scores if s > 0),
            "total": len(scores),
        }

    per_tier = {}
    for tier_id, scores in sorted(tier_scores.items()):
        per_tier[TIER_NAMES.get(tier_id, f"Tier {tier_id}")] = {
            "accuracy": round(sum(scores) / len(scores), 4),
            "correct": sum(1 for s in scores if s > 0),
            "total": len(scores),
        }

    result = {
        "overall_accuracy": round(overall, 4),
        "num_samples": len(samples),
        "num_correct": sum(1 for e in per_sample if e["accuracy"] > 0),
        "mc_accuracy": round(sum(mc_scores) / len(mc_scores), 4) if mc_scores else None,
        "mc_samples": len(mc_scores),
        "oe_accuracy": round(sum(oe_scores) / len(oe_scores), 4) if oe_scores else None,
        "oe_samples": len(oe_scores),
        "per_category": per_category,
        "per_tier": per_tier,
        "timestamp": datetime.now().isoformat(),
        "predictions_file": str(predictions_path),
        "per_sample": per_sample,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"VTC-Bench Evaluation Results")
    print(f"{'='*60}")
    print(f"Overall accuracy:   {overall:.4f} ({overall*100:.1f}%)")
    print(f"Samples:            {len(samples)}")
    print(f"Correct:            {result['num_correct']}")

    if mc_scores:
        mc_acc = sum(mc_scores) / len(mc_scores)
        print(f"\nMC accuracy:        {mc_acc:.4f} ({mc_acc*100:.1f}%) [{len(mc_scores)} samples]")
    if oe_scores:
        oe_acc = sum(oe_scores) / len(oe_scores)
        print(f"Open-ended accuracy:{oe_acc:.4f} ({oe_acc*100:.1f}%) [{len(oe_scores)} samples]")

    print(f"\n--- Per Category ---")
    for cat, info in per_category.items():
        print(f"  {cat:20s}  {info['accuracy']:.4f}  ({info['correct']}/{info['total']})")

    print(f"\n--- Per Tier ---")
    for tier_name, info in per_tier.items():
        print(f"  {tier_name:45s}  {info['accuracy']:.4f}  ({info['correct']}/{info['total']})")

    # Worst predictions
    failures = [e for e in per_sample if e["accuracy"] == 0]
    print(f"\n--- Sample Failures ({len(failures)} total) ---")
    for e in failures[:5]:
        print(f"  [{e['category']}] {e['question_id']}")
        print(f"    Predicted: '{e['prediction'][:80]}'")
        print(f"    Expected:  '{e['ground_truth']}'")
        print()

    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
