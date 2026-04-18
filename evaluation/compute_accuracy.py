"""Compute accuracy from predictions. Supports three evaluation modes:

- vqa:    TextVQA soft accuracy — min(1, matches/3) over 10 annotations
- mcq:    Exact match after normalization (for multiple-choice)
- binary: Yes/no accuracy + precision/recall/F1 (for POPE)

Usage:
    python evaluation/compute_accuracy.py --predictions results/textvqa/baseline/predictions.jsonl
    python evaluation/compute_accuracy.py --predictions results/pope/baseline/predictions.jsonl --mode binary
"""

import argparse
import json
import re
import string
from datetime import datetime
from pathlib import Path


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Follows the standard VQA evaluation:
    1. Lowercase
    2. Strip whitespace
    3. Remove punctuation
    4. Remove articles (a, an, the)
    5. Collapse multiple spaces
    """
    answer = answer.lower().strip()
    # Remove punctuation
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    # Collapse whitespace
    answer = " ".join(answer.split())
    return answer


def textvqa_accuracy(prediction: str, answers: list[str]) -> float:
    """Compute TextVQA soft accuracy for a single sample.

    Returns min(1, count / 3) where count is the number of human answers
    that match the prediction after normalization.
    """
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return 0.0
    matches = sum(1 for a in answers if normalize_answer(a) == pred_norm)
    return min(1.0, matches / 3.0)


# ! NOT CHECKED
def mcq_accuracy(prediction: str, answers: list[str]) -> float:
    """Exact match after normalization. Returns 1.0 or 0.0."""
    pred_norm = normalize_answer(prediction)
    return 1.0 if any(normalize_answer(a) == pred_norm for a in answers) else 0.0

# ! NOT CHECKED
def binary_accuracy(prediction: str, answers: list[str]) -> float:
    """Yes/no exact match. Returns 1.0 or 0.0."""
    pred_norm = normalize_answer(prediction)
    # Accept common variants
    pred_yn = "yes" if pred_norm in ("yes", "true", "1") else "no" if pred_norm in ("no", "false", "0") else pred_norm
    return 1.0 if any(normalize_answer(a) == pred_yn for a in answers) else 0.0


SCORERS = {
    "vqa": textvqa_accuracy,
    "mcq": mcq_accuracy,
    "binary": binary_accuracy,
}


def main():
    parser = argparse.ArgumentParser(description="Compute TextVQA accuracy")
    parser.add_argument(
        "--predictions",
        default="results/textvqa/baseline/predictions.jsonl",
        help="Path to predictions JSONL file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output accuracy JSON (default: same dir as predictions)",
    )
    parser.add_argument(
        "--mode",
        default="vqa",
        choices=["vqa", "mcq", "binary"],
        help="Evaluation mode: vqa (soft accuracy), mcq (exact match), binary (yes/no + F1)",
    )
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = predictions_path.parent / "accuracy.json"

    # Load predictions
    samples = []
    with open(predictions_path) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} predictions from {predictions_path}")

    scorer = SCORERS[args.mode]

    # Compute per-sample accuracy
    per_sample = []
    for s in samples:
        acc = scorer(s["prediction"], s["answers"])
        per_sample.append({
            "question_id": s["question_id"],
            "question": s["question"],
            "prediction": s["prediction"],
            "top_answer": max(set(s["answers"]), key=s["answers"].count),
            "accuracy": acc,
        })

    # Overall accuracy
    overall = sum(p["accuracy"] for p in per_sample) / len(per_sample) if per_sample else 0.0

    # Count correct (accuracy > 0)
    num_correct = sum(1 for p in per_sample if p["accuracy"] > 0)
    num_perfect = sum(1 for p in per_sample if p["accuracy"] == 1.0)

    # Sort by accuracy for failure analysis
    per_sample_sorted = sorted(per_sample, key=lambda p: p["accuracy"])

    # Build output
    result = {
        "mode": args.mode,
        "overall_accuracy": round(overall, 4),
        "num_samples": len(samples),
        "num_correct": num_correct,
        "num_perfect": num_perfect,
        "timestamp": datetime.now().isoformat(),
        "predictions_file": str(predictions_path),
        "per_sample": per_sample,
    }

    # Binary mode: add precision/recall/F1
    if args.mode == "binary":
        tp = sum(1 for s, p in zip(samples, per_sample) if s["answers"][0] == "yes" and p["accuracy"] == 1.0)
        fp = sum(1 for s, p in zip(samples, per_sample) if s["answers"][0] == "no" and p["accuracy"] == 0.0)
        fn = sum(1 for s, p in zip(samples, per_sample) if s["answers"][0] == "yes" and p["accuracy"] == 0.0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result["precision"] = round(precision, 4)
        result["recall"] = round(recall, 4)
        result["f1"] = round(f1, 4)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Results (mode={args.mode})")
    print(f"{'='*60}")
    print(f"Overall accuracy:   {overall:.4f} ({overall*100:.1f}%)")
    print(f"Samples:            {len(samples)}")
    print(f"Correct (acc > 0):  {num_correct} ({num_correct/len(samples)*100:.1f}%)")
    print(f"Perfect (acc = 1):  {num_perfect} ({num_perfect/len(samples)*100:.1f}%)")
    print(f"Output saved to:    {output_path}")
    if args.mode == "binary":
        print(f"Precision:          {result['precision']:.4f}")
        print(f"Recall:             {result['recall']:.4f}")
        print(f"F1:                 {result['f1']:.4f}")

    # Show worst failures
    print(f"\n--- Worst 5 Predictions ---")
    for p in per_sample_sorted[:5]:
        print(f"  Q: {p['question']}")
        print(f"  Predicted: '{p['prediction']}'")
        print(f"  Expected:  '{p['top_answer']}'")
        print(f"  Accuracy:  {p['accuracy']}")
        print()

    # Show best predictions
    print(f"--- Best 5 Predictions ---")
    for p in per_sample_sorted[-5:]:
        print(f"  Q: {p['question']}")
        print(f"  Predicted: '{p['prediction']}'")
        print(f"  Expected:  '{p['top_answer']}'")
        print(f"  Accuracy:  {p['accuracy']}")
        print()


if __name__ == "__main__":
    main()
