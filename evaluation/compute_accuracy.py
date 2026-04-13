"""Compute TextVQA soft accuracy from predictions.

TextVQA uses the VQA challenge accuracy metric:
    accuracy(pred, answers) = min(1, count(pred matches in answers) / 3)

where answers is the list of 10 human annotations per question.
Both prediction and answers are normalized before comparison.

Usage:
    python evaluation/compute_accuracy.py --predictions results/textvqa/baseline/predictions.jsonl
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

    # Compute per-sample accuracy
    per_sample = []
    for s in samples:
        acc = textvqa_accuracy(s["prediction"], s["answers"])
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
        "overall_accuracy": round(overall, 4),
        "num_samples": len(samples),
        "num_correct": num_correct,
        "num_perfect": num_perfect,
        "timestamp": datetime.now().isoformat(),
        "predictions_file": str(predictions_path),
        "per_sample": per_sample,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"TextVQA Baseline Accuracy")
    print(f"{'='*60}")
    print(f"Overall accuracy:   {overall:.4f} ({overall*100:.1f}%)")
    print(f"Samples:            {len(samples)}")
    print(f"Correct (acc > 0):  {num_correct} ({num_correct/len(samples)*100:.1f}%)")
    print(f"Perfect (acc = 1):  {num_perfect} ({num_perfect/len(samples)*100:.1f}%)")
    print(f"Output saved to:    {output_path}")

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
