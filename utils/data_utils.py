"""Shared data utilities."""
import json
import math
import re
from pathlib import Path

import numpy as np

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def compute_bbox_stats(raw_path, output_path=None, patch_size=16):
    """Compute bbox width/height/patch-count distribution from a raw.jsonl.

    Each row's assistant messages are scanned for <tool_call> blocks; any
    `bbox_2d` argument contributes (w, h) and patch_count = ceil(w/patch_size)
    * ceil(h/patch_size). Returns the stats dict and optionally writes JSON.
    """
    raw_path = Path(raw_path)
    widths, heights, patch_counts = [], [], []
    n_rows = n_calls = n_bad = 0

    with open(raw_path) as f:
        for line in f:
            row = json.loads(line)
            n_rows += 1
            for msg in row.get("messages", []):
                if msg.get("role") != "assistant":
                    continue
                for m in _TOOL_CALL_RE.finditer(msg["content"]):
                    n_calls += 1
                    try:
                        call = json.loads(m.group(1).strip())
                        bbox = call.get("arguments", {}).get("bbox_2d")
                        if not bbox or len(bbox) != 4:
                            continue
                        x1, y1, x2, y2 = bbox
                        w = max(0.0, float(x2) - float(x1))
                        h = max(0.0, float(y2) - float(y1))
                        pw = max(1, math.ceil(w / patch_size))
                        ph = max(1, math.ceil(h / patch_size))
                        widths.append(w)
                        heights.append(h)
                        patch_counts.append(pw * ph)
                    except Exception:
                        n_bad += 1

    def _percentiles(arr):
        a = np.asarray(arr)
        return {
            "count": int(a.size),
            "min": float(a.min()),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
        }

    def _histogram(arr, edges):
        a = np.asarray(arr)
        return {
            f"{lo}-{hi}": int(((a >= lo) & (a < hi)).sum())
            for lo, hi in zip(edges[:-1], edges[1:])
        }

    side_edges = [0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 10_000]
    patch_edges = [0, 256, 512, 1024, 2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072, 1_000_000_000]

    stats = {
        "raw_path": str(raw_path),
        "num_rows": n_rows,
        "num_tool_calls": n_calls,
        "num_bbox": len(patch_counts),
        "num_parse_errors": n_bad,
        "patch_size": patch_size,
        "width_stats": _percentiles(widths),
        "height_stats": _percentiles(heights),
        "patch_count_stats": _percentiles(patch_counts),
        "width_histogram": _histogram(widths, side_edges),
        "height_histogram": _histogram(heights, side_edges),
        "patch_count_histogram": _histogram(patch_counts, patch_edges),
    }

    if output_path is not None:
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--patch-size", type=int, default=16)
    args = parser.parse_args()
    stats = compute_bbox_stats(args.raw_path, args.output, args.patch_size)
    print(json.dumps(stats, indent=2))
