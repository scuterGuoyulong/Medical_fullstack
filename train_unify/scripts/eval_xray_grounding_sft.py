#!/usr/bin/env python3
"""
Validate and summarize weakly supervised X-ray grounding SFT data.

This is intentionally lightweight: it checks the generated JSON before costly
training and can optionally score model predictions if you provide a JSON with
`id` and `prediction`/`text` fields.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


REGION_ZH = {
    "left_upper_lung": "左上肺野",
    "left_mid_lung": "左中肺野",
    "left_lower_lung": "左下肺野",
    "right_upper_lung": "右上肺野",
    "right_mid_lung": "右中肺野",
    "right_lower_lung": "右下肺野",
    "left_lung": "左肺野",
    "right_lung": "右肺野",
    "bilateral_lungs": "双肺野",
    "cardiomediastinal": "心影纵隔区",
    "left_costophrenic_angle": "左肋膈角",
    "right_costophrenic_angle": "右肋膈角",
    "left_hilum": "左肺门",
    "right_hilum": "右肺门",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_image(path: str, image_folder: Path) -> Path:
    p = Path(path)
    if p.is_absolute() or str(path).startswith("http"):
        return p
    return image_folder / p


def final_answer(example: dict[str, Any]) -> str:
    conv = example.get("conversations") or []
    if len(conv) < 2:
        return ""
    return str(conv[1].get("value", ""))


def predict_label_from_text(text: str) -> str | None:
    text = text.strip().lower()
    if re.search(r"不一致|不匹配|不是|mismatch|not consistent|incorrect", text):
        return "mismatch"
    if re.search(r"一致|匹配|是|match|consistent|correct", text):
        return "match"
    return None


def text_mentions_region(text: str, region_key: str) -> bool:
    zh = REGION_ZH.get(region_key, "")
    key_tokens = region_key.replace("_", " ")
    if zh and zh in text:
        return True
    return key_tokens.lower() in text.lower()


def side(region_key: str) -> str:
    if region_key.startswith("left"):
        return "left"
    if region_key.startswith("right"):
        return "right"
    if region_key.startswith("bilateral"):
        return "bilateral"
    return "central"


def level(region_key: str) -> str:
    for item in ("upper", "mid", "lower", "hilum", "costophrenic"):
        if item in region_key:
            return item
    if region_key == "cardiomediastinal":
        return "central"
    return "whole"


def summarize_dataset(data: list[dict[str, Any]], image_folder: Path) -> dict[str, Any]:
    label_counts = Counter()
    region_counts = Counter()
    candidate_counts = Counter()
    side_errors = 0
    level_errors = 0
    missing_crops = 0
    bad_format = 0
    answer_label_ok = 0
    answer_region_ok = 0

    for ex in data:
        label = ex.get("grounding_label")
        region = ex.get("grounding_region")
        candidate = ex.get("candidate_region")
        label_counts[label] += 1
        region_counts[region] += 1
        candidate_counts[candidate] += 1

        images = ex.get("image")
        if not isinstance(images, list) or len(images) != 2:
            bad_format += 1
        else:
            crop_path = resolve_image(str(images[1]), image_folder)
            if not str(images[1]).startswith("http") and not crop_path.exists():
                missing_crops += 1

        if label == "mismatch" and region and candidate:
            side_errors += int(side(region) != side(candidate))
            level_errors += int(level(region) != level(candidate))

        ans = final_answer(ex)
        answer_label_ok += int(predict_label_from_text(ans) == label)
        if region:
            answer_region_ok += int(text_mentions_region(ans, region))

    total = len(data)
    return {
        "total": total,
        "label_counts": dict(label_counts),
        "region_counts": dict(region_counts),
        "candidate_counts": dict(candidate_counts),
        "bad_format": bad_format,
        "missing_crops": missing_crops,
        "mismatch_side_error_examples": side_errors,
        "mismatch_level_error_examples": level_errors,
        "answer_label_consistency": round(answer_label_ok / total, 4) if total else 0.0,
        "answer_region_mention_rate": round(answer_region_ok / total, 4) if total else 0.0,
    }


def load_predictions(path: Path) -> dict[str, str]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError("--predictions_json must be a list of objects")
    out: dict[str, str] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        text = row.get("prediction", row.get("text", row.get("output")))
        if row_id is not None and isinstance(text, str):
            out[str(row_id)] = text
    return out


def score_predictions(data: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    n = 0
    label_ok = 0
    region_ok = 0
    left_right_ok = 0
    for ex in data:
        ex_id = str(ex.get("id"))
        pred = predictions.get(ex_id)
        if not pred:
            continue
        n += 1
        label = ex.get("grounding_label")
        region = ex.get("grounding_region")
        pred_label = predict_label_from_text(pred)
        label_ok += int(pred_label == label)
        if region:
            region_ok += int(text_mentions_region(pred, region))
            if side(region) in {"left", "right"}:
                wrong = "右" if side(region) == "left" else "左"
                left_right_ok += int(wrong not in pred)
            else:
                left_right_ok += 1

    return {
        "prediction_count": n,
        "label_accuracy": round(label_ok / n, 4) if n else 0.0,
        "region_mention_accuracy": round(region_ok / n, 4) if n else 0.0,
        "left_right_consistency": round(left_right_ok / n, 4) if n else 0.0,
    }


def write_manual_review(data: list[dict[str, Any]], out_path: Path, sample_size: int, seed: int) -> None:
    rng = random.Random(seed)
    sample = data if sample_size <= 0 or sample_size >= len(data) else rng.sample(data, sample_size)
    rows = []
    for ex in sample:
        rows.append(
            {
                "id": ex.get("id"),
                "image": ex.get("image"),
                "grounding_label": ex.get("grounding_label"),
                "grounding_region": ex.get("grounding_region"),
                "candidate_region": ex.get("candidate_region"),
                "grounding_sentence": ex.get("grounding_sentence"),
                "answer": final_answer(ex),
                "manual_region_correct": None,
                "manual_sentence_supported": None,
                "notes": "",
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_json", required=True, help="Generated grounding SFT JSON.")
    parser.add_argument("--image_folder", required=True, help="Image root used by training.")
    parser.add_argument("--predictions_json", default=None, help="Optional list with id + prediction/text/output.")
    parser.add_argument("--manual_review_json", default=None, help="Optional output sample for manual review.")
    parser.add_argument("--manual_review_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(Path(args.data_json))
    if not isinstance(data, list):
        raise ValueError("--data_json must contain a list")
    image_folder = Path(args.image_folder)
    result = {"dataset": summarize_dataset(data, image_folder)}

    if args.predictions_json:
        result["predictions"] = score_predictions(data, load_predictions(Path(args.predictions_json)))

    if args.manual_review_json:
        write_manual_review(data, Path(args.manual_review_json), args.manual_review_size, args.seed)
        result["manual_review_json"] = os.path.abspath(args.manual_review_json)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
