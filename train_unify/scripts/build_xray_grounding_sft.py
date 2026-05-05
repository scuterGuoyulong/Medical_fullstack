#!/usr/bin/env python3
"""
Build weakly supervised chest X-ray grounding SFT data.

The script turns image-report pairs into the repository's native LLaVA-style
multi-image SFT JSON:

  image: [full_xray, region_crop]
  conversations[0].value: contains two <image> placeholders and one report sentence
  conversations[1].reasoning: Qwen3.5/Qwen3-VL-Thinking CoT supervision
  conversations[1].value: concise final grounding decision

It does not require bbox annotations. Regions are pseudo-crops from a fixed
chest X-ray anatomy template, so the output should be treated as weak labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("缺少依赖 Pillow，请先执行: pip install pillow") from exc


REPORT_FIELDS = (
    "report",
    "report_text",
    "findings",
    "impression",
    "conclusion",
    "text",
)


ABNORMALITY_PATTERNS = [
    r"\bopacity\b",
    r"\bopacities\b",
    r"\binfiltrate\w*\b",
    r"\bconsolidation\b",
    r"\batelectasis\b",
    r"\beffusion\b",
    r"\bpneumothorax\b",
    r"\bedema\b",
    r"\bcardiomegaly\b",
    r"\benlarg\w+\b",
    r"\bnodule\w*\b",
    r"\bmass\b",
    r"\bfracture\b",
    r"\bpleural\b",
    r"\bairspace\b",
    r"\bvascular congestion\b",
    r"\bclear\b",
    r"\bnormal\b",
    r"\bno\b",
    r"\bwithout\b",
    r"阴影",
    r"渗出",
    r"实变",
    r"不张",
    r"积液",
    r"气胸",
    r"水肿",
    r"心影",
    r"心脏",
    r"结节",
    r"肿块",
    r"骨折",
    r"胸膜",
    r"未见",
    r"无明显",
]


SIDE_TERMS = {
    "left": ["left", "lt", "左"],
    "right": ["right", "rt", "右"],
    "bilateral": ["bilateral", "both", "bibasal", "bibasilar", "双侧", "两侧"],
}

VERTICAL_TERMS = {
    "upper": ["upper", "apical", "apex", "上"],
    "mid": ["middle", "mid", "perihilar", "hilar", "中", "肺门"],
    "lower": ["lower", "basal", "base", "basilar", "costophrenic", "下", "基底", "肋膈角"],
}

ANATOMY_TERMS = {
    "cardiomediastinal": ["cardiomediastinal", "mediastinum", "heart", "cardiac", "心影", "纵隔", "心脏"],
    "left_costophrenic_angle": ["left costophrenic", "left cp angle", "左肋膈角"],
    "right_costophrenic_angle": ["right costophrenic", "right cp angle", "右肋膈角"],
    "left_hilum": ["left hilar", "left hilum", "左肺门"],
    "right_hilum": ["right hilar", "right hilum", "右肺门"],
}


@dataclass(frozen=True)
class Region:
    key: str
    zh_name: str
    en_name: str
    box: tuple[float, float, float, float]


def _regions_for_convention(radiology_view: bool) -> dict[str, Region]:
    """Return normalized crop boxes.

    radiology_view=True means the patient's left is displayed on image right,
    which is the common CXR convention.
    """

    left_x = (0.50, 0.96) if radiology_view else (0.04, 0.50)
    right_x = (0.04, 0.50) if radiology_view else (0.50, 0.96)
    verticals = {
        "upper": (0.10, 0.43),
        "mid": (0.33, 0.68),
        "lower": (0.58, 0.92),
    }

    regions: dict[str, Region] = {}
    for side, x_range, side_zh, side_en in (
        ("left", left_x, "左", "left"),
        ("right", right_x, "右", "right"),
    ):
        for level, y_range, level_zh, level_en in (
            ("upper", verticals["upper"], "上肺野", "upper lung zone"),
            ("mid", verticals["mid"], "中肺野", "middle lung zone"),
            ("lower", verticals["lower"], "下肺野", "lower lung zone"),
        ):
            key = f"{side}_{level}_lung"
            regions[key] = Region(
                key=key,
                zh_name=f"{side_zh}{level_zh}",
                en_name=f"{side_en} {level_en}",
                box=(x_range[0], y_range[0], x_range[1], y_range[1]),
            )

        regions[f"{side}_lung"] = Region(
            key=f"{side}_lung",
            zh_name=f"{side_zh}肺野",
            en_name=f"{side_en} lung",
            box=(x_range[0], 0.10, x_range[1], 0.92),
        )

    regions.update(
        {
            "bilateral_lungs": Region("bilateral_lungs", "双肺野", "bilateral lungs", (0.04, 0.10, 0.96, 0.92)),
            "cardiomediastinal": Region("cardiomediastinal", "心影纵隔区", "cardiomediastinal region", (0.30, 0.30, 0.70, 0.86)),
            "left_costophrenic_angle": Region("left_costophrenic_angle", "左肋膈角", "left costophrenic angle", (left_x[0], 0.70, left_x[1], 0.96)),
            "right_costophrenic_angle": Region("right_costophrenic_angle", "右肋膈角", "right costophrenic angle", (right_x[0], 0.70, right_x[1], 0.96)),
            "left_hilum": Region("left_hilum", "左肺门", "left hilum", (left_x[0] + 0.04, 0.34, left_x[1] - 0.08, 0.62)),
            "right_hilum": Region("right_hilum", "右肺门", "right hilum", (right_x[0] + 0.08, 0.34, right_x[1] - 0.04, 0.62)),
        }
    )
    return regions


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("data", "items", "records"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"JSON object must contain one of data/items/records: {path}")
    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON root in {path}: {type(data)!r}")
    return data


def first_image(image_field: Any) -> str | None:
    if isinstance(image_field, list) and image_field:
        return str(image_field[0])
    if isinstance(image_field, str) and image_field.strip():
        return image_field.strip()
    return None


def is_placeholder_image(image_name: str) -> bool:
    return Path(image_name).name.lower() == "qa_placeholder.jpg"


def extract_report(record: dict[str, Any]) -> str:
    chunks: list[str] = []
    for field in REPORT_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
    if chunks:
        return "\n".join(chunks)

    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            if turn.get("from") in {"gpt", "assistant"} and isinstance(turn.get("value"), str):
                chunks.append(turn["value"].strip())
    return "\n".join(x for x in chunks if x)


def split_report_sentences(report: str) -> list[str]:
    report = re.sub(r"\b(FINDINGS|IMPRESSION|CONCLUSION)\s*:", " ", report, flags=re.IGNORECASE)
    report = report.replace("\r", "\n")
    parts = re.split(r"(?<=[.!?。！？；;])\s+|\n+|(?<=。)", report)
    sentences: list[str] = []
    for part in parts:
        s = re.sub(r"\s+", " ", part).strip(" -\t")
        if 8 <= len(s) <= 260:
            sentences.append(s)
    return sentences


def contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def term_hit(text: str, terms: Iterable[str]) -> bool:
    text_l = text.lower()
    return any(term.lower() in text_l for term in terms)


def infer_regions(sentence: str, regions: dict[str, Region]) -> list[str]:
    candidates: list[str] = []

    for region_key, terms in ANATOMY_TERMS.items():
        if region_key in regions and term_hit(sentence, terms):
            candidates.append(region_key)

    side = None
    for side_key, terms in SIDE_TERMS.items():
        if term_hit(sentence, terms):
            side = side_key
            break

    level = None
    for level_key, terms in VERTICAL_TERMS.items():
        if term_hit(sentence, terms):
            level = level_key
            break

    if side == "bilateral":
        candidates.append("bilateral_lungs")
    elif side in {"left", "right"} and level:
        candidates.append(f"{side}_{level}_lung")
    elif side in {"left", "right"}:
        candidates.append(f"{side}_lung")
    elif level:
        candidates.extend([f"left_{level}_lung", f"right_{level}_lung"])

    seen = set()
    ordered = []
    for key in candidates:
        if key in regions and key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def resolve_image_path(image_name: str, image_folder: Path) -> Path:
    p = Path(image_name)
    if p.is_absolute():
        return p
    return image_folder / p


def rel_to_folder(path: Path, folder: Path) -> str:
    try:
        return path.resolve().relative_to(folder.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def crop_region(image_path: Path, region: Region, crop_root: Path, prefix: str) -> Path:
    crop_root.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = region.box
        box = (
            max(0, int(round(x1 * w))),
            max(0, int(round(y1 * h))),
            min(w, int(round(x2 * w))),
            min(h, int(round(y2 * h))),
        )
        crop = img.crop(box)
        out_path = crop_root / f"{prefix}_{region.key}.png"
        crop.save(out_path)
    return out_path


def stable_id(*parts: str) -> str:
    raw = "||".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def build_reasoning(sentence: str, region: Region, is_match: bool, negative_region: Region | None = None) -> str:
    if is_match:
        return (
            f"先解析报告句子：{sentence}\n"
            f"句子中的空间或解剖线索指向 {region.zh_name}（{region.en_name}）。"
            f"第二张图是从胸片中裁剪出的候选区域，覆盖该解剖位置。"
            "因此这句话和第二张局部图在空间上是一致的，可以把报告结论绑定到该区域。"
        )
    assert negative_region is not None
    return (
        f"先解析报告句子：{sentence}\n"
        f"句子中的空间或解剖线索指向 {region.zh_name}（{region.en_name}）。"
        f"但第二张图对应的是 {negative_region.zh_name}（{negative_region.en_name}），"
        "与句子描述的空间位置不一致。因此不能把该报告结论绑定到第二张局部图。"
    )


def build_prompt(sentence: str) -> str:
    return (
        "<image>\n<image>\n"
        "第一张是胸片全图，第二张是候选局部区域。"
        "请判断下面报告句子是否主要对应第二张局部区域。"
        "回答时先在 reasoning 中解析空间位置、图像区域和结论绑定关系，最终答案只给出是否一致、对应区域和结论。\n"
        f"报告句子：{sentence}"
    )


def build_example(
    record_id: str,
    full_image_rel: str,
    crop_rel: str,
    sentence: str,
    target_region: Region,
    label: str,
    negative_region: Region | None = None,
) -> dict[str, Any]:
    is_match = label == "match"
    final = (
        f"一致。对应区域：{target_region.zh_name}。结论：{sentence}"
        if is_match
        else f"不一致。句子对应区域：{target_region.zh_name}。第二张图不是该结论的主要对应区域。"
    )
    return {
        "id": record_id,
        "image": [full_image_rel, crop_rel],
        "grounding_label": label,
        "grounding_sentence": sentence,
        "grounding_region": target_region.key,
        "candidate_region": target_region.key if is_match else negative_region.key,
        "conversations": [
            {"from": "human", "value": build_prompt(sentence)},
            {
                "from": "gpt",
                "reasoning": build_reasoning(sentence, target_region, is_match, negative_region),
                "value": final,
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_json", required=True, help="Input JSON/JSONL/CSV with image-report pairs or SFT conversations.")
    parser.add_argument("--image_folder", required=True, help="Root folder for relative input image paths.")
    parser.add_argument("--output_json", required=True, help="Output LLaVA-style SFT JSON.")
    parser.add_argument("--crop_dir", default=None, help="Where to write region crops. Default: <image_folder>/xray_grounding_crops")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit input records before expansion. 0 means all.")
    parser.add_argument("--max_sentences_per_report", type=int, default=6)
    parser.add_argument("--negative_per_positive", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radiology_view", action=argparse.BooleanOptionalAction, default=True, help="Patient left is image right.")
    parser.add_argument("--include_normal_sentences", action="store_true", help="Keep localized normal/no-finding sentences too.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_path = Path(args.input_json)
    image_folder = Path(args.image_folder)
    crop_dir = Path(args.crop_dir) if args.crop_dir else image_folder / "xray_grounding_crops"
    regions = _regions_for_convention(args.radiology_view)
    records = load_records(input_path)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    examples: list[dict[str, Any]] = []
    stats = {
        "records": len(records),
        "records_with_report": 0,
        "localized_sentences": 0,
        "positive_examples": 0,
        "negative_examples": 0,
        "missing_images": 0,
        "skipped_placeholder_images": 0,
    }

    region_keys = list(regions)
    for idx, record in enumerate(records):
        image_name = first_image(record.get("image") or record.get("images") or record.get("path"))
        if not image_name:
            continue
        if is_placeholder_image(image_name):
            stats["skipped_placeholder_images"] += 1
            continue
        image_path = resolve_image_path(image_name, image_folder)
        if not image_path.exists():
            stats["missing_images"] += 1
            continue

        report = extract_report(record)
        if not report:
            continue
        stats["records_with_report"] += 1

        base_id = str(record.get("id") or record.get("study_id") or image_path.stem or idx)
        full_image_rel = image_name if not Path(image_name).is_absolute() else rel_to_folder(image_path, image_folder)
        sentences_added = 0
        for sentence in split_report_sentences(report):
            inferred = infer_regions(sentence, regions)
            if not inferred:
                continue
            if not args.include_normal_sentences and not contains_any(sentence, ABNORMALITY_PATTERNS):
                continue
            for region_key in inferred:
                target = regions[region_key]
                prefix = f"{base_id}_{stable_id(sentence, target.key)}"
                crop_path = crop_region(image_path, target, crop_dir, prefix)
                crop_rel = rel_to_folder(crop_path, image_folder)
                ex_id = f"{base_id}_pos_{stable_id(sentence, target.key)}"
                examples.append(build_example(ex_id, full_image_rel, crop_rel, sentence, target, "match"))
                stats["positive_examples"] += 1

                negative_pool = [k for k in region_keys if k != target.key and not (target.key.startswith("left") and k.startswith("left"))]
                rng.shuffle(negative_pool)
                for neg_key in negative_pool[: max(0, args.negative_per_positive)]:
                    negative = regions[neg_key]
                    neg_prefix = f"{base_id}_{stable_id(sentence, target.key, negative.key)}"
                    neg_crop_path = crop_region(image_path, negative, crop_dir, neg_prefix)
                    neg_crop_rel = rel_to_folder(neg_crop_path, image_folder)
                    neg_id = f"{base_id}_neg_{stable_id(sentence, target.key, negative.key)}"
                    examples.append(build_example(neg_id, full_image_rel, neg_crop_rel, sentence, target, "mismatch", negative))
                    stats["negative_examples"] += 1

                stats["localized_sentences"] += 1
                sentences_added += 1
                if sentences_added >= args.max_sentences_per_report:
                    break
            if sentences_added >= args.max_sentences_per_report:
                break

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**stats, "output_examples": len(examples), "output_json": str(output_path), "crop_dir": str(crop_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
