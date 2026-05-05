# -*- coding: utf-8 -*-
"""
将「row_idx -> 图像相对路径」合并进 mimic_report_zh.jsonl（写入 meta.image），
便于后续 convert_to_training_formats.py --mode cpt_qwen_vl_json 生成 Qwen-VL 训练数据。

paths_jsonl 每行示例：{"row_idx": 0, "image": "p10/p123/s456789/xxx.jpg"}
或与 export_mimic_cxr_jsonl 导出一致、含 image 字段的行。
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional


def _row_id(row: Dict[str, Any], id_key: str) -> Optional[str]:
    if id_key in row and row[id_key] is not None:
        return str(row[id_key])
    meta = row.get("meta") or {}
    if id_key in meta and meta[id_key] is not None:
        return str(meta[id_key])
    if "_resume_key" in meta:
        return str(meta["_resume_key"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths_jsonl", required=True, help="含 row_idx 与 image/path 的 JSONL")
    ap.add_argument("--report_jsonl", required=True, help="translate_reports 等输出的报告 JSONL")
    ap.add_argument("--output", required=True)
    ap.add_argument("--id_key", default="row_idx")
    args = ap.parse_args()

    id_to_image: Dict[str, str] = {}
    with open(args.paths_jsonl, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = r.get(args.id_key)
            if rid is None:
                continue
            img = r.get("image") or r.get("path") or r.get("image_path")
            if img:
                id_to_image[str(rid)] = str(img).strip()

    n_in = n_out = n_merged = 0
    with open(args.report_jsonl, "r", encoding="utf-8") as inf, open(
        args.output, "w", encoding="utf-8"
    ) as out:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            rid = _row_id(row, args.id_key)
            if rid is not None and rid in id_to_image:
                meta = dict(row.get("meta") or {})
                meta["image"] = id_to_image[rid]
                row["meta"] = meta
                n_merged += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(
        json.dumps(
            {
                "report_lines": n_in,
                "written": n_out,
                "merged_image_meta": n_merged,
                "path_map_size": len(id_to_image),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
