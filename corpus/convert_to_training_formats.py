# -*- coding: utf-8 -*-
"""
中间 JSONL -> 训练格式：
- qa_sharegpt：question/answer -> ShareGPT conversations（可合并多条为 train_*.jsonl）
- report_alpaca：MedicalReportRecord 字段 -> instruction/input/output
- cpt_sharegpt：将 QA ShareGPT 与「报告伪对话」合并为统一 CLM 语料（conversations）
- cpt_qwen_vl_json：供 Qwen-VL-Series-Finetune train_sft 使用（单文件 JSON 数组；报告含 <image> + image 相对路径）
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterable, List, Optional

from schemas import MedicalQARecord, MedicalReportRecord

# 与 Qwen-VL-Series-Finetune/src/constants.py 中 LLAVA_IMAGE_TOKEN 一致
LLAVA_IMAGE_TOKEN = "<image>"

DEFAULT_REPORT_VL_PROMPT = (
    "请根据这张胸部影像，用中文撰写结构化报告，分「影像所见」与「印象」两段。"
)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def row_to_qa_sharegpt(row: Dict[str, Any], qk: str, ak: str) -> Dict[str, Any]:
    q = str(row.get(qk) or "").strip()
    a = str(row.get(ak) or "").strip()
    if not q or not a:
        return {}
    rec = MedicalQARecord(
        question=q,
        answer=a,
        source=str(row.get("source") or row.get("_source") or ""),
        meta={k: v for k, v in row.items() if k not in (qk, ak, "source", "_source")},
    )
    return rec.to_sharegpt()


def row_to_report_alpaca(row: Dict[str, Any]) -> Dict[str, Any]:
    meta_keys = ("source", "_resume_key", "study_id", "subject_id")
    meta = {k: row[k] for k in meta_keys if k in row and row[k] is not None}
    rec = MedicalReportRecord(
        instruction=str(row.get("instruction") or "").strip()
        or "根据给定影像报告要点，生成规范中文报告（含 Findings 与 Impression 两段）。",
        findings_en=row.get("findings_en"),
        impression_en=row.get("impression_en"),
        findings_zh=row.get("findings_zh"),
        impression_zh=row.get("impression_zh"),
        full_report_zh=row.get("full_report_zh"),
        source=str(row.get("source") or ""),
        meta=meta,
    )
    if not (rec.findings_zh or rec.impression_zh or rec.full_report_zh):
        return {}
    return rec.to_alpaca_sft()


def report_alpaca_to_cpt_conversation(row_alpaca: Dict[str, Any]) -> Dict[str, Any]:
    """单轮：用户给 instruction，助手给 output，用于与 QA ShareGPT 列结构一致。"""
    ins = str(row_alpaca.get("instruction") or "").strip()
    out = str(row_alpaca.get("output") or "").strip()
    if not ins or not out:
        return {}
    return {
        "conversations": [
            {"from": "human", "value": ins},
            {"from": "gpt", "value": out},
        ],
        "meta": row_alpaca.get("meta") or {},
    }


def _resolve_report_image_rel(row: Dict[str, Any]) -> Optional[str]:
    for k in ("image", "path", "image_path"):
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    meta = row.get("meta") or {}
    for k in ("image", "path", "image_path"):
        v = meta.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def row_to_qa_qwen_vl(
    row: Dict[str, Any],
    qk: str,
    ak: str,
    placeholder_image: Optional[str],
) -> Dict[str, Any]:
    """纯文本 QA；若与含图样本混训，需传占位图文件名（相对 image_folder）。"""
    base = row_to_qa_sharegpt(row, qk, ak)
    if not base:
        return {}
    if placeholder_image and str(placeholder_image).strip():
        base = dict(base)
        base["image"] = str(placeholder_image).strip()
        conv = list(base["conversations"])
        first = dict(conv[0])
        first["value"] = f"{LLAVA_IMAGE_TOKEN}\n" + first["value"]
        conv[0] = first
        base["conversations"] = conv
    return base


def row_to_report_qwen_vl(
    row: Dict[str, Any],
    report_vl_prompt: str,
) -> Dict[str, Any]:
    """影像 -> 中文报告；human 为 <image> + 短指令，不用英译 instruction。"""
    img = _resolve_report_image_rel(row)
    alp = row_to_report_alpaca(row)
    if not img or not alp:
        return {}
    out = str(alp.get("output") or "").strip()
    if not out:
        return {}
    human = f"{LLAVA_IMAGE_TOKEN}\n{report_vl_prompt.strip()}"
    return {
        "image": img,
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": out},
        ],
        "meta": {"source": "report_vl", **(alp.get("meta") or {})},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        required=True,
        choices=("qa_sharegpt", "report_alpaca", "cpt_sharegpt", "cpt_qwen_vl_json"),
    )
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        help="可多次指定。cpt_sharegpt / cpt_qwen_vl_json：前若干为 QA JSONL，最后一个为报告；"
        "cpt_qwen_vl_json 仅一条输入时视为仅报告 JSONL",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--question_key", default="question")
    ap.add_argument("--answer_key", default="answer")
    ap.add_argument(
        "--report_vl_prompt",
        default=DEFAULT_REPORT_VL_PROMPT,
        help="Qwen-VL 报告任务 user 文本（已自动加 <image> 前缀）",
    )
    ap.add_argument(
        "--qa_placeholder_image",
        default="",
        help="与报告混训时：QA 样本使用的占位图文件名（相对 Qwen-VL 的 --image_folder），"
        "避免 batch 内部分样本无 pixel_values；纯报告可留空",
    )
    ap.add_argument(
        "--pretty_json",
        action="store_true",
        help="cpt_qwen_vl_json 输出格式化缩进（文件更大）",
    )
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    if args.mode == "qa_sharegpt":
        for path in args.input:
            for row in iter_jsonl(path):
                o = row_to_qa_sharegpt(row, args.question_key, args.answer_key)
                if o:
                    rows.append(o)
    elif args.mode == "report_alpaca":
        for path in args.input:
            for row in iter_jsonl(path):
                o = row_to_report_alpaca(row)
                if o:
                    rows.append(o)
    elif args.mode == "cpt_sharegpt":
        # 除最后一个文件外视为 QA；最后一个视为报告中间表（若只有 1 个文件则仅 QA）
        qa_paths = args.input[:-1] if len(args.input) > 1 else list(args.input)
        report_path = args.input[-1] if len(args.input) > 1 else None
        for path in qa_paths:
            for row in iter_jsonl(path):
                o = row_to_qa_sharegpt(row, args.question_key, args.answer_key)
                if o:
                    rows.append(o)
        if report_path:
            for row in iter_jsonl(report_path):
                alp = row_to_report_alpaca(row)
                if alp:
                    conv = report_alpaca_to_cpt_conversation(alp)
                    if conv:
                        rows.append(conv)
    elif args.mode == "cpt_qwen_vl_json":
        qa_paths = args.input[:-1] if len(args.input) > 1 else []
        report_path = args.input[-1]
        ph = args.qa_placeholder_image.strip() or None
        if qa_paths and not ph:
            print(
                "[warn] 已包含 QA 与影像报告混合：未设置 --qa_placeholder_image 时，"
                "Qwen-VL 的 DataCollator 在同 batch 内可能同时出现有图/无图样本而报错；"
                "请提供一张小图文件名（相对 --image_folder）作为 QA 占位图。",
                file=sys.stderr,
            )
        n_skip_report = 0
        for path in qa_paths:
            for row in iter_jsonl(path):
                o = row_to_qa_qwen_vl(
                    row, args.question_key, args.answer_key, ph
                )
                if o:
                    rows.append(o)
        for row in iter_jsonl(report_path):
            o = row_to_report_qwen_vl(row, args.report_vl_prompt)
            if o:
                rows.append(o)
            else:
                n_skip_report += 1
        indent = 2 if args.pretty_json else None
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=indent)
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "count": len(rows),
                    "report_rows_skipped_no_image_or_output": n_skip_report,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return

    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"mode": args.mode, "count": len(rows)}, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
