# -*- coding: utf-8 -*-
"""
英文放射/医学报告段落 -> 中文（本地 Qwen）。
输入 JSONL：每行可含 findings / impression 或 findings_en / impression_en（与 MIMIC-CXR 导出一致）。
输出：结构化 MedicalReportRecord 字段 + meta。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, Set

from schemas import MedicalReportRecord

from local_qwen_client import batch_chat_complete, load_model_tokenizer

# 与 Medical_Qwen 训练侧 --disable_thinking 一致：只产出正文，禁止链式思考/英文分析。
SYS = (
    "你是资深放射科医学翻译。将用户给出的英文胸部影像报告片段译为规范、正式的简体中文。"
    "要求：忠实原意；医学术语准确并与常用影像报告用语一致；句式简洁正式。"
    "严禁输出思考过程、步骤说明、英文、拼音、Markdown、列表编号、引号包裹的全段译文、"
    "或“译文如下”“Translation:”等任何非正文内容。只输出一段连续中文，不要小标题。"
)

_FIELD_LABEL_ZH = {
    "findings": "影像所见（Findings）",
    "impression": "印象（Impression）",
}


def _get_finding_impression(row: Dict[str, Any]) -> tuple[str, str]:
    f = row.get("findings_en") or row.get("findings") or ""
    i = row.get("impression_en") or row.get("impression") or ""
    return str(f).strip(), str(i).strip()


def _translate_field(
    model,
    tokenizer,
    field_label: str,
    en_text: str,
    max_new_tokens: int,
    temperature: float,
    *,
    disable_thinking: bool,
) -> str:
    if not en_text:
        return ""
    zh_name = _FIELD_LABEL_ZH.get(field_label, field_label)
    user = (
        f"以下英文为胸部 X 线/CT 报告中的「{zh_name}」段落。"
        f"请译为简体中文报告正文：仅输出译文本身，不要任何前缀或解释。\n\n{en_text}"
    )
    return batch_chat_complete(
        model,
        tokenizer,
        [user],
        system_text=SYS,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        disable_thinking=disable_thinking,
    )[0].strip()


def _compose_full_report_zh(findings_zh: str, impression_zh: str) -> str:
    f = (findings_zh or "").strip()
    i = (impression_zh or "").strip()
    if f and i:
        return f"影像所见：\n{f}\n\n印象：\n{i}"
    return f or i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 JSONL")
    ap.add_argument("--output", required=True, help="输出 JSONL")
    ap.add_argument(
        "--model_name_or_path",
        required=True,
        help="例如本地 Qwen3.5-4B 目录",
    )
    ap.add_argument("--tokenizer_name_or_path", default=None)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument(
        "--instruction",
        default="根据给定影像报告要点，生成规范中文报告（含 Findings 与 Impression 两段）。",
        help="写入 MedicalReportRecord.instruction，供下游 SFT",
    )
    ap.add_argument("--source_tag", default="translated_en_report")
    ap.add_argument(
        "--id_key",
        default="row_idx",
        help="用于断点续跑的主键字段（MIMIC 导出为 row_idx；其他数据可改为 study_id 等）",
    )
    ap.add_argument(
        "--progress_every",
        type=int,
        default=1,
        help="每完成多少条写入就打印一次进度到 stderr（默认 1，便于确认未卡死）",
    )
    ap.add_argument(
        "--disable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="关闭 Qwen 思考模式（与 Medical_Qwen supervised_finetuning --disable_thinking 对齐；默认开启）",
    )
    args = ap.parse_args()

    model, tokenizer = load_model_tokenizer(
        args.model_name_or_path,
        tokenizer_path=args.tokenizer_name_or_path,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    done: Set[str] = set()
    if os.path.isfile(args.output):
        with open(args.output, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    k = str(r.get("_resume_key", ""))
                    if k:
                        done.add(k)
                except json.JSONDecodeError:
                    continue

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    mode = "a" if done else "w"
    out_fp = open(args.output, mode, encoding="utf-8")
    n_ok = 0
    n_skip = 0

    t0 = time.perf_counter()
    with open(args.input, "r", encoding="utf-8") as inf:
        lines = [ln.strip() for ln in inf if ln.strip()]
    n_lines = len(lines)
    print(
        json.dumps(
            {
                "phase": "translate_start",
                "input_lines": n_lines,
                "resume_already_done": len(done),
                "note": "每条样本 2 次 generate(findings+impression)；30633 条约 6.1 万次解码，需数小时属正常",
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )

    first_gen = True
    for idx, line in enumerate(lines):
        row = json.loads(line)
        resume_key = str(row.get(args.id_key, idx))
        if resume_key in done:
            n_skip += 1
            continue
        fe, imp = _get_finding_impression(row)
        if not fe and not imp:
            continue
        if first_gen:
            print(
                "[translate_reports] 正在进行首次推理（CUDA/算子预热，可能 1～5 分钟无新输出）…",
                file=sys.stderr,
                flush=True,
            )
        f_zh = _translate_field(
            model,
            tokenizer,
            "findings",
            fe,
            args.max_new_tokens,
            args.temperature,
            disable_thinking=args.disable_thinking,
        )
        if first_gen:
            print(
                "[translate_reports] findings 首条完成，开始 impression …",
                file=sys.stderr,
                flush=True,
            )
        i_zh = _translate_field(
            model,
            tokenizer,
            "impression",
            imp,
            args.max_new_tokens,
            args.temperature,
            disable_thinking=args.disable_thinking,
        )
        first_gen = False
        full_zh = _compose_full_report_zh(f_zh, i_zh) or None
        rec = MedicalReportRecord(
            instruction=args.instruction,
            findings_en=fe or None,
            impression_en=imp or None,
            findings_zh=f_zh or None,
            impression_zh=i_zh or None,
            full_report_zh=full_zh,
            source=args.source_tag,
            meta={
                **{
                    k: v
                    for k, v in row.items()
                    if k not in ("findings", "impression", "findings_en", "impression_en")
                },
                "_resume_key": resume_key,
            },
        )
        d = {**asdict(rec), "_resume_key": resume_key}
        out_fp.write(json.dumps(d, ensure_ascii=False) + "\n")
        out_fp.flush()
        n_ok += 1
        pe = max(1, args.progress_every)
        if n_ok % pe == 0:
            elapsed = time.perf_counter() - t0
            print(
                json.dumps(
                    {
                        "processed": n_ok,
                        "skipped_resume": n_skip,
                        "elapsed_sec": round(elapsed, 1),
                        "avg_sec_per_row": round(elapsed / n_ok, 2) if n_ok else 0,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )

    out_fp.close()
    print(json.dumps({"written": n_ok, "skipped_resume": n_skip}, ensure_ascii=False))


if __name__ == "__main__":
    main()
