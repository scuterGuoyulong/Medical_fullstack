# -*- coding: utf-8 -*-
"""
从大型 JSONL（如 Alpaca 风格 instruction/input/output）中顺序扫描：
  精确去重 → 保留约 N 条 → 可选本地 Qwen 润色答案。

适用 train_zh_0.json 这类「每行一个 JSON」文件；不要求扩展名为 .jsonl。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from typing import Any, Dict, List, Set

# 与 rewrite_quality.py 一致；本脚本避免顶层 import transformers（便于无 GPU/无 mpmath 时先做 --rewrite none）
SYS = (
    "你是医学文本编辑。请在不新增临床事实、不改动核心诊断结论的前提下，"
    "将用户给出的中文问答「答案」润色为更规范、完整、可读的医疗表述。"
    "若原文已足够好，可轻微调整标点与分段。只输出润色后的答案正文，不要重复问题。"
)


def is_low_quality_answer(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 12:
        return True
    if len(t) < 40 and t.count("。") == 0 and t.count(".") == 0:
        return True
    if t:
        mc = max(t.count(c) for c in set(t))
        if mc / len(t) > 0.34:
            return True
    if re.search(r"(测试|test|TODO|待补充|不知道){3,}", t, re.I):
        return True
    return False


def _norm(s: str) -> str:
    return " ".join((s or "").split())


def _dedup_key(row: Dict[str, Any], qk: str, ik: str, ak: str) -> str:
    q = _norm(str(row.get(qk) or ""))
    inp = _norm(str(row.get(ik) or ""))
    a = _norm(str(row.get(ak) or ""))
    raw = f"{q}\n{inp}\n{a}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _collect_unique(
    path: str,
    target: int,
    max_scan: int,
    qk: str,
    ik: str,
    ak: str,
) -> tuple[List[Dict[str, Any]], int, int]:
    seen: Set[str] = set()
    kept: List[Dict[str, Any]] = []
    scanned = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if scanned >= max_scan:
                break
            scanned += 1
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = str(row.get(qk) or "").strip()
            a = str(row.get(ak) or "").strip()
            if not q or not a:
                continue
            dk = _dedup_key(row, qk, ik, ak)
            if dk in seen:
                continue
            seen.add(dk)
            row["_dedup_sha256"] = dk
            kept.append(row)
            if len(kept) >= target:
                break
    return kept, scanned, len(seen)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="大 JSONL：顺序扫描 → 精确去重保留约 N 条 → 可选 Qwen 改写答案",
    )
    ap.add_argument(
        "--input",
        required=True,
        help="输入文件（每行一个 JSON，如 train_zh_0.json）",
    )
    ap.add_argument("--output", required=True, help="输出 JSONL")
    ap.add_argument("--question_key", default="instruction", help="问题字段名")
    ap.add_argument("--input_key", default="input", help="Alpaca input 字段（可空）")
    ap.add_argument("--answer_key", default="output", help="答案字段名")
    ap.add_argument(
        "--target",
        type=int,
        default=100,
        help="去重后保留条数（默认 100）",
    )
    ap.add_argument(
        "--max_scan_lines",
        type=int,
        default=200_000,
        help="最多扫描多少行（防止全文件重复时读太久；默认 20 万行）",
    )
    ap.add_argument(
        "--rewrite",
        choices=("none", "all", "low_quality"),
        default="all",
        help="none 不改写；all 每条答案都润色；low_quality 仅启发式低质量（默认 all）",
    )
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--tokenizer_name_or_path", default=None)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument(
        "--disable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="关闭 Qwen 思考模式（默认开启）",
    )
    ap.add_argument(
        "--clean_instruction",
        choices=("none", "heuristic", "llm"),
        default="none",
        help="清洗 instruction：heuristic 规则去模板/去重；llm 在规则后再提炼核心问句（需模型）",
    )
    args = ap.parse_args()

    kept, scanned, n_unique = _collect_unique(
        args.input,
        target=args.target,
        max_scan=args.max_scan_lines,
        qk=args.question_key,
        ik=args.input_key,
        ak=args.answer_key,
    )

    need_model = args.rewrite != "none" or args.clean_instruction == "llm"
    model = tok = None
    if need_model:
        if not args.model_name_or_path:
            raise SystemExit("--rewrite 非 none 或 --clean_instruction llm 时需要 --model_name_or_path")
        from local_qwen_client import chat_complete, load_model_tokenizer

        model, tok = load_model_tokenizer(
            args.model_name_or_path,
            tokenizer_path=args.tokenizer_name_or_path,
            load_in_4bit=args.load_in_4bit,
        )

    n_polish = 0
    n_q_clean = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for row in kept:
            if args.clean_instruction != "none":
                from instruction_cleanup import process_jsonl_row

                row = process_jsonl_row(
                    row,
                    args.question_key,
                    mode=args.clean_instruction,
                    model=model,
                    tokenizer=tok,
                    llm_max_new_tokens=args.max_new_tokens,
                    llm_temperature=args.temperature,
                    disable_thinking=args.disable_thinking,
                )
                if row.get("_instruction_clean_mode") == "llm":
                    n_q_clean += 1
                elif row.get("_instruction_clean_mode") == "heuristic" and row.get(
                    "_instruction_heuristic_changed"
                ):
                    n_q_clean += 1
            q = str(row.get(args.question_key) or "").strip()
            a = str(row.get(args.answer_key) or "").strip()
            new_a = a
            row["_low_quality_flag"] = is_low_quality_answer(a)
            do_rw = False
            if args.rewrite == "all":
                do_rw = True
            elif args.rewrite == "low_quality":
                do_rw = row["_low_quality_flag"]
            if do_rw and model is not None and tok is not None:
                user = f"问题：{q}\n\n原始答案：{a}\n\n请输出润色后的答案："
                new_a = chat_complete(
                    model,
                    tok,
                    user,
                    system_text=SYS,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    disable_thinking=args.disable_thinking,
                )
                row["_rewritten"] = True
                n_polish += 1
            else:
                row["_rewritten"] = False
            row[args.answer_key] = new_a
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "scanned_lines": scanned,
                "kept_after_dedup": len(kept),
                "target": args.target,
                "rewrite_mode": args.rewrite,
                "clean_instruction": args.clean_instruction,
                "instruction_touched_rows": n_q_clean,
                "llm_rewrites": n_polish,
                "note": "若 kept < target，可提高 --max_scan_lines 或检查重复率",
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    if len(kept) < args.target:
        print(
            f"[warn] 仅得到 {len(kept)} 条（目标 {args.target}），已扫描 {scanned} 行。",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
