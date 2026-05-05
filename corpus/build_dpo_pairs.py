# -*- coding: utf-8 -*-
"""
构造 Medical_Qwen/dpo_training.py 所需 JSONL：
字段 system, history, question, response_chosen, response_rejected

- explicit：行内已有 response_rejected（或 answer_rejected）
- synthetic_trunc：用截断/噪声生成弱负例（仅用于打通流水线，正式对齐请用人工或 RM）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Any, Dict, List


def synthetic_rejected(good: str, rng: random.Random) -> str:
    g = good.strip()
    if len(g) < 8:
        return "信息不足，无法回答。"
    cut = max(8, len(g) // 3)
    bad = g[:cut].rstrip()
    if rng.random() < 0.3:
        bad = "根据上述内容，" + bad
    return bad


def _alpaca_prompt_question(instruction: str, inp: str) -> str:
    ins = instruction.strip()
    inp = (inp or "").strip()
    if inp:
        return f"### Instruction:\n{ins}\n### Input:\n{inp}\n### Response:\n"
    return f"### Instruction:\n{ins}\n### Response:\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL")
    ap.add_argument("--output", required=True)
    ap.add_argument("--scenario", choices=("qa", "report"), default="qa")
    ap.add_argument("--mode", choices=("explicit", "synthetic_trunc"), default="synthetic_trunc")
    ap.add_argument("--question_key", default="question")
    ap.add_argument("--chosen_key", default="answer")
    ap.add_argument("--rejected_key", default="answer_rejected")
    ap.add_argument("--instruction_key", default="instruction")
    ap.add_argument("--input_key", default="input")
    ap.add_argument("--output_key", default="output")
    ap.add_argument("--output_rejected_key", default="output_rejected")
    ap.add_argument("--system", default="", help="DPO system 列")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    n = 0

    with open(args.input, "r", encoding="utf-8") as inf, open(
        args.output, "w", encoding="utf-8"
    ) as out:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            row: Dict[str, Any] = json.loads(line)
            hist: List = row.get("history") or []
            if not isinstance(hist, list):
                hist = []

            if args.scenario == "qa":
                q = str(row.get(args.question_key) or "").strip()
                chosen = str(row.get(args.chosen_key) or "").strip()
            else:
                ins = str(row.get(args.instruction_key) or "").strip()
                inp = str(row.get(args.input_key) or "").strip()
                chosen = str(row.get(args.output_key) or "").strip()
                q = _alpaca_prompt_question(ins, inp)
            if not q or not chosen:
                continue
            if args.mode == "explicit":
                rk = (
                    args.rejected_key
                    if args.scenario == "qa"
                    else args.output_rejected_key
                )
                rej = str(row.get(rk) or "").strip()
                if not rej:
                    continue
            else:
                rej = synthetic_rejected(chosen, rng)
            dpo_row = {
                "system": str(row.get("system") or args.system or ""),
                "history": hist,
                "question": q,
                "response_chosen": chosen,
                "response_rejected": rej,
            }
            out.write(json.dumps(dpo_row, ensure_ascii=False) + "\n")
            n += 1

    print(json.dumps({"dpo_pairs": n, "mode": args.mode}, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
