# -*- coding: utf-8 -*-
"""
对比两条 JSONL 的 BLEU-4（需 nltk）：reference 列 vs candidate 列。
用于 SFT/DPO 前后人工填预测列后打表，不替代完整评测集。
"""

from __future__ import annotations

import argparse
import json
import re
import sys

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    _s = SmoothingFunction().method4
except ImportError:
    print("pip install nltk", file=sys.stderr)
    raise


def norm(t: str) -> str:
    t = (t or "").replace("\r", " ")
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="含 reference 列的 JSONL")
    ap.add_argument("--hyps", required=True, help="含 hypothesis 列的 JSONL（与 refs 行序对齐）")
    ap.add_argument("--ref_key", default="output")
    ap.add_argument("--hyp_key", default="prediction")
    args = ap.parse_args()

    refs, hyps = [], []
    with open(args.refs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                refs.append(norm(json.loads(line).get(args.ref_key, "")))
    with open(args.hyps, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                hyps.append(norm(json.loads(line).get(args.hyp_key, "")))
    n = min(len(refs), len(hyps))
    scores = []
    for i in range(n):
        r, h = refs[i], hyps[i]
        if not r or not h:
            continue
        rt = [list(r)]
        pt = list(h)
        scores.append(
            sentence_bleu(rt, pt, weights=(0.25,) * 4, smoothing_function=_s)
        )
    mean_b4 = sum(scores) / len(scores) if scores else 0.0
    print(json.dumps({"pairs": len(scores), "bleu4_mean": mean_b4}, ensure_ascii=False))


if __name__ == "__main__":
    main()
