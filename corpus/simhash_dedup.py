# -*- coding: utf-8 -*-
"""
医疗问答语料 SimHash 近似去重（适合百万级以内；更大请换 MinHash LSH 或 Spark）。
保留策略：同簇保留首条或最长一条（可配）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple


def tokenize_mixed(text: str) -> List[str]:
    if not text:
        return []
    t = text.strip().lower()
    parts = re.findall(r"[\u4e00-\u9fff]{1,3}|[a-z0-9]+", t)
    if len(parts) < 3:
        parts.extend(list(t.replace(" ", "")))
    return parts if parts else ["_"]


def simhash_64(text: str) -> int:
    bits = 64
    v = [0] * bits
    for tok in tokenize_mixed(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dedup_records(
    records: List[Dict],
    text_key: str,
    max_hamming: int = 3,
    keep: str = "longest",
    combine_keys: Optional[List[str]] = None,
) -> Tuple[List[Dict], int]:
    """
    顺序扫描 + 与已保留集合比汉明距离；n 较大时较慢，可先分 shard 再合并。
    """

    def row_text(r: Dict) -> str:
        if combine_keys:
            return " ".join(str(r.get(k) or "") for k in combine_keys).strip()
        return str(r.get(text_key) or "")

    kept: List[Dict] = []
    hashes: List[int] = []
    texts: List[str] = []
    removed = 0

    for r in records:
        tx = row_text(r)
        if not tx.strip():
            continue
        h = simhash_64(tx)
        dup_idx = None
        for j, h2 in enumerate(hashes):
            if hamming(h, h2) <= max_hamming:
                dup_idx = j
                break
        if dup_idx is None:
            kept.append(r)
            hashes.append(h)
            texts.append(tx)
            continue
        removed += 1
        if keep == "longest" and len(tx) > len(texts[dup_idx]):
            kept[dup_idx] = r
            hashes[dup_idx] = h
            texts[dup_idx] = tx

    return kept, removed


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description="SimHash 去重 JSONL（医疗问答等）")
    ap.add_argument("--input", required=True, help="输入 .jsonl")
    ap.add_argument("--output", required=True, help="输出 .jsonl")
    ap.add_argument("--text_key", default="text", help="用于指纹的单字段")
    ap.add_argument(
        "--combine_keys",
        default="",
        help="逗号分隔多字段拼接为文本再做指纹，如 question,answer",
    )
    ap.add_argument("--max_hamming", type=int, default=3)
    ap.add_argument("--keep", choices=("first", "longest"), default="longest")
    args = ap.parse_args()

    recs = list(iter_jsonl(args.input))
    ck = [x.strip() for x in args.combine_keys.split(",") if x.strip()] or None
    text_key = args.text_key if not ck else args.text_key

    kept, removed = dedup_records(
        recs,
        text_key=text_key,
        max_hamming=args.max_hamming,
        keep=args.keep,
        combine_keys=ck,
    )
    with open(args.output, "w", encoding="utf-8") as out:
        for r in kept:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {"input": len(recs), "kept": len(kept), "removed": removed},
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
