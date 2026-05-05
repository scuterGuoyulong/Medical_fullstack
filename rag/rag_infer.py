# -*- coding: utf-8 -*-
"""
检索增强：加载 index_kb.py 生成的 pkl，对 query 取 top-k 片段，拼入 system 或 user 前缀。
可与 Medical_Qwen/inference.py 或 vLLM 部署组合使用。
"""

from __future__ import annotations

import argparse
import json
import pickle
from typing import Any, Dict, List, Tuple


def load_index(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def search_char_overlap(index: Dict[str, Any], query: str, top_k: int) -> List[Tuple[int, float]]:
    qs = set(query)
    scores = []
    for i, cs in enumerate(index["char_sets"]):
        scores.append((i, float(len(qs & cs))))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def search_tfidf(index: Dict[str, Any], query: str, top_k: int) -> List[Tuple[int, float]]:
    from sklearn.metrics.pairwise import cosine_similarity

    vec = index["vectorizer"]
    mat = index["matrix"]
    q = vec.transform([query])
    sims = cosine_similarity(q, mat)[0]
    idx = sims.argsort()[::-1][:top_k]
    return [(int(i), float(sims[i])) for i in idx]


def search_st(index: Dict[str, Any], query: str, top_k: int) -> List[Tuple[int, float]]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(index["model_name"])
    q = model.encode([query], normalize_embeddings=True)[0]
    emb = index["embeddings"]
    sims = np.dot(emb, q)
    idx = sims.argsort()[::-1][:top_k]
    return [(int(i), float(sims[i])) for i in idx]


def retrieve(
    index_path: str,
    query: str,
    top_k: int = 4,
    max_context_chars: int = 3000,
) -> Dict[str, Any]:
    """
    供 Python/FastAPI 调用：加载 index_kb.py 生成的 pkl，返回检索上下文与拼接好的 user 前缀。
    """
    index = load_index(index_path)
    t = index["type"]
    if t == "tfidf":
        pairs = search_tfidf(index, query, top_k)
    elif t == "char_overlap":
        pairs = search_char_overlap(index, query, top_k)
    else:
        pairs = search_st(index, query, top_k)
    ctx = build_context(index, pairs, max_context_chars)
    user_prefix = (
        "以下是权威知识库摘录，请优先依据摘录作答；摘录不足时再依赖模型知识。\n\n"
        + ctx
        + "\n\n用户问题：\n"
    )
    return {
        "retrieved_scores": pairs,
        "context": ctx,
        "user_prefix": user_prefix,
    }


def build_context(index: Dict[str, Any], pairs: List[Tuple[int, float]], max_chars: int) -> str:
    chunks: List[str] = index["chunks"]
    meta: List[Dict] = index["meta"]
    parts: List[str] = []
    used = 0
    for i, sc in pairs:
        if sc <= 0:
            continue
        m = meta[i]
        header = f"[{m.get('source')} #{m.get('chunk_id')}]"
        block = f"{header}\n{chunks[i]}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_pkl", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--max_context_chars", type=int, default=3000)
    args = ap.parse_args()

    out = retrieve(
        args.index_pkl,
        args.query,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
    )
    printable = dict(out)
    printable["retrieved_scores"] = [[int(i), float(s)] for i, s in out["retrieved_scores"]]
    print(json.dumps(printable, ensure_ascii=False))


if __name__ == "__main__":
    main()
