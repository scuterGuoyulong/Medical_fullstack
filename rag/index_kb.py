# -*- coding: utf-8 -*-
"""
医疗知识库切块 + 检索索引（默认 sklearn TF-IDF；可选 sentence-transformers + 内积）。
用法：python index_kb.py --kb_dir /path/to/txt --out index.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from typing import Any, Dict, List, Tuple

_CHUNK = re.compile(r"\n{2,}|\r\n\r\n")


def read_text_files(root: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            if not name.lower().endswith((".txt", ".md")):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                out.append((rel, f.read()))
    return out


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    paras = [p.strip() for p in _CHUNK.split(text) if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= max_chars:
                buf = p
            else:
                step = max_chars - overlap
                for i in range(0, len(p), step):
                    chunks.append(p[i : i + max_chars])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def build_tfidf_index(chunks: List[str], meta: List[Dict[str, Any]]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=200_000,
        min_df=1,
    )
    mat = vec.fit_transform(chunks)
    return {"type": "tfidf", "vectorizer": vec, "matrix": mat, "meta": meta}


def build_char_overlap_index(chunks: List[str], meta: List[Dict[str, Any]]):
    """无 sklearn 时的轻量回退：按字符集合重叠度检索（适合小规模 KB）。"""
    sets = [set(c) for c in chunks]
    return {"type": "char_overlap", "char_sets": sets, "meta": meta}


def build_st_index(chunks: List[str], meta: List[Dict[str, Any]], model_name: str):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    return {"type": "st", "model_name": model_name, "embeddings": emb, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_chars", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument(
        "--backend",
        choices=("tfidf", "char_overlap", "sentence_transformers"),
        default="tfidf",
    )
    ap.add_argument("--st_model", default="BAAI/bge-small-zh-v1.5")
    args = ap.parse_args()

    files = read_text_files(args.kb_dir)
    if not files:
        print(json.dumps({"error": "no txt/md under kb_dir"}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    chunks: List[str] = []
    meta: List[Dict[str, Any]] = []
    for rel, text in files:
        for i, ch in enumerate(chunk_text(text, args.max_chars, args.overlap)):
            chunks.append(ch)
            meta.append({"source": rel, "chunk_id": i})

    if args.backend == "sentence_transformers":
        index = build_st_index(chunks, meta, args.st_model)
    elif args.backend == "char_overlap":
        index = build_char_overlap_index(chunks, meta)
    else:
        try:
            index = build_tfidf_index(chunks, meta)
        except ImportError:
            print(
                json.dumps(
                    {"warn": "sklearn missing, fallback char_overlap"},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            index = build_char_overlap_index(chunks, meta)

    index["chunks"] = chunks
    with open(args.out, "wb") as f:
        pickle.dump(index, f)
    print(json.dumps({"chunks": len(chunks), "files": len(files), "out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
