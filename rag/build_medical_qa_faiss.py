#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a hybrid FAISS/BM25 knowledge base from medical QA JSONL.

Each QA row is stored as one chunk:
  问题：{instruction}
  补充信息：{input}
  答案：{output}

The index contains:
  question.index.faiss: embedding of instruction only
  doc.index.faiss: embedding of question + answer
  bm25.json: tokenized corpus for keyword/BM25 retrieval

Default embedding model: BAAI/bge-m3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATA = (
    "/home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/"
    "SuperResolution_train_prx/andes_vl/DataSets/medical/finetune/"
    "train_zh_0_sample100_rewrite_llm.jsonl"
)
DEFAULT_OUT = (
    "/home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/"
    "SuperResolution_train_prx/andes_vl/medical_fullstack/rag/indexes/"
    "medical_qa_bge_m3_faiss"
)

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def stable_id(row: dict[str, Any], line_no: int) -> str:
    raw = row.get("_dedup_sha256")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    text = f"{row.get('instruction', '')}\n{row.get('input', '')}\n{row.get('output', '')}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{digest}_{line_no}"


def build_chunk(row: dict[str, Any]) -> str:
    question = str(row.get("instruction", "")).strip()
    extra = str(row.get("input", "")).strip()
    answer = str(row.get("output", "")).strip()

    parts = [f"问题：{question}"]
    if extra:
        parts.append(f"补充信息：{extra}")
    parts.append(f"答案：{answer}")
    return "\n".join(parts)


def tokenize_for_bm25(text: str) -> list[str]:
    try:
        import jieba

        tokens = [t.strip().lower() for t in jieba.lcut(text) if t.strip()]
    except ImportError:
        tokens = [t.group(0).lower() for t in TOKEN_RE.finditer(text)]
    return tokens


def build_bm25_payload(docs: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> dict[str, Any]:
    corpus_tokens = [tokenize_for_bm25(d["text"]) for d in docs]
    doc_freq: dict[str, int] = {}
    for toks in corpus_tokens:
        for tok in set(toks):
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    n_docs = len(corpus_tokens)
    avgdl = sum(len(toks) for toks in corpus_tokens) / n_docs if n_docs else 0.0
    idf = {
        tok: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        for tok, df in doc_freq.items()
    }
    return {
        "type": "bm25_tokenized_corpus",
        "tokenizer": "jieba_if_available_else_regex",
        "k1": k1,
        "b": b,
        "avgdl": avgdl,
        "doc_tokens": corpus_tokens,
        "idf": idf,
    }


def load_qa_jsonl(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = str(row.get("instruction", "")).strip()
            answer = str(row.get("output", "")).strip()
            if not question or not answer:
                continue

            doc_id = stable_id(row, line_no)
            docs.append(
                {
                    "doc_id": doc_id,
                    "text": build_chunk(row),
                    "question": question,
                    "answer": answer,
                    "input": str(row.get("input", "")).strip(),
                    "line_no": line_no,
                    "low_quality": bool(row.get("_low_quality_flag", False)),
                    "metadata": {
                        "_dedup_sha256": row.get("_dedup_sha256"),
                        "_low_quality_flag": row.get("_low_quality_flag"),
                        "_rewritten": row.get("_rewritten"),
                        "_instruction_heuristic_changed": row.get("_instruction_heuristic_changed"),
                        "_instruction_cleaned": row.get("_instruction_cleaned"),
                        "_instruction_clean_mode": row.get("_instruction_clean_mode"),
                    },
                }
            )
    return docs


def encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: str | None,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("缺少依赖 sentence-transformers，请先执行: pip install sentence-transformers") from exc

    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    model = SentenceTransformer(model_name, **kwargs)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(emb, dtype="float32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_jsonl", default=DEFAULT_DATA, help="medical QA JSONL path")
    parser.add_argument("--out_dir", default=DEFAULT_OUT, help="directory to write index.faiss/docs.jsonl/config.json")
    parser.add_argument("--embedding_model", default="BAAI/bge-m3")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="", help='optional sentence-transformers device, e.g. "cuda:0" or "cpu"')
    args = parser.parse_args()

    data_path = Path(args.data_jsonl)
    out_dir = Path(args.out_dir)
    if not data_path.is_file():
        raise SystemExit(f"找不到数据文件: {data_path}")

    docs = load_qa_jsonl(data_path)
    if not docs:
        raise SystemExit("没有可入库的 QA 样本")

    doc_embeddings = encode_texts(
        [d["text"] for d in docs],
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        device=args.device.strip() or None,
    )
    question_embeddings = encode_texts(
        [d["question"] for d in docs],
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        device=args.device.strip() or None,
    )

    try:
        import faiss
    except ImportError as exc:
        raise SystemExit("缺少依赖 faiss，请先执行: pip install faiss-cpu") from exc

    doc_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    doc_index.add(doc_embeddings)
    question_index = faiss.IndexFlatIP(question_embeddings.shape[1])
    question_index.add(question_embeddings)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(question_index, str(out_dir / "question.index.faiss"))
    faiss.write_index(doc_index, str(out_dir / "doc.index.faiss"))
    # Backward-compatible alias for older tools that expected a single index.
    faiss.write_index(doc_index, str(out_dir / "index.faiss"))
    with (out_dir / "docs.jsonl").open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    (out_dir / "bm25.json").write_text(
        json.dumps(build_bm25_payload(docs), ensure_ascii=False),
        encoding="utf-8",
    )

    config = {
        "type": "medical_qa_hybrid_faiss_bm25",
        "data_jsonl": str(data_path),
        "embedding_model": args.embedding_model,
        "normalize_embeddings": True,
        "faiss_metric": "inner_product",
        "chunk_policy": "one_qa_per_chunk_all_rows",
        "vector_score_weights": {
            "question_embedding": 0.7,
            "doc_embedding": 0.3,
        },
        "quality_control": {
            "low_quality_penalty": 0.85,
        },
        "num_docs": len(docs),
        "embedding_dim": int(doc_embeddings.shape[1]),
        "files": {
            "question_faiss": "question.index.faiss",
            "doc_faiss": "doc.index.faiss",
            "legacy_faiss": "index.faiss",
            "bm25": "bm25.json",
            "docs": "docs.jsonl",
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "docs": len(docs),
                "low_quality_docs": sum(1 for d in docs if d["low_quality"]),
                "embedding_model": args.embedding_model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
