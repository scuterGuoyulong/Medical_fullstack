#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrieve medical QA chunks from a hybrid FAISS/BM25 index and build a safe RAG prompt.

Pipeline:
  query -> embedding -> question/doc FAISS recall + BM25 recall -> score fusion
  -> low-quality penalty -> optional reranker -> TopN prompt with safety constraints.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INDEX_DIR = (
    "/home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/"
    "SuperResolution_train_prx/andes_vl/medical_fullstack/rag/indexes/"
    "medical_qa_bge_m3_faiss"
)

RAG_PROMPT_TEMPLATE = """你是医学问答助手。请严格基于下面的参考资料回答用户问题。
如果资料不足，请说明“当前资料不足，建议结合医生面诊或检查结果判断”。
不要编造诊断、药物剂量、检查结论。

医疗安全规则：
1. 不能把参考资料中的“可能”改成“确诊”。
2. 涉及用药、剂量、停药、换药，必须提示遵医嘱。
3. 涉及胸痛、呼吸困难、意识障碍、大出血、孕产急症等，要提示及时就医。
4. 检索不到可靠资料时，不要强答。
5. 回答末尾用“参考依据：片段1、片段2”标明依据，便于审计。

参考资料：
{retrieved_context}

用户问题：
{question}

回答："""

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def tokenize_for_bm25(text: str) -> list[str]:
    try:
        import jieba

        return [t.strip().lower() for t in jieba.lcut(text) if t.strip()]
    except ImportError:
        return [t.group(0).lower() for t in TOKEN_RE.finditer(text)]


def minmax_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo = min(vals)
    hi = max(vals)
    if math.isclose(hi, lo):
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class MedicalQARetriever:
    def __init__(
        self,
        index_dir: str | Path,
        embedding_model: str | None = None,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        reranker_device: str | None = None,
        enable_reranker: bool = True,
        question_weight: float = 0.7,
        doc_weight: float = 0.3,
        bm25_weight: float = 0.25,
        low_quality_penalty: float = 0.85,
        score_threshold: float = 0.0,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.config = self._load_config()
        self.embedding_model_name = embedding_model or self.config.get("embedding_model") or "BAAI/bge-m3"
        self.reranker_model_name = reranker_model
        self.device = device
        self.reranker_device = reranker_device or device
        self.enable_reranker = enable_reranker
        self.question_weight = question_weight
        self.doc_weight = doc_weight
        self.bm25_weight = bm25_weight
        self.low_quality_penalty = low_quality_penalty
        self.score_threshold = score_threshold

        try:
            import faiss
        except ImportError as exc:
            raise SystemExit("缺少依赖 faiss，请先执行: pip install faiss-cpu") from exc

        question_index_path = self.index_dir / "question.index.faiss"
        doc_index_path = self.index_dir / "doc.index.faiss"
        if question_index_path.is_file() and doc_index_path.is_file():
            self.question_index = faiss.read_index(str(question_index_path))
            self.doc_index = faiss.read_index(str(doc_index_path))
        else:
            legacy = faiss.read_index(str(self.index_dir / "index.faiss"))
            self.question_index = legacy
            self.doc_index = legacy
        self.docs = self._load_docs()
        self.bm25 = self._load_bm25()
        self.embedder = self._load_embedder()
        self.reranker = self._load_reranker() if enable_reranker else None

    def _load_config(self) -> dict[str, Any]:
        path = self.index_dir / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 config.json: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_docs(self) -> list[dict[str, Any]]:
        path = self.index_dir / "docs.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 docs.jsonl: {path}")
        docs: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        return docs

    def _load_bm25(self) -> dict[str, Any] | None:
        path = self.index_dir / "bm25.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit("缺少依赖 sentence-transformers，请先执行: pip install sentence-transformers") from exc

        kwargs: dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        return SentenceTransformer(self.embedding_model_name, **kwargs)

    def _load_reranker(self):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise SystemExit("缺少依赖 sentence-transformers，请先执行: pip install sentence-transformers") from exc

        kwargs: dict[str, Any] = {}
        if self.reranker_device:
            kwargs["device"] = self.reranker_device
        return CrossEncoder(self.reranker_model_name, **kwargs)

    def embed_query(self, query: str) -> np.ndarray:
        emb = self.embedder.encode([query], normalize_embeddings=True)
        return np.asarray(emb, dtype="float32")

    def _faiss_scores(self, index, q_emb: np.ndarray, top_k: int) -> dict[int, float]:
        scores, indices = index.search(q_emb, top_k)
        out: dict[int, float] = {}
        for idx, score in zip(indices[0], scores[0], strict=True):
            if idx >= 0:
                out[int(idx)] = float(score)
        return out

    def _bm25_scores(self, query: str, top_k: int) -> dict[int, float]:
        if not self.bm25:
            return {}
        q_tokens = tokenize_for_bm25(query)
        if not q_tokens:
            return {}
        doc_tokens: list[list[str]] = self.bm25["doc_tokens"]
        idf: dict[str, float] = self.bm25["idf"]
        avgdl = float(self.bm25.get("avgdl") or 0.0)
        k1 = float(self.bm25.get("k1", 1.5))
        b = float(self.bm25.get("b", 0.75))

        scores: list[tuple[int, float]] = []
        for doc_idx, toks in enumerate(doc_tokens):
            if not toks:
                continue
            tf: dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            dl = len(toks)
            score = 0.0
            for tok in q_tokens:
                freq = tf.get(tok, 0)
                if freq == 0:
                    continue
                denom = freq + k1 * (1 - b + b * dl / avgdl) if avgdl > 0 else freq + k1
                score += float(idf.get(tok, 0.0)) * (freq * (k1 + 1)) / denom
            if score > 0:
                scores.append((doc_idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return dict(scores[:top_k])

    def search(
        self,
        query: str,
        top_k: int = 8,
        rerank_top_n: int = 3,
        vector_top_k: int | None = None,
        bm25_top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        vector_top_k = vector_top_k or top_k
        bm25_top_k = bm25_top_k or top_k
        q_emb = self.embed_query(query)
        question_scores = self._faiss_scores(self.question_index, q_emb, vector_top_k)
        doc_scores = self._faiss_scores(self.doc_index, q_emb, vector_top_k)
        bm25_scores = self._bm25_scores(query, bm25_top_k)
        bm25_norm = minmax_normalize(bm25_scores)

        candidate_ids = set(question_scores) | set(doc_scores) | set(bm25_scores)
        candidates: list[dict[str, Any]] = []
        for idx in candidate_ids:
            doc = dict(self.docs[idx])
            question_score = question_scores.get(idx, 0.0)
            doc_score = doc_scores.get(idx, 0.0)
            bm25_score = bm25_scores.get(idx, 0.0)
            vector_score = self.question_weight * question_score + self.doc_weight * doc_score
            hybrid_score = vector_score + self.bm25_weight * bm25_norm.get(idx, 0.0)
            quality_weight = self.low_quality_penalty if doc.get("low_quality") else 1.0
            pre_rerank_score = hybrid_score * quality_weight
            doc.update(
                {
                    "faiss_id": idx,
                    "question_score": float(question_score),
                    "doc_score": float(doc_score),
                    "bm25_score": float(bm25_score),
                    "bm25_score_norm": float(bm25_norm.get(idx, 0.0)),
                    "vector_score": float(vector_score),
                    "hybrid_score": float(hybrid_score),
                    "quality_weight": float(quality_weight),
                    "pre_rerank_score": float(pre_rerank_score),
                    # Keep this key for callers that still read faiss_score.
                    "faiss_score": float(pre_rerank_score),
                }
            )
            candidates.append(doc)

        if self.reranker is not None and candidates:
            pairs = [[query, doc["text"]] for doc in candidates]
            rerank_scores = self.reranker.predict(pairs)
            for doc, score in zip(candidates, rerank_scores, strict=True):
                doc["rerank_score"] = float(score)
                doc["final_score"] = float(score) * float(doc["quality_weight"])
            candidates.sort(key=lambda x: x["final_score"], reverse=True)
        else:
            for doc in candidates:
                doc["final_score"] = doc["pre_rerank_score"]
            candidates.sort(key=lambda x: x["final_score"], reverse=True)

        threshold = self.score_threshold if score_threshold is None else score_threshold
        if candidates and candidates[0]["final_score"] < threshold:
            return []
        for i, doc in enumerate(candidates, start=1):
            doc["final_rank"] = i
        return candidates[:rerank_top_n]


def build_retrieved_context(docs: list[dict[str, Any]], max_context_chars: int = 4000) -> str:
    parts: list[str] = []
    used = 0
    for i, doc in enumerate(docs, start=1):
        score = doc.get("final_score", doc.get("rerank_score", doc.get("faiss_score", 0.0)))
        header = (
            f"[片段{i}] doc_id={doc.get('doc_id')} "
            f"line={doc.get('line_no')} score={float(score):.4f}"
        )
        if doc.get("low_quality"):
            header += " low_quality=true"
        block = f"{header}\n{doc['text']}"
        if parts and used + len(block) > max_context_chars:
            break
        parts.append(block)
        used += len(block)
    if not parts:
        return "无可用参考资料。"
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, retrieved_docs: list[dict[str, Any]], max_context_chars: int = 4000) -> str:
    context = build_retrieved_context(retrieved_docs, max_context_chars=max_context_chars)
    return RAG_PROMPT_TEMPLATE.format(retrieved_context=context, question=question)


def retrieve_and_build_prompt(
    index_dir: str | Path,
    question: str,
    top_k: int = 8,
    rerank_top_n: int = 3,
    max_context_chars: int = 4000,
    embedding_model: str | None = None,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    device: str | None = None,
    reranker_device: str | None = None,
    enable_reranker: bool = True,
    question_weight: float = 0.7,
    doc_weight: float = 0.3,
    bm25_weight: float = 0.25,
    low_quality_penalty: float = 0.85,
    score_threshold: float = 0.0,
    vector_top_k: int | None = None,
    bm25_top_k: int | None = None,
) -> dict[str, Any]:
    retriever = MedicalQARetriever(
        index_dir=index_dir,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        device=device,
        reranker_device=reranker_device,
        enable_reranker=enable_reranker,
        question_weight=question_weight,
        doc_weight=doc_weight,
        bm25_weight=bm25_weight,
        low_quality_penalty=low_quality_penalty,
        score_threshold=score_threshold,
    )
    docs = retriever.search(
        question,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k,
        score_threshold=score_threshold,
    )
    context = build_retrieved_context(docs, max_context_chars=max_context_chars)
    prompt = RAG_PROMPT_TEMPLATE.format(retrieved_context=context, question=question)
    return {
        "question": question,
        "top_k": top_k,
        "rerank_top_n": rerank_top_n,
        "retrieved_context": context,
        "prompt": prompt,
        "docs": docs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--vector_top_k", type=int, default=20)
    parser.add_argument("--bm25_top_k", type=int, default=20)
    parser.add_argument("--rerank_top_n", type=int, default=3)
    parser.add_argument("--max_context_chars", type=int, default=4000)
    parser.add_argument("--embedding_model", default="", help="override embedding model from config")
    parser.add_argument("--reranker_model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default="")
    parser.add_argument("--reranker_device", default="")
    parser.add_argument("--no_rerank", action="store_true")
    parser.add_argument("--question_weight", type=float, default=0.7)
    parser.add_argument("--doc_weight", type=float, default=0.3)
    parser.add_argument("--bm25_weight", type=float, default=0.25)
    parser.add_argument("--low_quality_penalty", type=float, default=0.85)
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--print_prompt_only", action="store_true")
    args = parser.parse_args()

    out = retrieve_and_build_prompt(
        index_dir=args.index_dir,
        question=args.query,
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
        max_context_chars=args.max_context_chars,
        embedding_model=args.embedding_model.strip() or None,
        reranker_model=args.reranker_model,
        device=args.device.strip() or None,
        reranker_device=args.reranker_device.strip() or None,
        enable_reranker=not args.no_rerank,
        question_weight=args.question_weight,
        doc_weight=args.doc_weight,
        bm25_weight=args.bm25_weight,
        low_quality_penalty=args.low_quality_penalty,
        score_threshold=args.score_threshold,
        vector_top_k=args.vector_top_k,
        bm25_top_k=args.bm25_top_k,
    )

    if args.print_prompt_only:
        print(out["prompt"])
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
