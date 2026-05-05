# -*- coding: utf-8 -*-
"""
Qwen-VL 推理 + 医疗知识库 RAG（TF-IDF / sentence-transformers 等 index_kb 产物）。

依赖（在已安装 Qwen-VL-Series-Finetune 训练环境基础上）:
  pip install fastapi uvicorn qwen-vl-utils

启动（见同目录 run_vlm_rag_server.sh）:
  在 andes_vl 下设置 VLM_MODEL_PATH、可选 RAG_INDEX_PKL，再运行本文件。

说明:
  - 检索只对纯文本 query；图像仅作为 VLM 多模态输入，不参与向量检索。
  - image_path 必须为服务端可读本地路径（内网服务用；勿对公网暴露任意路径读取）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Qwen-VL-Series-Finetune/src 加入 path，以加载 utils / model
_SERVE_DIR = Path(__file__).resolve().parent
_MEDICAL_FULLSTACK = _SERVE_DIR.parent
_ANDES_VL_ROOT = _MEDICAL_FULLSTACK.parent
_QWEN_SRC = _ANDES_VL_ROOT / "Qwen-VL-Series-Finetune" / "src"
if _QWEN_SRC.is_dir():
    sys.path.insert(0, str(_QWEN_SRC))
else:
    raise RuntimeError(f"未找到 Qwen-VL 源码目录: {_QWEN_SRC}")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qwen_vl_utils import process_vision_info

from utils import (
    disable_torch_init,
    get_model_name_from_path,
    load_pretrained_model,
)

# RAG 模块（与 vlm 分离，仅文本检索）
sys.path.insert(0, str(_MEDICAL_FULLSTACK))
from rag.rag_infer import retrieve  # noqa: E402
from rag.medical_qa_rag import MedicalQARetriever, build_prompt, build_retrieved_context  # noqa: E402

app = FastAPI(title="Qwen-VL + Medical RAG", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = None
model = None
device: str = "cuda"
_default_rag_pkl: Optional[str] = None
_default_rag_index_dir: Optional[str] = None
_medical_qa_retriever: Optional[MedicalQARetriever] = None


class InferBody(BaseModel):
    query: str = Field(..., description="用户问题（用于检索与生成）")
    image_path: Optional[str] = Field(None, description="服务端本地图像路径，可选")
    use_rag: bool = Field(True, description="是否使用启动时配置的索引做 RAG")
    top_k: int = Field(4, ge=1, le=32)
    vector_top_k: int = Field(20, ge=1, le=128)
    bm25_top_k: int = Field(20, ge=1, le=128)
    rerank_top_n: int = Field(3, ge=1, le=16)
    score_threshold: float = Field(0.0, ge=-10.0, le=10.0)
    max_context_chars: int = Field(3000, ge=500, le=16000)
    max_new_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)


class InferResponse(BaseModel):
    text: str
    rag_context: Optional[str] = None
    retrieved_scores: Optional[List[List[Any]]] = None


def _build_conversation(
    query: str,
    image_path: Optional[str],
    rag_context: Optional[str],
    rag_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    system = "你是专业医疗助手，回答须准确、谨慎；不确定时请说明并建议线下就医。"
    user_text = rag_prompt or query
    if rag_context and not rag_prompt:
        user_text = (
            "【知识库摘录】\n"
            + rag_context
            + "\n\n【问题】\n"
            + query
            + "\n请结合摘录与图像（如有）作答。"
        )
    user_content: List[Dict[str, Any]] = []
    if image_path:
        user_content.append({"type": "image", "image": image_path})
    user_content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


@torch.inference_mode()
def _generate_one(conversation: List[Dict[str, Any]], body: InferBody) -> str:
    assert processor is not None and model is not None
    prompt = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    input_len = inputs["input_ids"].shape[1]
    do_sample = body.temperature > 0
    out = model.generate(
        **inputs,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature if do_sample else None,
        do_sample=do_sample,
        repetition_penalty=body.repetition_penalty,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    new_tokens = out[0, input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


@app.on_event("startup")
def _startup() -> None:
    global processor, model, device, _default_rag_pkl, _default_rag_index_dir, _medical_qa_retriever
    mp = os.environ.get("VLM_MODEL_PATH", "").strip()
    mb = os.environ.get("VLM_MODEL_BASE", "").strip() or None
    device = os.environ.get("VLM_DEVICE", "cuda")
    _default_rag_pkl = os.environ.get("RAG_INDEX_PKL", "").strip() or None
    _default_rag_index_dir = os.environ.get("RAG_INDEX_DIR", "").strip() or None
    if not mp:
        raise RuntimeError("请设置环境变量 VLM_MODEL_PATH 指向模型目录（含 config.json）")
    disable_torch_init()
    use_flash = os.environ.get("VLM_DISABLE_FLASH", "").lower() not in ("1", "true", "yes")
    load_4bit = os.environ.get("VLM_LOAD_4BIT", "").lower() in ("1", "true", "yes")
    load_8bit = os.environ.get("VLM_LOAD_8BIT", "").lower() in ("1", "true", "yes")
    model_name = get_model_name_from_path(mp)
    processor, model = load_pretrained_model(
        model_path=mp,
        model_base=mb,
        model_name=model_name,
        device_map=device,
        device=device,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        use_flash_attn=use_flash,
    )
    if _default_rag_index_dir:
        _medical_qa_retriever = MedicalQARetriever(
            index_dir=_default_rag_index_dir,
            embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", "").strip() or None,
            reranker_model=os.environ.get("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            device=os.environ.get("RAG_EMBED_DEVICE", "").strip() or None,
            reranker_device=os.environ.get("RAG_RERANKER_DEVICE", "").strip() or None,
            enable_reranker=os.environ.get("RAG_DISABLE_RERANK", "").lower() not in ("1", "true", "yes"),
            question_weight=float(os.environ.get("RAG_QUESTION_WEIGHT", "0.7")),
            doc_weight=float(os.environ.get("RAG_DOC_WEIGHT", "0.3")),
            bm25_weight=float(os.environ.get("RAG_BM25_WEIGHT", "0.25")),
            low_quality_penalty=float(os.environ.get("RAG_LOW_QUALITY_PENALTY", "0.85")),
        )


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "rag_index_pkl": _default_rag_pkl or "",
        "rag_index_dir": _default_rag_index_dir or "",
    }


@app.post("/v1/infer", response_model=InferResponse)
def infer(body: InferBody) -> InferResponse:
    if body.image_path and not os.path.isfile(body.image_path):
        raise HTTPException(400, f"图像不存在或不可读: {body.image_path}")

    rag_context: Optional[str] = None
    rag_prompt: Optional[str] = None
    scores: Optional[List[List[Any]]] = None
    if body.use_rag and _medical_qa_retriever is not None:
        docs = _medical_qa_retriever.search(
            body.query,
            top_k=body.top_k,
            rerank_top_n=body.rerank_top_n,
            vector_top_k=body.vector_top_k,
            bm25_top_k=body.bm25_top_k,
            score_threshold=body.score_threshold,
        )
        rag_context = build_retrieved_context(docs, max_context_chars=body.max_context_chars)
        rag_prompt = build_prompt(body.query, docs, max_context_chars=body.max_context_chars)
        scores = [
            [
                int(d.get("final_rank", i + 1)),
                str(d.get("doc_id")),
                float(d.get("final_score", d.get("rerank_score", d.get("faiss_score", 0.0)))),
            ]
            for i, d in enumerate(docs)
        ]
    elif body.use_rag and _default_rag_pkl and os.path.isfile(_default_rag_pkl):
        r = retrieve(
            _default_rag_pkl,
            body.query,
            top_k=body.top_k,
            max_context_chars=body.max_context_chars,
        )
        rag_context = r.get("context") or None
        scores = r.get("retrieved_scores")  # List[Tuple[int,float]] -> JSON as nested lists

    conv = _build_conversation(body.query, body.image_path, rag_context, rag_prompt=rag_prompt)
    try:
        text = _generate_one(conv, body)
    except Exception as e:
        raise HTTPException(500, f"生成失败: {e!s}") from e

    # JSON-serialize scores: legacy pkl returns (idx, score); FAISS QA returns [rank, doc_id, score].
    ser_scores = None
    if scores is not None:
        ser_scores = []
        for item in scores:
            if len(item) == 2:
                i, s = item
                ser_scores.append([int(i), float(s)])
            else:
                rank, doc_id, s = item[:3]
                ser_scores.append([int(rank), str(doc_id), float(s)])

    return InferResponse(
        text=text,
        rag_context=rag_context,
        retrieved_scores=ser_scores,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8088")))
    parser.add_argument(
        "--model-path",
        default=os.environ.get("VLM_MODEL_PATH", ""),
        help="或设置环境变量 VLM_MODEL_PATH",
    )
    parser.add_argument("--model-base", default=os.environ.get("VLM_MODEL_BASE", ""))
    parser.add_argument("--rag-index-pkl", default=os.environ.get("RAG_INDEX_PKL", ""))
    parser.add_argument("--rag-index-dir", default=os.environ.get("RAG_INDEX_DIR", ""))
    parser.add_argument("--device", default=os.environ.get("VLM_DEVICE", "cuda"))
    args = parser.parse_args()
    if not args.model_path:
        parser.error("需要 --model-path 或环境变量 VLM_MODEL_PATH")
    os.environ["VLM_MODEL_PATH"] = args.model_path
    if args.model_base:
        os.environ["VLM_MODEL_BASE"] = args.model_base
    if args.rag_index_pkl:
        os.environ["RAG_INDEX_PKL"] = args.rag_index_pkl
    if args.rag_index_dir:
        os.environ["RAG_INDEX_DIR"] = args.rag_index_dir
    os.environ["VLM_DEVICE"] = args.device

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
