#!/usr/bin/env bash
# Qwen-VL + RAG FastAPI。请先完成 Qwen-VL-Series-Finetune 依赖与模型路径。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANDES_VL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${ANDES_VL_ROOT}/Qwen-VL-Series-Finetune/src:${PYTHONPATH:-}"

# 必填：HF 布局模型目录（微调后 checkpoint 或基座）
: "${VLM_MODEL_PATH:?请 export VLM_MODEL_PATH=/path/to/Qwen3.5-4B 或你的 output/checkpoint-xxx}"

# 可选：LoRA 合并失败时需填基座；标准全量模型留空
export VLM_MODEL_BASE="${VLM_MODEL_BASE:-}"

# 可选：新的 QA FAISS 目录；优先于旧 pkl，适配 build_medical_qa_faiss.py
export RAG_INDEX_DIR="${RAG_INDEX_DIR:-}"

# 可选：index_kb.py 生成的旧 pkl；不设置则仅多模态生成、不做 RAG
export RAG_INDEX_PKL="${RAG_INDEX_PKL:-}"

export VLM_DEVICE="${VLM_DEVICE:-cuda}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8088}"

# 无 flash-attn 时可: export VLM_DISABLE_FLASH=1
# 显存紧: export VLM_LOAD_4BIT=1

exec python3 "${SCRIPT_DIR}/vlm_rag_fastapi.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --model-path "${VLM_MODEL_PATH}" \
  ${VLM_MODEL_BASE:+--model-base "${VLM_MODEL_BASE}"} \
  ${RAG_INDEX_DIR:+--rag-index-dir "${RAG_INDEX_DIR}"} \
  ${RAG_INDEX_PKL:+--rag-index-pkl "${RAG_INDEX_PKL}"} \
  --device "${VLM_DEVICE}"
