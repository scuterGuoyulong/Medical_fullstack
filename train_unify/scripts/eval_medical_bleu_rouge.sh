#!/bin/bash
# 基座 vs 微调：在同一份医学 JSON 上计算 BLEU-1~4 与 ROUGE-L。
# 需: pip install jieba nltk rouge-score
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

ANDES_VL_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ANDES_VL_ROOT}/models/models/Qwen/Qwen3___5-4B}"
DATA_JSON="${DATA_JSON:-${ANDES_VL_ROOT}/DataSets/medical/mixed_sft/train_qa_report_qwen_vl.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${ANDES_VL_ROOT}/DataSets/mimic-cxr-jpeg-sample200}"
FINETUNED_DIR="${FINETUNED_DIR:-${REPO_ROOT}/output/qwen35_4b_medical_sft}"
OUT_JSON="${OUT_JSON:-${REPO_ROOT}/output/medical_eval_bleu_rouge.json}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python scripts/eval_medical_bleu_rouge.py \
  --base_model "${MODEL_DIR}" \
  --finetuned_model "${FINETUNED_DIR}" \
  --data_path "${DATA_JSON}" \
  --image_folder "${IMAGE_FOLDER}" \
  --bf16 \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --output_json "${OUT_JSON}"
