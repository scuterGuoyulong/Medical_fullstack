#!/bin/bash
# 评测指定 checkpoint 在训练集上的 BLEU/ROUGE/BERTScore。
# 需: pip install jieba nltk rouge-score bert-score
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

ANDES_VL_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/output/qwen35_xray_grounding_cot_lora/checkpoint-120}"
DATA_JSON="${DATA_JSON:-/home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/SuperResolution_train_prx/andes_vl/DataSets/medical/mixed_sft/train_qa_report_qwen_vl.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${ANDES_VL_ROOT}/DataSets/mimic-cxr-jpeg-sample200}"

OUT_JSON="${OUT_JSON:-${REPO_ROOT}/output/eval_ckpt120_train_metrics.json}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python scripts/eval_checkpoint_metrics.py \
  --checkpoint "${CHECKPOINT_DIR}" \
  --data_path "${DATA_JSON}" \
  --image_folder "${IMAGE_FOLDER}" \
  --bf16 \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_samples "${MAX_SAMPLES}" \
  --output_json "${OUT_JSON}"

