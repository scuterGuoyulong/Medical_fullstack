#!/usr/bin/env bash
# 双场景统一增量预训练（CLM）：数据为 ShareGPT conversations，train_mode=clm
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MQ="${ROOT}/Medical_Qwen"
export TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-${ROOT}/medical_fullstack/data/cpt_unified}"
export OUTPUT_DIR="${OUTPUT_DIR:-${MQ}/outputs-pt-medical-unified-cpt}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
export TRAIN_MODE="${TRAIN_MODE:-clm}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"

if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  echo "请设置 MODEL_NAME_OR_PATH 指向 Qwen3.5-4B（或 CPT 起点）" >&2
  exit 1
fi

# 目录内需有 train_*.json / train_*.jsonl（见 corpus/convert_to_training_formats.py --mode cpt_sharegpt）
if [[ ! -d "${TRAIN_FILE_DIR}" ]]; then
  echo "TRAIN_FILE_DIR 不存在: ${TRAIN_FILE_DIR}" >&2
  exit 1
fi

cd "${MQ}"
bash run_pt_incremental_deepspeed.sh
