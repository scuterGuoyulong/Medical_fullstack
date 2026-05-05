#!/usr/bin/env bash
# 医疗问答 SFT（Alpaca/ShareGPT 经 pretraining 自动识别；此处推荐 ShareGPT -> 转 text 前为 conversations）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MQ="${ROOT}/Medical_Qwen"
export TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-${ROOT}/medical_fullstack/data/sft_qa}"
export OUTPUT_DIR="${OUTPUT_DIR:-${MQ}/outputs-sft-medical-qa}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
export TRAIN_MODE="${TRAIN_MODE:-sft}"

if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  echo "请设置 MODEL_NAME_OR_PATH（可为 CPT 后 checkpoint）" >&2
  exit 1
fi

cd "${MQ}"
bash run_pt_incremental_deepspeed.sh
