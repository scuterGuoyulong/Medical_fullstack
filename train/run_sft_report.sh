#!/usr/bin/env bash
# 医疗报告生成 SFT：数据为 Alpaca 列 instruction/input/output
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MQ="${ROOT}/Medical_Qwen"
export TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-${ROOT}/medical_fullstack/data/sft_report}"
export OUTPUT_DIR="${OUTPUT_DIR:-${MQ}/outputs-sft-medical-report}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
export TRAIN_MODE="${TRAIN_MODE:-sft}"

if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  echo "请设置 MODEL_NAME_OR_PATH（可为 CPT 后 checkpoint）" >&2
  exit 1
fi

cd "${MQ}"
bash run_pt_incremental_deepspeed.sh
