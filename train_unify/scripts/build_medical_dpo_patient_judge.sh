#!/usr/bin/env bash
# 在 Qwen-VL-Series-Finetune 目录下执行。
# 混合策略（默认）：chosen=参考答案轻改（文本）+ rejected=SFT-VL（医生腔或高温）；
# 法官为文本 Qwen3.5（默认：上级 andes_vl 下 models/models/Qwen/Qwen3___5-4B；单卡请 export JUDGE_DEVICE=cpu）。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

SFT_JSON="${SFT_JSON:-${ROOT}/../DataSets/medical/mixed_sft/train_qa_report_qwen_vl.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${ROOT}/../DataSets/medical/mixed_sft}"
ANDES_VL_ROOT="$(cd "${ROOT}/.." && pwd)"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${MODEL_PATH:-${ANDES_VL_ROOT}/models/models/Qwen/Qwen3___5-4B}}"
VL_MODEL_PATH="${VL_MODEL_PATH:-${ROOT}/output/qwen35_4b_medical_sft}"
VL_BASE_MODEL="${VL_BASE_MODEL:-}"
OUT_JSON="${OUT_JSON:-${ROOT}/output/medical_dpo_patient_judged.json}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BUILD_MODE="${BUILD_MODE:-hybrid}"
CHOSEN_MODE="${CHOSEN_MODE:-reference_light}"
REJECTED_MODE="${REJECTED_MODE:-sft_doctor}"
JUDGE_DEVICE="${JUDGE_DEVICE:-auto}"

CMD=(python scripts/build_medical_dpo_patient_judge.py
  --build_mode "$BUILD_MODE"
  --sft_json "$SFT_JSON"
  --output_json "$OUT_JSON"
  --judge_model_path "$JUDGE_MODEL_PATH"
  --image_folder "$IMAGE_FOLDER"
  --chosen_mode "$CHOSEN_MODE"
  --rejected_mode "$REJECTED_MODE"
  --judge_device "$JUDGE_DEVICE"
  --max_samples "$MAX_SAMPLES"
  --ensure_placeholder
  --bf16
)

if [[ "$BUILD_MODE" == "hybrid" ]]; then
  CMD+=(--vl_model_path "$VL_MODEL_PATH")
  if [[ -n "$VL_BASE_MODEL" ]]; then
    CMD+=(--vl_base_model "$VL_BASE_MODEL")
  fi
fi

"${CMD[@]}"
