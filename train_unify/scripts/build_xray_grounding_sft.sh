#!/usr/bin/env bash
# Build weakly supervised chest X-ray grounding SFT data.
# Run from the Qwen-VL-Series-Finetune repository root.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

INPUT_JSON="${INPUT_JSON:-${ROOT}/../DataSets/medical/mixed_sft/train_qa_report_qwen_vl.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${ROOT}/../DataSets/mimic-cxr-jpeg-sample200}"
OUT_JSON="${OUT_JSON:-${ROOT}/output/xray_grounding_cot_sft.json}"
CROP_DIR="${CROP_DIR:-${IMAGE_FOLDER}/xray_grounding_crops}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_SENTENCES_PER_REPORT="${MAX_SENTENCES_PER_REPORT:-6}"
NEGATIVE_PER_POSITIVE="${NEGATIVE_PER_POSITIVE:-1}"
SEED="${SEED:-42}"

python scripts/build_xray_grounding_sft.py \
  --input_json "$INPUT_JSON" \
  --image_folder "$IMAGE_FOLDER" \
  --output_json "$OUT_JSON" \
  --crop_dir "$CROP_DIR" \
  --max_samples "$MAX_SAMPLES" \
  --max_sentences_per_report "$MAX_SENTENCES_PER_REPORT" \
  --negative_per_positive "$NEGATIVE_PER_POSITIVE" \
  --seed "$SEED" \
  --radiology_view
