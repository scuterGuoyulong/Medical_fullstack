#!/usr/bin/env bash
# Validate generated chest X-ray grounding SFT JSON and create a manual review sample.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_JSON="${DATA_JSON:-${ROOT}/output/xray_grounding_cot_sft.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${ROOT}/../DataSets/mimic-cxr-jpeg-sample200}"
MANUAL_REVIEW_JSON="${MANUAL_REVIEW_JSON:-${ROOT}/output/xray_grounding_manual_review.json}"
MANUAL_REVIEW_SIZE="${MANUAL_REVIEW_SIZE:-200}"

CMD=(python scripts/eval_xray_grounding_sft.py
  --data_json "$DATA_JSON"
  --image_folder "$IMAGE_FOLDER"
  --manual_review_json "$MANUAL_REVIEW_JSON"
  --manual_review_size "$MANUAL_REVIEW_SIZE"
)

if [[ -n "${PREDICTIONS_JSON:-}" ]]; then
  CMD+=(--predictions_json "$PREDICTIONS_JSON")
fi

"${CMD[@]}"
