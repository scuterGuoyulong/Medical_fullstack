#!/bin/bash
# Qwen3.5 LoRA SFT for weakly supervised chest X-ray grounding + long CoT.
#
# Expected data:
#   bash scripts/build_xray_grounding_sft.sh
# The generated JSON uses image=[full_xray, region_crop] and assistant.reasoning.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

ANDES_VL_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ANDES_VL_ROOT}/models/models/Qwen/Qwen3___5-4B}"
DATA_JSON="${DATA_JSON:-${REPO_ROOT}/output/xray_grounding_cot_sft.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${ANDES_VL_ROOT}/DataSets/mimic-cxr-jpeg-sample200}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/qwen35_xray_grounding_cot_lora}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-1}"
NUM_DEVICES="${NUM_DEVICES:-1}"
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
if [ "$GRAD_ACCUM_STEPS" -lt 1 ]; then
  GRAD_ACCUM_STEPS=1
fi

SAVE_STEPS="${SAVE_STEPS:-10}"
RUN_SAMPLE_EVAL_DURING_TRAIN="${RUN_SAMPLE_EVAL_DURING_TRAIN:-False}"
SAMPLE_EVAL_NUM_SAMPLES="${SAMPLE_EVAL_NUM_SAMPLES:-8}"
SAMPLE_EVAL_MAX_NEW_TOKENS="${SAMPLE_EVAL_MAX_NEW_TOKENS:-256}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

EVAL_ARGS=(--eval_strategy no)
case "${RUN_SAMPLE_EVAL_DURING_TRAIN}" in
  True|true|1|yes|YES)
    # Qwen3.5 linear attention can fail in model.generate() under DeepSpeed ZeRO/offload.
    # Keep this opt-in; for stable sample checks, generate after saving a checkpoint.
    EVAL_ARGS=(
      --eval_path "${SAMPLE_EVAL_JSON:-${DATA_JSON}}"
      --eval_image_folder "${IMAGE_FOLDER}"
      --eval_max_samples "${SAMPLE_EVAL_NUM_SAMPLES}"
      --eval_strategy steps
      --eval_steps "${SAVE_STEPS}"
      --per_device_eval_batch_size 1
      --prediction_loss_only False
      --generation_max_new_tokens "${SAMPLE_EVAL_MAX_NEW_TOKENS}"
      --sample_eval_save_predictions True
    )
    ;;
esac

if command -v deepspeed >/dev/null 2>&1; then
  DEEPSPEED_LAUNCHER=(deepspeed)
else
  DEEPSPEED_LAUNCHER=(python -m deepspeed.launcher.runner)
fi

"${DEEPSPEED_LAUNCHER[@]}" --num_gpus="${NUM_DEVICES}" src/train/train_sft.py \
    --use_liger_kernel False \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank "${LORA_RANK:-32}" \
    --lora_alpha "${LORA_ALPHA:-64}" \
    --lora_dropout "${LORA_DROPOUT:-0.05}" \
    --num_lora_modules -1 \
    --deepspeed "${DEEPSPEED_CONFIG:-scripts/zero3_offload.json}" \
    --model_id "${MODEL_DIR}" \
    --data_path "${DATA_JSON}" \
    --image_folder "${IMAGE_FOLDER}" \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --bits 16 \
    --disable_flash_attn2 True \
    --enable_reasoning True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
    --per_device_train_batch_size "${BATCH_PER_DEVICE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --image_min_pixels $((384 * 32 * 32)) \
    --image_max_pixels $((1280 * 32 * 32)) \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --merger_lr "${MERGER_LR:-1e-5}" \
    --vision_lr "${VISION_LR:-2e-6}" \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    "${EVAL_ARGS[@]}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 5 \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}"
