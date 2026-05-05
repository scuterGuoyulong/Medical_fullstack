#!/usr/bin/env bash
# GRPO 偏好/格式强化（沿用 Medical_Qwen/grpo_training.py；数据格式见该脚本说明）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MQ="${ROOT}/Medical_Qwen"
cd "${MQ}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
torchrun --nproc_per_node "${NPROC:-2}" --master_port "${MASTER_PORT:-29520}" grpo_training.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH:?请设置 MODEL_NAME_OR_PATH}" \
  --train_file_dir "${GRPO_TRAIN_DIR:-${ROOT}/medical_fullstack/data/grpo}" \
  --train_samples "${GRPO_TRAIN_SAMPLES:--1}" \
  --output_dir "${OUTPUT_DIR:-${MQ}/outputs-grpo-medical-pref}" \
  --dtype bfloat16 \
  --bf16 True \
  --report_to tensorboard \
  --remove_unused_columns False \
  --use_peft "${USE_PEFT:-True}" \
  --qlora "${QGRPO_QLORA:-False}" \
  --load_in_4bit "${GRPO_4BIT:-False}" \
  --per_device_train_batch_size "${GRPO_BS:-2}" \
  --gradient_accumulation_steps "${GRPO_GAS:-4}" \
  --max_prompt_length "${GRPO_MAX_PROMPT:-2048}" \
  --max_completion_length "${GRPO_MAX_COMP:-512}" \
  --num_train_epochs "${GRPO_EPOCHS:-1}" \
  --learning_rate "${GRPO_LR:-5e-7}" \
  --save_steps "${GRPO_SAVE:-100}" \
  --logging_steps 10 \
  --use_vllm False
