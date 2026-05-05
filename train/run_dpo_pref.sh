#!/usr/bin/env bash
# DPO 偏好对齐：数据目录含 system/history/question/response_chosen/response_rejected 的 JSONL
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MQ="${ROOT}/Medical_Qwen"
cd "${MQ}"

MODEL="${MODEL_NAME_OR_PATH:?请设置 MODEL_NAME_OR_PATH（SFT 后 checkpoint）}"
OUT="${OUTPUT_DIR:-${MQ}/outputs-dpo-medical-pref}"
DATA="${DPO_TRAIN_DIR:-${ROOT}/medical_fullstack/data/dpo}"

torchrun --nproc_per_node "${NPROC:-2}" --master_port "${MASTER_PORT:-29519}" dpo_training.py \
  --model_name_or_path "${MODEL}" \
  --template_name qwen \
  --train_file_dir "${DATA}" \
  --validation_split_percentage 5 \
  --per_device_train_batch_size "${DPO_BS:-1}" \
  --gradient_accumulation_steps "${DPO_GAS:-16}" \
  --per_device_eval_batch_size 1 \
  --do_train \
  --do_eval \
  --use_peft "${USE_PEFT:-True}" \
  --max_source_length "${DPO_MAX_SRC:-1024}" \
  --max_target_length "${DPO_MAX_TGT:-512}" \
  --learning_rate "${DPO_LR:-5e-6}" \
  --max_steps "${DPO_MAX_STEPS:-500}" \
  --eval_steps "${DPO_EVAL_STEPS:-50}" \
  --save_steps "${DPO_SAVE_STEPS:-100}" \
  --output_dir "${OUT}" \
  --target_modules all \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 True \
  --fp16 False \
  --device_map auto \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing True \
  --cache_dir "${CACHE_DIR:-./cache}"
