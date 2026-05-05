#!/bin/bash
# SFT for local Qwen3.5-4B (HF layout, model_type=qwen3_5) with image + text.
#
# 环境：按仓库 requirements.txt 安装（含 transformers==5.3.0）；旧版 Transformers 无法识别 qwen3_5。
# 其他：CUDA、DeepSpeed；Qwen3.5 建议 --disable_flash_attn2 True（SDPA，见 README Training Notes）。
# 数据：conversations（human/gpt）+ 顶层 image 相对路径；<image> 会转为官方 vision 占位。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

ANDES_VL_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/output/qwen35_4b_medical_sft}"
DATA_JSON="${DATA_JSON:-${ANDES_VL_ROOT}/DataSets/medical/mixed_sft/train_qa_report_qwen_vl.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${ANDES_VL_ROOT}/DataSets/mimic-cxr-jpeg-sample200}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/qwen35_4b_medical_sft}"

# 医学 BLEU/ROUGE 评测挂在 Trainer 的 on_save 上：只有「保存 checkpoint」时才会跑。
# 因此 save_steps 必须与 medical_eval_bleu_steps 一致（或为其整数倍且保存点落在整除处），
# 且间隔要 ≤ 单轮总步数，否则会像「到 49/130 步仍从未保存」一样一直不评测。
SAVE_AND_EVAL_EVERY="${SAVE_AND_EVAL_EVERY:-50}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-16}"
NUM_DEVICES="${NUM_DEVICES:-1}"
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
if [ "$GRAD_ACCUM_STEPS" -lt 1 ]; then
  GRAD_ACCUM_STEPS=1
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# deepspeed 需已安装；若未生成 ~/.local/bin 或 conda 的 deepspeed 可执行文件，则用模块方式启动
if command -v deepspeed >/dev/null 2>&1; then
  DEEPSPEED_LAUNCHER=(deepspeed)
else
  DEEPSPEED_LAUNCHER=(python -m deepspeed.launcher.runner)
fi

"${DEEPSPEED_LAUNCHER[@]}" --num_gpus="${NUM_DEVICES}" src/train/train_sft.py \
    --use_liger_kernel False \
    --deepspeed scripts/zero3_offload.json \
    --model_id "${MODEL_DIR}" \
    --data_path "${DATA_JSON}" \
    --image_folder "${IMAGE_FOLDER}" \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 10 \
    --per_device_train_batch_size "${BATCH_PER_DEVICE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --image_min_pixels $((512 * 32 * 32)) \
    --image_max_pixels $((1280 * 32 * 32)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps "${SAVE_AND_EVAL_EVERY}" \
    --save_total_limit 3 \
    --medical_eval_bleu_steps "${SAVE_AND_EVAL_EVERY}" \
    --medical_eval_validation_root "${OUTPUT_DIR}/validation" \
    --dataloader_num_workers 4
