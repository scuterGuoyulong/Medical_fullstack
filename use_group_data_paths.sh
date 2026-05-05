#!/usr/bin/env bash
# 在「本仓库 + 你们组内机器」上的数据与模型路径。用法：
#   source /path/to/medical_fullstack/use_group_data_paths.sh
# 然后再跑语料脚本或设置 TRAIN_FILE_DIR 做训练。

export ANDES_VL_ROOT="/home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/SuperResolution_train_prx/andes_vl"
export MEDICAL_DS="${ANDES_VL_ROOT}/DataSets/medical"
export MIMIC_CXR_ROOT="${ANDES_VL_ROOT}/DataSets/mimic-cxr-dataset"
export MODEL_NAME_OR_PATH="${ANDES_VL_ROOT}/models/models/Qwen/Qwen3___5-4B"

export MEDICAL_FULLSTACK="${ANDES_VL_ROOT}/medical_fullstack"
export MEDICAL_DATA_ROOT="${MEDICAL_FULLSTACK}/data"

# ---------- 医疗问答（已是 Alpaca：instruction / input / output）----------
# 直接给 Medical_Qwen pretraining.py SFT 用（目录内需 train_*.json）：
export MEDICAL_QA_SFT_DIR="${MEDICAL_DS}/finetune"
# 例如：train_zh_0.json、valid_zh_0.json、train_en_1.json 等

# 若要对中文问答做 SimHash，先转成 JSONL 再跑 simhash_dedup.py（全量 195 万条极慢，建议先 --max_rows 抽样）
export MEDICAL_QA_JSONL_SAMPLE="${MEDICAL_DATA_ROOT}/raw/medical_qa_zh_sample.jsonl"

# ---------- 医疗报告（MIMIC-CXR 英文 findings/impression）----------
export MIMIC_REPORT_EN_JSONL="${MEDICAL_DATA_ROOT}/raw/mimic_report_en.jsonl"
export MIMIC_REPORT_ZH_JSONL="${MEDICAL_DATA_ROOT}/clean/mimic_report_zh.jsonl"

# ---------- 预训练百科（CLM，text 字段）----------
export MEDICAL_PRETRAIN_DIR="${MEDICAL_DS}/pretrain"

# ---------- DPO 现成偏好对（已有 response_chosen / response_rejected）----------
export MEDICAL_REWARD_DIR="${MEDICAL_DS}/reward"
