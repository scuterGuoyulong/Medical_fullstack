#!/usr/bin/env bash
# 编排：报告英译中、问答去重+润色、转训练格式、划分输出目录（需自行准备原始 JSONL / 知识库）
set -euo pipefail
CORPUS="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${CORPUS}/../.." && pwd)"
DATA="${MEDICAL_DATA_ROOT:-${ROOT}/medical_fullstack/data}"
MODEL="${MODEL_NAME_OR_PATH:?请设置 MODEL_NAME_OR_PATH（Qwen3.5-4B）}"

mkdir -p "${DATA}/raw" "${DATA}/clean" "${DATA}/cpt_unified" "${DATA}/sft_qa" "${DATA}/sft_report" "${DATA}/dpo"

REPORT_EN_JSONL="${REPORT_EN_JSONL:-${DATA}/raw/report_en.jsonl}"
QA_RAW_JSONL="${QA_RAW_JSONL:-${DATA}/raw/qa_raw.jsonl}"

# 1) 报告：英文 findings/impression -> 中文结构化（耗时长，可注释掉已跑过的步骤）
if [[ -f "${REPORT_EN_JSONL}" ]]; then
  python "${CORPUS}/translate_reports.py" \
    --input "${REPORT_EN_JSONL}" \
    --output "${DATA}/clean/report_zh.jsonl" \
    --model_name_or_path "${MODEL}"
fi

# 2) 问答：SimHash 去重
if [[ -f "${QA_RAW_JSONL}" ]]; then
  python "${CORPUS}/simhash_dedup.py" \
    --input "${QA_RAW_JSONL}" \
    --output "${DATA}/clean/qa_dedup.jsonl" \
    --combine_keys question,answer \
    --max_hamming 3 \
    --keep longest
  python "${CORPUS}/rewrite_quality.py" \
    --input "${DATA}/clean/qa_dedup.jsonl" \
    --output "${DATA}/clean/qa_polished.jsonl" \
    --mode llm \
    --model_name_or_path "${MODEL}"
fi

# 3) 训练格式
if [[ -f "${DATA}/clean/qa_polished.jsonl" ]]; then
  python "${CORPUS}/convert_to_training_formats.py" --mode qa_sharegpt \
    --input "${DATA}/clean/qa_polished.jsonl" \
    --output "${DATA}/sft_qa/train_medical_qa_sharegpt.jsonl"
fi
if [[ -f "${DATA}/clean/report_zh.jsonl" ]]; then
  python "${CORPUS}/convert_to_training_formats.py" --mode report_alpaca \
    --input "${DATA}/clean/report_zh.jsonl" \
    --output "${DATA}/sft_report/train_medical_report_alpaca.jsonl"
fi
if [[ -f "${DATA}/clean/qa_polished.jsonl" ]] && [[ -f "${DATA}/clean/report_zh.jsonl" ]]; then
  python "${CORPUS}/convert_to_training_formats.py" --mode cpt_sharegpt \
    --input "${DATA}/clean/qa_polished.jsonl" \
    --input "${DATA}/clean/report_zh.jsonl" \
    --output "${DATA}/cpt_unified/train_medical_unified_cpt.jsonl"
fi

# 4) DPO 示例对（弱负例仅作流水线验证）
if [[ -f "${DATA}/clean/qa_polished.jsonl" ]]; then
  python "${CORPUS}/build_dpo_pairs.py" \
    --input "${DATA}/clean/qa_polished.jsonl" \
    --output "${DATA}/dpo/train_dpo_qa.jsonl" \
    --scenario qa \
    --mode synthetic_trunc
fi

echo "完成。请将各 train_*.jsonl 放入 Medical_Qwen 期望的目录或设置 TRAIN_FILE_DIR。"
