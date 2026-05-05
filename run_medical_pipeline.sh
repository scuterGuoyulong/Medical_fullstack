#!/usr/bin/env bash
# 医疗全栈一键/分步入口：路径、冒烟数据、CPT / SFT / DPO / GRPO / RAG 索引
# 用法: ./run_medical_pipeline.sh help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDICAL_FULLSTACK="${MEDICAL_FULLSTACK:-${SCRIPT_DIR}}"
ANDES_VL_ROOT="${ANDES_VL_ROOT:-$(cd "${MEDICAL_FULLSTACK}/.." && pwd)}"
MQ="${ANDES_VL_ROOT}/Medical_Qwen"
CORPUS="${MEDICAL_FULLSTACK}/corpus"
TRAIN_SH="${MEDICAL_FULLSTACK}/train"

DEFAULT_MODEL="${ANDES_VL_ROOT}/models/models/Qwen/Qwen3___5-4B"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${DEFAULT_MODEL}}"
export MEDICAL_DS="${MEDICAL_DS:-${ANDES_VL_ROOT}/DataSets/medical}"

export SMOKE_SFT_N="${SMOKE_SFT_N:-512}"
export SMOKE_DPO_N="${SMOKE_DPO_N:-64}"
export SMOKE_DPO_STEPS="${SMOKE_DPO_STEPS:-50}"

usage() {
  cat <<'EOF'
医疗流水线（medical_fullstack）

  环境变量（常用）:
    MODEL_NAME_OR_PATH   基座或上阶段 checkpoint（目录内需有 config.json）
    ANDES_VL_ROOT        默认: medical_fullstack 的上一级
    MEDICAL_DS           默认: ${ANDES_VL_ROOT}/DataSets/medical
    SMOKE_SFT_N / SMOKE_DPO_N / SMOKE_DPO_STEPS  冒烟规模
    NPROC、CUDA_VISIBLE_DEVICES、TRAIN_FILE_DIR、OUTPUT_DIR、DPO_TRAIN_DIR 等同 train/*.sh

  命令:
    help              本说明
    print-env         打印建议 export
    prepare-smoke-sft 从 MEDICAL_DS/finetune 生成 data/sft_smoke
    prepare-smoke-dpo 从 DPO_SOURCE_DIR 下 jsonl 生成 data/dpo_smoke
    rag-index         需 KB_DIR、OUT_PKL
    cpt               增量 CLM（TRAIN_FILE_DIR 默认 data/cpt_unified）
    sft               SFT（TRAIN_FILE_DIR 默认 MEDICAL_DS/finetune）
    sft-smoke         SFT，数据用 data/sft_smoke
    dpo               DPO（设 DPO_TRAIN_DIR；模型建议 SFT checkpoint）
    dpo-smoke         小 DPO 数据 + DPO_MAX_STEPS=SMOKE_DPO_STEPS
    grpo-smoke        GRPO 冒烟
    all-smoke         prepare-smoke-sft -> sft-smoke -> prepare-smoke-dpo -> dpo-smoke -> grpo-smoke

  示例:
    ./run_medical_pipeline.sh print-env
    ./run_medical_pipeline.sh prepare-smoke-sft && ./run_medical_pipeline.sh sft-smoke
    KB_DIR=/path/to/kb OUT_PKL=/tmp/kb.pkl ./run_medical_pipeline.sh rag-index
    export TRAIN_FILE_DIR=/data/cpt_train_jsonl && ./run_medical_pipeline.sh cpt
EOF
}

die() { echo "[ERROR] $*" >&2; exit 1; }
log() { echo "[pipeline] $*"; }

require_dir() { [[ -d "$1" ]] || die "目录不存在: $1"; }

check_model() {
  [[ -d "${MODEL_NAME_OR_PATH}" ]] || die "MODEL_NAME_OR_PATH 不是目录: ${MODEL_NAME_OR_PATH}"
  [[ -f "${MODEL_NAME_OR_PATH}/config.json" ]] || die "缺少 config.json: ${MODEL_NAME_OR_PATH}"
}

cmd_print_env() {
  cat <<EOF
export ANDES_VL_ROOT="${ANDES_VL_ROOT}"
export MEDICAL_FULLSTACK="${MEDICAL_FULLSTACK}"
export MEDICAL_DS="${MEDICAL_DS}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"
export MQ="${MQ}"
export CORPUS="${CORPUS}"
EOF
  if [[ -d "${MQ}" ]]; then
    log "Medical_Qwen: OK"
  else
    log "警告: 未找到 Medical_Qwen: ${MQ}"
  fi
  if [[ -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
    log "模型路径: OK"
  else
    log "警告: 模型路径可能无效"
  fi
}

cmd_prepare_smoke_sft() {
  local src="${SFT_SMOKE_SOURCE:-${MEDICAL_DS}/finetune}"
  local dst="${MEDICAL_FULLSTACK}/data/sft_smoke"
  require_dir "${src}"
  mkdir -p "${dst}"
  src="${src}" dst="${dst}" n="${SMOKE_SFT_N}" python3 - <<'PY'
import json, glob, os, sys
src = os.environ["src"]
dst = os.environ["dst"]
n = int(os.environ["n"])
paths = sorted(
    glob.glob(os.path.join(src, "train_*.json"))
    + glob.glob(os.path.join(src, "train_*.jsonl"))
)
if not paths:
    print("no train_*.json/jsonl in", src, file=sys.stderr)
    sys.exit(1)
for p in paths:
    base = os.path.basename(p)
    out = os.path.join(dst, base)
    if p.endswith(".jsonl"):
        lines = []
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                lines.append(line)
        with open(out, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print("wrote", out, "lines", len(lines))
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("expected JSON array in", p, file=sys.stderr)
            sys.exit(1)
        data = data[:n]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("wrote", out, "items", len(data))
    break
PY
  log "SFT 冒烟数据: ${dst}"
}

# 将源目录中每个 *.jsonl 截断为前 N 行写入 data/dpo_smoke
cmd_prepare_smoke_dpo() {
  local src="${DPO_SOURCE_DIR:-${MEDICAL_FULLSTACK}/data/dpo}"
  local dst="${MEDICAL_FULLSTACK}/data/dpo_smoke"
  require_dir "${src}"
  mkdir -p "${dst}"
  local n="${SMOKE_DPO_N}"
  shopt -s nullglob
  local f
  local any=0
  for f in "${src}"/*.jsonl; do
    any=1
    head -n "${n}" "${f}" > "${dst}/$(basename "${f}")"
    log "wrote ${dst}/$(basename "${f}")"
  done
  shopt -u nullglob
  [[ "${any}" -eq 1 ]] || die "在 ${src} 下未找到 *.jsonl，请先构造 DPO 数据或设置 DPO_SOURCE_DIR"
}

cmd_rag_index() {
  local kb="${KB_DIR:-}"
  local out="${OUT_PKL:-}"
  [[ -n "${kb}" ]] || die "请设置 KB_DIR=含 .txt/.md 的知识库目录"
  [[ -n "${out}" ]] || die "请设置 OUT_PKL=输出的 index.pkl 路径"
  require_dir "${kb}"
  python3 "${MEDICAL_FULLSTACK}/rag/index_kb.py" --kb_dir "${kb}" --out "${out}" \
    --backend "${RAG_BACKEND:-tfidf}"
  log "索引已写入: ${out}  检索: python3 rag/rag_infer.py --index_pkl ${out} --query \"...\""
}

cmd_cpt() {
  check_model
  require_dir "${MQ}"
  export TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-${MEDICAL_FULLSTACK}/data/cpt_unified}"
  require_dir "${TRAIN_FILE_DIR}"
  export TRAIN_MODE="${TRAIN_MODE:-clm}"
  export OUTPUT_DIR="${OUTPUT_DIR:-${MQ}/outputs-cpt-medical-pipeline}"
  bash "${TRAIN_SH}/run_cpt_unified.sh"
}

cmd_sft() {
  check_model
  require_dir "${MQ}"
  export TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-${MEDICAL_DS}/finetune}"
  require_dir "${TRAIN_FILE_DIR}"
  export TRAIN_MODE="${TRAIN_MODE:-sft}"
  export OUTPUT_DIR="${OUTPUT_DIR:-${MQ}/outputs-sft-medical-pipeline}"
  bash "${TRAIN_SH}/run_sft_qa.sh"
}

cmd_sft_smoke() {
  local smoke="${MEDICAL_FULLSTACK}/data/sft_smoke"
  require_dir "${smoke}"
  check_model
  require_dir "${MQ}"
  export TRAIN_FILE_DIR="${smoke}"
  export TRAIN_MODE=sft
  export OUTPUT_DIR="${OUTPUT_DIR:-${MQ}/outputs-sft-medical-smoke}"
  export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
  bash "${TRAIN_SH}/run_sft_qa.sh"
}

cmd_dpo() {
  check_model
  require_dir "${MQ}"
  export DPO_TRAIN_DIR="${DPO_TRAIN_DIR:?请设置 DPO_TRAIN_DIR（含 DPO 格式 jsonl 的目录）}"
  require_dir "${DPO_TRAIN_DIR}"
  export OUTPUT_DIR="${OUTPUT_DIR:-${MQ}/outputs-dpo-medical-pipeline}"
  bash "${TRAIN_SH}/run_dpo_pref.sh"
}

cmd_dpo_smoke() {
  local ddir="${DPO_TRAIN_DIR:-${MEDICAL_FULLSTACK}/data/dpo_smoke}"
  require_dir "${ddir}"
  check_model
  require_dir "${MQ}"
  export DPO_TRAIN_DIR="${ddir}"
  export DPO_MAX_STEPS="${SMOKE_DPO_STEPS}"
  export OUTPUT_DIR="${OUTPUT_DIR:-${MQ}/outputs-dpo-medical-smoke}"
  cd "${MQ}"
  torchrun --nproc_per_node "${NPROC:-2}" --master_port "${MASTER_PORT:-29519}" dpo_training.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --template_name qwen \
    --train_file_dir "${DPO_TRAIN_DIR}" \
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
    --max_steps "${DPO_MAX_STEPS}" \
    --eval_steps "${DPO_EVAL_STEPS:-25}" \
    --save_steps "${DPO_SAVE_STEPS:-25}" \
    --max_train_samples "${SMOKE_DPO_N}" \
    --output_dir "${OUTPUT_DIR}" \
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
}

cmd_grpo_smoke() {
  check_model
  require_dir "${MQ}"
  export GRPO_TRAIN_DIR="${GRPO_TRAIN_DIR:-${MEDICAL_FULLSTACK}/data/grpo_smoke}"
  require_dir "${GRPO_TRAIN_DIR}"
  export GRPO_TRAIN_SAMPLES="${GRPO_TRAIN_SAMPLES:-32}"
  export GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
  export GRPO_BS="${GRPO_BS:-1}"
  export OUTPUT_DIR="${OUTPUT_DIR:-${MQ}/outputs-grpo-medical-smoke}"
  bash "${TRAIN_SH}/run_grpo_pref.sh"
}

main() {
  local sub="${1:-help}"
  case "${sub}" in
    help|--help|-h) usage ;;
    print-env) cmd_print_env ;;
    prepare-smoke-sft) cmd_prepare_smoke_sft ;;
    prepare-smoke-dpo) cmd_prepare_smoke_dpo ;;
    rag-index) cmd_rag_index ;;
    cpt) cmd_cpt ;;
    sft) cmd_sft ;;
    sft-smoke) cmd_sft_smoke ;;
    dpo) cmd_dpo ;;
    dpo-smoke) cmd_dpo_smoke ;;
    grpo-smoke) cmd_grpo_smoke ;;
    all-smoke)
      cmd_prepare_smoke_sft
      cmd_sft_smoke
      if [[ -d "${MEDICAL_FULLSTACK}/data/dpo" ]] && compgen -G "${MEDICAL_FULLSTACK}/data/dpo/*.jsonl" >/dev/null; then
        DPO_SOURCE_DIR="${MEDICAL_FULLSTACK}/data/dpo" cmd_prepare_smoke_dpo
        export MODEL_NAME_OR_PATH="${MQ}/outputs-sft-medical-smoke"
        cmd_dpo_smoke
        export MODEL_NAME_OR_PATH="${MQ}/outputs-dpo-medical-smoke"
        cmd_grpo_smoke
      else
        log "未找到 data/dpo/*.jsonl，all-smoke 在 SFT 后结束。补充 DPO 数据后可: prepare-smoke-dpo && dpo-smoke && grpo-smoke"
      fi
      ;;
    *)
      die "未知命令: ${sub}  （执行 help 查看用法）"
      ;;
  esac
}

main "$@"
