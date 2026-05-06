#!/usr/bin/env python3
"""
从医学 SFT JSON（ShareGPT + image）构造患者端偏好 DPO 数据。

支持两种构建模式：

1) hybrid（推荐）：折中策略
   - chosen：reference 轻改（文本模型）或 SFT-VL + 患者向 system 单条采样
   - rejected：SFT-VL 高温度采样，或 SFT-VL + 医生腔 system 单条采样
   - 法官：文本大模型，仅看「问题 + 回答甲/乙」（不看图）

2) text_only：仅用文本模型对参考答案做「患者向 / 劣化」双稿 + 法官（旧逻辑）

输出：id, image, prompt, chosen, rejected（与 README DPO 示例一致）。

示例（混合 + SFT 权重）：
  cd medical_fullstack/train_unify
  export PYTHONPATH=src:$PYTHONPATH
  python scripts/build_medical_dpo_patient_judge.py \\
    --build_mode hybrid \\
    --vl_model_path output/qwen35_4b_medical_sft \\
    --judge_model_path ../../models/models/Qwen/Qwen3___5-4B \\
    --chosen_mode reference_light \\
    --rejected_mode sft_doctor \\
    --image_folder ../../DataSets/medical/mixed_sft \\
    --ensure_placeholder \\
    --max_samples 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]  # medical_fullstack/train_unify
ANDES_VL_ROOT = REPO_ROOT.parent.parent  # andes_vl（含 models/models/Qwen/...）
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.load_model import load_qwen_vl_generation_model  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402


SYS_REFERENCE_LIGHT = """你是医学编辑。下面「参考答案」内容正确，请做**轻度**患者向润色：
- 尽量保留原文句子与信息点，不要扩写成长篇；
- 可调整分段与标点，开头用一两句直接回应问题；
- 语气稍温和；必要处给术语加括号通俗解释；
- 不新增事实、不新诊断、不编造检查结论；不大幅删减要点。

直接输出润色后的正文，不要前言后语。"""

SYS_PATIENT_WRITER = """你是一名医学科普编辑。任务：在严格保持与「参考答案」的医学事实一致的前提下（不新增诊断、不编造检查或影像所见、不改动核心结论），将内容改写为面向普通患者的回复。

必须做到：语气共情、温和；开头先用一两句直接回应患者问题；正文用短段落或分点，便于扫读；专业词首次出现时简短通俗解释；信息不足时明确说明需结合面诊或检查、不能网上确诊；提醒不能替代医生诊疗；涉及用药或剂量时强调遵医嘱；如有急重症可能（如胸痛、呼吸困难、意识障碍、大出血等）须提示尽快就医。

不要删减关键医学要点。直接输出改写后的正文，不要标题、不要「改写如下」等套话。"""

SYS_BAD_WRITER = """你是一名助手。下面有一份医学问答的「参考答案」。请故意改写成「不适合患者端 App」的版本，用于作为负例对照：

要求：堆砌术语与缩写、少分段（尽量大段文字）；语气冷淡或命令式；尽量不写就医边界与不确定性；把推测写得像已确诊；不要编造参考答案中完全没有的新疾病名或新药名。仍需大体保留原文的主要信息点，不要敷衍成一两句话。

直接输出改写后的正文，不要解释。"""

SYS_JUDGE = """你是患者端医学问答产品的质量评审（法官）。

你会看到：用户问题，以及两篇匿名回答「回答甲」「回答乙」。其中一篇应更符合患者端产品标准（通俗、结构清晰、边界与安全提示更合理），另一篇为刻意劣化的对照稿。

评判优先级：
1）医学内容不与问题明显矛盾、不凭空捏造严重错误事实；
2）更适合普通患者理解；
3）风险与边界（不确定表述、建议就医、勿自行用药等）更合理。

仅输出一行 JSON 对象，不要 markdown，不要其它文字。格式严格为：
{"preferred":"甲","valid":true}
或
{"preferred":"乙","valid":true}
若两篇均存在严重医学错误或与问题无关，则 valid 为 false，preferred 可为 null。"""

SYS_VL_PATIENT = """你是面向普通患者的医学助手。回答要求：语气共情、温和；先简短回应核心问题，再分点或短段落说明；专业词用通俗话解释；信息不足时说明需面诊或检查、不能替代医生；用药须强调遵医嘱；有急重症风险时提示尽快就医。"""

SYS_VL_DOCTOR = """你是面向临床医生的助手。回答要求：优先使用专业术语与缩写，表述紧凑；少写科普式解释；不要对患者端式的免责声明与「建议尽快就医」等话术；语气客观、偏学术笔记风格。"""


def _strip_think(text: str) -> str:
    if "</redacted_thinking>" in text:
        text = text.split("</redacted_thinking>")[-1]
    text = re.sub(r"^<redacted_thinking>[\s\S]*?</redacted_thinking>\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _parse_judge_json(line: str) -> dict | None:
    line = _strip_think(line).strip()
    line = line.replace("\u201c", '"').replace("\u201d", '"').replace("：", ":")
    m = re.search(r"\{[^{}]*\}", line)
    if not m:
        return None
    raw = m.group()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw2 = re.sub(r"'(甲|乙|null|true|false)'", r'"\1"', raw)
        try:
            return json.loads(raw2)
        except json.JSONDecodeError:
            return None


def build_chat_inputs(tokenizer, system: str, user: str, enable_thinking: bool = False):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        text = tokenizer.apply_chat_template(
            messages, **kwargs, enable_thinking=enable_thinking
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return tokenizer(text, return_tensors="pt")


def ensure_decoder_only_left_padding(processor) -> None:
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"


def build_vl_conversation(
    question: str,
    image_path: str,
    image_min_pixels: int,
    image_max_pixels: int,
    system: str | None = None,
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        {
            "type": "image",
            "image": image_path,
            "min_pixels": image_min_pixels,
            "max_pixels": image_max_pixels,
        },
        {"type": "text", "text": question},
    ]
    conv: list[dict[str, Any]] = []
    if system and system.strip():
        conv.append({"role": "system", "content": system.strip()})
    conv.append({"role": "user", "content": user_content})
    return conv


def resolve_image_path(image_field: str | list, image_folder: str) -> str:
    if isinstance(image_field, list):
        image_field = image_field[0]
    p = image_field
    if not os.path.isabs(p) and not str(p).startswith("http"):
        p = os.path.join(image_folder, p)
    return p


@torch.inference_mode()
def generate_text_one(
    model,
    tokenizer,
    system: str,
    user: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    enable_thinking: bool = False,
) -> str:
    inputs = build_chat_inputs(tokenizer, system, user, enable_thinking=enable_thinking)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    do_sample = temperature > 0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(0.01, temperature) if do_sample else 1.0,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen = out[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return _strip_think(text).strip()


@torch.inference_mode()
def generate_vl_one(
    model,
    processor,
    conversation: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    gen_kw: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = max(0.01, temperature)
        gen_kw["top_p"] = top_p
    else:
        gen_kw["do_sample"] = False
    gen_ids = model.generate(**inputs, **gen_kw)
    in_len = inputs["input_ids"].shape[1]
    new_tokens = gen_ids[0, in_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_sft_fields(obj: dict) -> tuple[str, str, str, str]:
    convs = obj["conversations"]
    human = next(x for x in convs if x["from"] == "human")
    gpt = next(x for x in convs if x["from"] == "gpt")
    prompt = human["value"]
    reference = gpt["value"]
    image = obj.get("image", "")
    if isinstance(image, list):
        image = image[0]
    mid = obj.get("meta", {}).get("_dedup_sha256") or str(uuid.uuid4())
    return prompt, reference, str(image), mid


def ensure_placeholder(image_folder: Path) -> None:
    dest = image_folder / "qa_placeholder.jpg"
    if dest.exists():
        return
    try:
        from PIL import Image
    except ImportError:
        print(
            "[warn] 未安装 pillow，无法自动生成 qa_placeholder.jpg；"
            "请自备占位图或 pip install pillow",
            file=sys.stderr,
        )
        return
    image_folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(dest, format="JPEG", quality=85)
    print(f"[info] 已写入占位图: {dest}")


def pick_judge_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg.startswith("cuda"):
        return torch.device(arg)
    # auto
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        return torch.device("cuda:1")
    return torch.device("cpu")


def load_vl_model(
    vl_path: str,
    vl_base_model: str | None,
    attn_impl: str,
    torch_dtype: torch.dtype,
    device_map: str | dict,
):
    adapter_cfg = os.path.join(vl_path, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        if not vl_base_model or not os.path.isdir(vl_base_model):
            raise SystemExit("检测到 LoRA（adapter_config.json），请同时传入 --vl_base_model 指向合并前的基座。")
        from peft import PeftModel

        base = load_qwen_vl_generation_model(
            vl_base_model,
            dtype=torch_dtype,
            attn_implementation=attn_impl,
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(base, vl_path)
        model.eval()
        processor = AutoProcessor.from_pretrained(vl_path)
    else:
        model = load_qwen_vl_generation_model(
            vl_path,
            dtype=torch_dtype,
            attn_implementation=attn_impl,
            device_map=device_map,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(vl_path)
    ensure_decoder_only_left_padding(processor)
    return model, processor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--build_mode",
        choices=("hybrid", "text_only"),
        default="hybrid",
        help="hybrid：VL 生成 chosen/rejected 之一或两者 + 文本法官；text_only：仅用文本模型双稿",
    )
    ap.add_argument(
        "--sft_json",
        type=str,
        default=str(REPO_ROOT.parent / "DataSets/medical/mixed_sft/train_qa_report_qwen_vl.json"),
    )
    ap.add_argument("--output_json", type=str, default=str(REPO_ROOT / "output/medical_dpo_patient_judged.json"))
    ap.add_argument(
        "--judge_model_path",
        type=str,
        default=str(ANDES_VL_ROOT / "models" / "models" / "Qwen" / "Qwen3___5-4B"),
        help="文本模型：做法官；hybrid 下还可用于 chosen_mode=reference_light（默认与 eval_medical_bleu_rouge 一致）",
    )
    ap.add_argument(
        "--model_path",
        type=str,
        default="",
        help="已弃用别名：等同于 --judge_model_path（仅当未传 judge_model_path 时使用）",
    )
    ap.add_argument(
        "--vl_model_path",
        type=str,
        default="",
        help="SFT 后的 Qwen-VL 目录（hybrid 必填，除非仅用 reference_light + text_only rejected，当前 hybrid 必用）",
    )
    ap.add_argument(
        "--vl_base_model",
        type=str,
        default="",
        help="若 vl_model_path 为 LoRA，则填基座模型目录",
    )
    ap.add_argument("--image_folder", type=str, default=str(REPO_ROOT.parent / "DataSets/medical/mixed_sft"))
    ap.add_argument("--image_min_pixels", type=int, default=512 * 32 * 32)
    ap.add_argument("--image_max_pixels", type=int, default=1280 * 32 * 32)
    ap.add_argument(
        "--chosen_mode",
        choices=("reference_light", "sft_patient"),
        default="reference_light",
        help="chosen：参考答案轻改（文本）或 SFT-VL+患者 system",
    )
    ap.add_argument(
        "--rejected_mode",
        choices=("sft_hot", "sft_doctor"),
        default="sft_doctor",
        help="rejected：SFT-VL 高温度采样 或 医生腔 system",
    )
    ap.add_argument("--max_samples", type=int, default=0, help="0 表示全量")
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens_vl", type=int, default=1024)
    ap.add_argument("--max_new_tokens_text", type=int, default=2048)
    ap.add_argument("--max_new_tokens_judge", type=int, default=256)
    ap.add_argument("--temperature_reference_light", type=float, default=0.35)
    ap.add_argument("--temperature_sft_patient", type=float, default=0.5)
    ap.add_argument("--temperature_rejected_hot", type=float, default=1.15)
    ap.add_argument("--temperature_judge", type=float, default=0.2)
    ap.add_argument(
        "--temperature_text_only",
        type=float,
        default=0.6,
        help="build_mode=text_only 时，患者稿与劣化稿生成的采样温度",
    )
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--skip_judge", action="store_true")
    ap.add_argument("--ensure_placeholder", action="store_true")
    ap.add_argument(
        "--judge_device",
        type=str,
        default="auto",
        help="法官模型所在设备：auto（双卡时 cuda:1，否则 cpu）/ cpu / cuda:0 …；避免与 VL 抢显存",
    )
    ap.add_argument(
        "--vl_device_map",
        type=str,
        default="auto",
        help='VL 模型 device_map，如 "auto" 或 "cuda:0"',
    )
    ap.add_argument(
        "--use_flash_attention_2",
        action="store_true",
        help="VL 使用 flash_attention_2；默认 sdpa",
    )
    ap.add_argument("--bf16", action="store_true", help="VL 与文本模型推理用 bf16（推荐）")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    judge_path = args.judge_model_path
    if args.model_path.strip():
        judge_path = args.model_path.strip()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    sft_path = Path(args.sft_json)
    out_path = Path(args.output_json)
    image_folder = Path(args.image_folder)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.ensure_placeholder:
        ensure_placeholder(image_folder)

    if not sft_path.is_file():
        raise SystemExit(f"找不到 SFT 文件: {sft_path}")

    if args.bf16 and args.fp16:
        raise SystemExit("不要同时指定 --bf16 与 --fp16")

    if args.build_mode == "hybrid" and not args.vl_model_path.strip():
        raise SystemExit("build_mode=hybrid 需要 --vl_model_path 指向 SFT 后的 Qwen-VL。")

    with open(sft_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    end = len(data) if args.max_samples <= 0 else min(len(data), args.start_index + args.max_samples)
    slice_ = data[args.start_index : end]

    if not os.path.isdir(judge_path):
        raise SystemExit(f"法官模型目录不存在: {judge_path}")

    text_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    if torch.cuda.is_available() and not args.fp16 and not args.bf16:
        text_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    judge_dev = pick_judge_device(args.judge_device)
    judge_map = {"": str(judge_dev)} if judge_dev.type == "cuda" else {"": "cpu"}
    judge_tok = AutoTokenizer.from_pretrained(judge_path, trust_remote_code=True)
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_path,
        trust_remote_code=True,
        torch_dtype=text_dtype,
        device_map=judge_map,
    )
    judge_model.eval()

    vl_model = None
    vl_processor = None
    vl_dev: torch.device | None = None
    if args.build_mode == "hybrid":
        vl_path = args.vl_model_path.strip()
        vl_base = args.vl_base_model.strip() or None
        vl_dm = args.vl_device_map
        if vl_dm not in ("auto", "balanced", "sequential") and vl_dm.startswith("cuda"):
            vl_dm = {"": vl_dm}
        attn_impl = "flash_attention_2" if args.use_flash_attention_2 else "sdpa"
        vl_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        vl_model, vl_processor = load_vl_model(vl_path, vl_base, attn_impl, vl_dtype, vl_dm)
        vl_dev = next(vl_model.parameters()).device

    results: list[dict] = []
    stats = {"ok": 0, "skip_judge_fail": 0, "skip_wrong_preference": 0, "skip_empty": 0, "skip_same": 0}

    for i, obj in enumerate(slice_):
        global_i = args.start_index + i
        try:
            prompt, reference, image_name, sid = extract_sft_fields(obj)
        except (StopIteration, KeyError) as e:
            print(f"[warn] 跳过 malformed 样本 index={global_i}: {e}", file=sys.stderr)
            continue

        q_plain = re.sub(r"^<image>\s*", "", prompt).strip()
        img_abs = resolve_image_path(image_name, str(image_folder))
        if args.build_mode == "hybrid" and not img_abs.startswith("http") and not os.path.isfile(img_abs):
            print(f"[warn] 图像不存在，跳过 index={global_i}: {img_abs}", file=sys.stderr)
            continue

        chosen_text = ""
        rejected_text = ""

        try:
            if args.build_mode == "text_only":
                user_patient = (
                    f"用户问题（文字）：\n{q_plain}\n\n参考答案：\n{reference}\n\n请按要求输出患者向正文。"
                )
                user_bad = f"参考答案：\n{reference}\n\n请按要求输出对照稿正文。"
                chosen_text = generate_text_one(
                    judge_model,
                    judge_tok,
                    SYS_PATIENT_WRITER,
                    user_patient,
                    args.max_new_tokens_text,
                    args.temperature_text_only,
                    args.top_p,
                    judge_dev,
                )
                rejected_text = generate_text_one(
                    judge_model,
                    judge_tok,
                    SYS_BAD_WRITER,
                    user_bad,
                    args.max_new_tokens_text,
                    args.temperature_text_only,
                    args.top_p,
                    judge_dev,
                )
            else:
                assert vl_model is not None and vl_processor is not None and vl_dev is not None
                if args.chosen_mode == "reference_light":
                    user_light = f"用户问题：\n{q_plain}\n\n参考答案：\n{reference}\n\n请输出润色后的正文。"
                    chosen_text = generate_text_one(
                        judge_model,
                        judge_tok,
                        SYS_REFERENCE_LIGHT,
                        user_light,
                        args.max_new_tokens_text,
                        args.temperature_reference_light,
                        args.top_p,
                        judge_dev,
                    )
                else:
                    conv_p = build_vl_conversation(
                        q_plain,
                        img_abs,
                        args.image_min_pixels,
                        args.image_max_pixels,
                        SYS_VL_PATIENT,
                    )
                    chosen_text = generate_vl_one(
                        vl_model,
                        vl_processor,
                        conv_p,
                        vl_dev,
                        args.max_new_tokens_vl,
                        do_sample=True,
                        temperature=args.temperature_sft_patient,
                        top_p=args.top_p,
                    )

                if args.rejected_mode == "sft_hot":
                    conv_r = build_vl_conversation(
                        q_plain,
                        img_abs,
                        args.image_min_pixels,
                        args.image_max_pixels,
                        system=None,
                    )
                    rejected_text = generate_vl_one(
                        vl_model,
                        vl_processor,
                        conv_r,
                        vl_dev,
                        args.max_new_tokens_vl,
                        do_sample=True,
                        temperature=args.temperature_rejected_hot,
                        top_p=args.top_p,
                    )
                else:
                    conv_r = build_vl_conversation(
                        q_plain,
                        img_abs,
                        args.image_min_pixels,
                        args.image_max_pixels,
                        SYS_VL_DOCTOR,
                    )
                    rejected_text = generate_vl_one(
                        vl_model,
                        vl_processor,
                        conv_r,
                        vl_dev,
                        args.max_new_tokens_vl,
                        do_sample=True,
                        temperature=args.temperature_sft_patient,
                        top_p=args.top_p,
                    )
        except Exception as e:
            print(f"[warn] 生成失败 index={global_i}: {e}", file=sys.stderr)
            continue

        if not chosen_text or not rejected_text:
            stats["skip_empty"] += 1
            continue
        if chosen_text.strip() == rejected_text.strip():
            stats["skip_same"] += 1
            continue

        if args.skip_judge:
            results.append(
                {
                    "id": sid,
                    "image": image_name,
                    "prompt": prompt,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                }
            )
            stats["ok"] += 1
            continue

        patient_first = random.random() < 0.5
        if patient_first:
            text_a, text_b = chosen_text, rejected_text
            preferred_tag_for_patient = "甲"
        else:
            text_a, text_b = rejected_text, chosen_text
            preferred_tag_for_patient = "乙"

        judge_user = (
            f"用户问题：\n{q_plain}\n\n"
            f"回答甲：\n{text_a}\n\n"
            f"回答乙：\n{text_b}\n\n"
            "请输出一行 JSON。"
        )
        try:
            judge_out = generate_text_one(
                judge_model,
                judge_tok,
                SYS_JUDGE,
                judge_user,
                args.max_new_tokens_judge,
                args.temperature_judge,
                args.top_p,
                judge_dev,
            )
        except Exception as e:
            print(f"[warn] 法官推理失败 index={global_i}: {e}", file=sys.stderr)
            stats["skip_judge_fail"] += 1
            continue

        parsed = _parse_judge_json(judge_out)
        if not parsed or "valid" not in parsed:
            stats["skip_judge_fail"] += 1
            continue
        if not parsed.get("valid"):
            stats["skip_judge_fail"] += 1
            continue

        pref = parsed.get("preferred")
        if pref not in ("甲", "乙"):
            stats["skip_judge_fail"] += 1
            continue

        if pref != preferred_tag_for_patient:
            stats["skip_wrong_preference"] += 1
            continue

        results.append(
            {
                "id": sid,
                "image": image_name,
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
            }
        )
        stats["ok"] += 1

        if (stats["ok"] % 10) == 0 and stats["ok"] > 0:
            print(f"[progress] committed={stats['ok']} (global index ~{global_i})")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "output": str(out_path),
                "written": len(results),
                "stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
