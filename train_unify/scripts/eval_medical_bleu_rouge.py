#!/usr/bin/env python3
"""
在医学 SFT 同款 JSON 上，对基座模型与微调 checkpoint 分别推理，计算 BLEU-1~4 与 ROUGE-L。

中文：使用 jieba 分词后计算 sentence BLEU（宏平均）与 ROUGE-L F1。

依赖（若未安装）:
  pip install jieba nltk rouge-score

用法示例:
  cd Qwen-VL-Series-Finetune
  PYTHONPATH=src python scripts/eval_medical_bleu_rouge.py \\
    --base_model /path/to/Qwen3___5-4B \\
    --finetuned_model /path/to/output/qwen35_4b_medical_sft \\
    --data_path /path/to/train_qa_report_qwen_vl.json \\
    --image_folder /path/to/mimic-cxr-jpeg-sample200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.load_model import load_qwen_vl_generation_model  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from dataset.data_utils import get_qwen_multimodal_settings  # noqa: E402
from constants import LLAVA_IMAGE_TOKEN  # noqa: E402


def _require_eval_deps():
    try:
        import jieba  # noqa: F401
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # noqa: F401
        from rouge_score import rouge_scorer  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "缺少评测依赖，请执行: pip install jieba nltk rouge-score\n" f"原始错误: {e}"
        ) from e


def tokenize_zh(text: str) -> list[str]:
    import jieba

    text = (text or "").strip()
    if not text:
        return []
    return list(jieba.cut(text))


def bleu_n(reference: str, hypothesis: str, n: int, smoothing) -> float:
    from nltk.translate.bleu_score import sentence_bleu

    ref_toks = tokenize_zh(reference)
    hyp_toks = tokenize_zh(hypothesis)
    if not hyp_toks:
        return 0.0
    if not ref_toks:
        return 0.0
    weights_map = {
        1: (1.0, 0.0, 0.0, 0.0),
        2: (0.5, 0.5, 0.0, 0.0),
        3: (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        4: (0.25, 0.25, 0.25, 0.25),
    }
    return float(
        sentence_bleu(
            [ref_toks],
            hyp_toks,
            weights=weights_map[n],
            smoothing_function=smoothing,
        )
    )


def rouge_l_f1(reference: str, hypothesis: str, scorer) -> float:
    ref_j = " ".join(tokenize_zh(reference))
    hyp_j = " ".join(tokenize_zh(hypothesis))
    if not ref_j.strip() or not hyp_j.strip():
        return 0.0
    return float(scorer.score(ref_j, hyp_j)["rougeL"].fmeasure)


def resolve_image_path(image_field: str | list, image_folder: str) -> str:
    if isinstance(image_field, list):
        image_field = image_field[0]
    p = image_field
    if not os.path.isabs(p) and not p.startswith("http"):
        p = os.path.join(image_folder, p)
    return p


def extract_question(human_value: str) -> str:
    """去掉 <image> 等前缀，得到纯文本问题。"""
    t = human_value.replace(LLAVA_IMAGE_TOKEN, "").strip()
    t = re.sub(r"^\s*\n+", "", t).strip()
    return t


def build_conversation(
    question: str,
    image_path: str,
    image_min_pixels: int,
    image_max_pixels: int,
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
    return [{"role": "user", "content": user_content}]


def iter_batches(items: list[Any], batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size 必须 >= 1")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def ensure_decoder_only_left_padding(processor) -> None:
    """decoder-only 模型 batch 生成要求左填充，否则 transformers 会警告且结果可能异常。"""
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"


@torch.inference_mode()
def generate_answer(
    model,
    processor,
    conversation: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
    dtype: torch.dtype,
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
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    in_len = inputs["input_ids"].shape[1]
    new_tokens = gen_ids[0, in_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_answers_batch(
    model,
    processor,
    conversations: list[list[dict[str, Any]]],
    device: torch.device,
    max_new_tokens: int,
    dtype: torch.dtype,
) -> list[str]:
    prompts: list[str] = []
    batch_images: list[Any] = []

    for conversation in conversations:
        prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        if video_inputs:
            raise ValueError("当前评测脚本的 batch 推理仅支持图像，不支持视频。")
        if isinstance(image_inputs, list) and len(image_inputs) == 1:
            image_inputs = image_inputs[0]
        prompts.append(prompt)
        batch_images.append(image_inputs)

    inputs = processor(
        text=prompts,
        images=batch_images,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    # 必须与 HF generate 约定一致：输出前 prompt_len 列是（左/右填充后的）整段 input，
    # 续写从 prompt_len 开始。不能用 attention_mask.sum：左填充时 sum 是「非 pad 数」，
    # 不等于张量里 prompt 结束列索引，会导致切片错位、解码为空或乱码。
    prompt_len = inputs["input_ids"].shape[1]

    outputs: list[str] = []
    for i in range(len(conversations)):
        new_tokens = gen_ids[i, prompt_len:]
        outputs.append(processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def load_model_and_processor(
    model_path: str,
    attn_implementation: str,
    torch_dtype: torch.dtype,
    device_map: str | dict,
):
    model = load_qwen_vl_generation_model(
        model_path,
        dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    ensure_decoder_only_left_padding(processor)
    return model, processor


def maybe_load_peft(base_path: str, finetuned_path: str, attn_impl: str, torch_dtype, device_map):
    from peft import PeftModel

    base = load_qwen_vl_generation_model(
        base_path,
        dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base, finetuned_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(finetuned_path)
    ensure_decoder_only_left_padding(processor)
    return model, processor


def aggregate_metrics(refs: list[str], hyps: list[str]) -> dict[str, float]:
    _require_eval_deps()
    from nltk.translate.bleu_score import SmoothingFunction
    from rouge_score import rouge_scorer

    import logging

    import jieba

    logging.getLogger("jieba").setLevel(logging.ERROR)
    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    n_samples = len(refs)
    sums = {f"bleu{n}": 0.0 for n in range(1, 5)}
    rsum = 0.0
    for ref, hyp in zip(refs, hyps, strict=True):
        for n in range(1, 5):
            sums[f"bleu{n}"] += bleu_n(ref, hyp, n, smooth)
        rsum += rouge_l_f1(ref, hyp, scorer)
    if n_samples == 0:
        return {**{f"bleu{n}": 0.0 for n in range(1, 5)}, "rougeL": 0.0}
    out = {k: v / n_samples for k, v in sums.items()}
    out["rougeL"] = rsum / n_samples
    return out


def main():
    parser = argparse.ArgumentParser(description="BLEU / ROUGE-L 评测（基座 vs 微调）")
    parser.add_argument("--base_model", type=str, required=True, help="原始 HuggingFace 模型目录")
    parser.add_argument(
        "--finetuned_model",
        type=str,
        required=True,
        help="微调输出目录（全量权重或 LoRA；LoRA 时需与训练时相同的 base）",
    )
    parser.add_argument("--data_path", type=str, required=True, help="与训练相同的 JSON 列表路径")
    parser.add_argument("--image_folder", type=str, required=True, help="图像根目录")
    parser.add_argument(
        "--processor_source",
        type=str,
        default="base",
        choices=("base", "finetuned"),
        help="processor 从基座还是微调目录加载（两者 tokenizer 一般一致）",
    )
    parser.add_argument("--image_min_pixels", type=int, default=512 * 32 * 32)
    parser.add_argument("--image_max_pixels", type=int, default=1280 * 32 * 32)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1, help="评测推理 batch size，显存不足时调小")
    parser.add_argument("--max_samples", type=int, default=-1, help=">0 时只评前 N 条（调试）")
    parser.add_argument("--bf16", action="store_true", help="推理用 bfloat16（与训练一致可开）")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--use_flash_attention_2",
        action="store_true",
        help="使用 flash_attention_2；默认使用 SDPA（与 medical_sft 脚本一致）",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='传给 from_pretrained，如 "auto" 或 "cuda:0"',
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="若设置，将预测与分数写入该 JSON 文件",
    )
    args = parser.parse_args()

    if args.bf16 and args.fp16:
        raise SystemExit("不要同时指定 --bf16 与 --fp16")
    if args.batch_size < 1:
        raise SystemExit("--batch_size 必须 >= 1")

    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    attn_impl = "flash_attention_2" if args.use_flash_attention_2 else "sdpa"
    device_map = args.device_map
    if device_map not in ("auto", "balanced", "sequential") and device_map.startswith("cuda"):
        device_map = {"": device_map}

    _require_eval_deps()

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.max_samples and args.max_samples > 0:
        data = data[: args.max_samples]

    model_id_for_patch = args.base_model
    _, _, _ = get_qwen_multimodal_settings(model_id_for_patch)

    refs: list[str] = []
    conversations: list[list[dict[str, Any]]] = []
    for item in data:
        convs = item["conversations"]
        human = next(c["value"] for c in convs if c["from"] == "human")
        ref = next(c["value"] for c in convs if c["from"] == "gpt")
        refs.append(ref.strip())
        q = extract_question(human)
        img_path = resolve_image_path(item.get("image", ""), args.image_folder)
        conversations.append(
            build_conversation(q, img_path, args.image_min_pixels, args.image_max_pixels)
        )

    preds_base: list[str] = []
    preds_ft: list[str] = []

    # ----- base -----
    print("[1/2] 加载基座模型并推理 …", flush=True)
    model_b, proc_b = load_model_and_processor(
        args.base_model, attn_impl, torch_dtype, device_map
    )
    dev = next(model_b.parameters()).device
    processed = 0
    for batch in iter_batches(conversations, args.batch_size):
        if len(batch) == 1:
            preds_base.append(
                generate_answer(
                    model_b, proc_b, batch[0], dev, args.max_new_tokens, torch_dtype
                )
            )
        else:
            preds_base.extend(
                generate_answers_batch(
                    model_b, proc_b, batch, dev, args.max_new_tokens, torch_dtype
                )
            )
        processed += len(batch)
        if processed % 10 == 0 or processed == len(conversations) or processed == len(batch):
            print(f"  base {processed}/{len(conversations)} (batch={len(batch)})", flush=True)
    del model_b
    torch.cuda.empty_cache()

    # ----- finetuned -----
    print("[2/2] 加载微调模型并推理 …", flush=True)
    ft_path = args.finetuned_model
    adapter_cfg = os.path.join(ft_path, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        model_f, proc_f = maybe_load_peft(
            args.base_model, ft_path, attn_impl, torch_dtype, device_map
        )
    else:
        model_f, proc_f = load_model_and_processor(
            ft_path, attn_impl, torch_dtype, device_map
        )
    if args.processor_source == "base":
        proc_f = AutoProcessor.from_pretrained(args.base_model)
        ensure_decoder_only_left_padding(proc_f)

    dev = next(model_f.parameters()).device
    processed = 0
    for batch in iter_batches(conversations, args.batch_size):
        if len(batch) == 1:
            preds_ft.append(
                generate_answer(
                    model_f, proc_f, batch[0], dev, args.max_new_tokens, torch_dtype
                )
            )
        else:
            preds_ft.extend(
                generate_answers_batch(
                    model_f, proc_f, batch, dev, args.max_new_tokens, torch_dtype
                )
            )
        processed += len(batch)
        if processed % 10 == 0 or processed == len(conversations) or processed == len(batch):
            print(f"  ft {processed}/{len(conversations)} (batch={len(batch)})", flush=True)
    del model_f
    torch.cuda.empty_cache()

    m_base = aggregate_metrics(refs, preds_base)
    m_ft = aggregate_metrics(refs, preds_ft)

    print("\n========== 评测结果（宏平均 sentence BLEU + ROUGE-L F1）==========")
    print(f"样本数: {len(refs)}")
    print("\n[基座]")
    for n in range(1, 5):
        print(f"  BLEU-{n}: {m_base[f'bleu{n}']:.4f}")
    print(f"  ROUGE-L: {m_base['rougeL']:.4f}")
    print("\n[微调]")
    for n in range(1, 5):
        print(f"  BLEU-{n}: {m_ft[f'bleu{n}']:.4f}")
    print(f"  ROUGE-L: {m_ft['rougeL']:.4f}")
    print("================================================================\n")

    if args.output_json:
        out = {
            "metrics_base": m_base,
            "metrics_finetuned": m_ft,
            "num_samples": len(refs),
            "samples": [
                {
                    "reference": r,
                    "pred_base": pb,
                    "pred_finetuned": pf,
                }
                for r, pb, pf in zip(refs, preds_base, preds_ft, strict=True)
            ],
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"已写入: {args.output_json}")


if __name__ == "__main__":
    main()
