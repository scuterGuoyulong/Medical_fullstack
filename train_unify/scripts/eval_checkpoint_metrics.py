#!/usr/bin/env python3
"""
对指定 LoRA checkpoint（如 checkpoint-120）在训练集（或任意同格式 JSON）上做生成评测，
并计算 BLEU-1~4、ROUGE-L、BERTScore。

数据格式：与项目 SFT 训练一致的 LLaVA-style JSON 列表，item["conversations"] 包含 human/gpt。
支持单图或多图（item["image"] 为 str 或 list[str]）。

依赖（若未安装）:
  pip install jieba nltk rouge-score bert-score

示例:
  cd Qwen-VL-Series-Finetune
  PYTHONPATH=src python scripts/eval_checkpoint_metrics.py \
    --checkpoint output/qwen35_xray_grounding_cot_lora/checkpoint-120 \
    --data_path output/xray_grounding_cot_sft.json \
    --image_folder /path/to/mimic-cxr-jpeg-sample200 \
    --bf16 \
    --batch_size 4 \
    --max_new_tokens 256 \
    --output_json output/eval_ckpt120_train_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.load_model import load_qwen_vl_generation_model  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from constants import LLAVA_IMAGE_TOKEN  # noqa: E402


def _require_eval_deps() -> None:
    try:
        import jieba  # noqa: F401
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # noqa: F401
        from rouge_score import rouge_scorer  # noqa: F401
        from bert_score import score as bert_score  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "缺少评测依赖，请执行: pip install jieba nltk rouge-score bert-score\n"
            f"原始错误: {e}"
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
    if not hyp_toks or not ref_toks:
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


def extract_question(human_value: str) -> str:
    t = human_value.replace(LLAVA_IMAGE_TOKEN, "").strip()
    t = re.sub(r"^\s*\n+", "", t).strip()
    return t


def _resolve_one_image_path(p: str, image_folder: str) -> str:
    if not os.path.isabs(p) and not p.startswith("http"):
        return os.path.join(image_folder, p)
    return p


def resolve_image_paths(image_field: str | list, image_folder: str) -> list[str]:
    if isinstance(image_field, list):
        return [_resolve_one_image_path(x, image_folder) for x in image_field]
    if not image_field:
        return []
    return [_resolve_one_image_path(str(image_field), image_folder)]


def build_conversation_multi(
    question: str,
    image_paths: list[str],
    image_min_pixels: int,
    image_max_pixels: int,
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = []
    for p in image_paths:
        user_content.append(
            {
                "type": "image",
                "image": p,
                "min_pixels": image_min_pixels,
                "max_pixels": image_max_pixels,
            }
        )
    user_content.append({"type": "text", "text": question})
    return [{"role": "user", "content": user_content}]


def iter_batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size 必须 >= 1")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def ensure_decoder_only_left_padding(processor) -> None:
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"


@torch.inference_mode()
def generate_answers_batch(
    model,
    processor,
    conversations: list[list[dict[str, Any]]],
    device: torch.device,
    max_new_tokens: int,
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
            raise ValueError("当前脚本仅支持图像，不支持视频。")
        prompts.append(prompt)
        batch_images.append(image_inputs)

    inputs = processor(
        text=prompts,
        images=batch_images,
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to(device)

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    outputs: list[str] = []
    for i in range(len(conversations)):
        new_tokens = gen_ids[i, prompt_len:]
        outputs.append(processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def _infer_base_model_from_checkpoint(checkpoint_dir: str) -> str | None:
    readme = Path(checkpoint_dir) / "README.md"
    if not readme.is_file():
        return None
    txt = readme.read_text(encoding="utf-8", errors="ignore")
    # huggingface model card header:
    # ---
    # base_model: /path/to/base
    # library_name: peft
    # ---
    m = re.search(r"^\s*base_model:\s*(.+?)\s*$", txt, flags=re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip()


def load_peft_checkpoint(base_model: str, checkpoint_dir: str, attn_impl: str, torch_dtype, device_map):
    from peft import PeftModel

    base = load_qwen_vl_generation_model(
        base_model,
        dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval()
    # processor/tokenizer from base is typically safer for checkpoints
    processor = AutoProcessor.from_pretrained(base_model)
    ensure_decoder_only_left_padding(processor)
    return model, processor


def aggregate_metrics(refs: list[str], hyps: list[str], bert_model_type: str, bert_device: str | None):
    _require_eval_deps()
    from nltk.translate.bleu_score import SmoothingFunction
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score

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

    # BERTScore (macro average)
    # Note: BERTScore expects lists of strings; use baseline rescaling for stability.
    P, R, F1 = bert_score(
        cands=hyps,
        refs=refs,
        lang="zh",
        model_type=bert_model_type,
        device=bert_device,
        rescale_with_baseline=True,
        verbose=False,
    )
    bert_p = float(P.mean().item()) if len(P) else 0.0
    bert_r = float(R.mean().item()) if len(R) else 0.0
    bert_f1 = float(F1.mean().item()) if len(F1) else 0.0

    if n_samples == 0:
        out = {**{f"bleu{n}": 0.0 for n in range(1, 5)}, "rougeL": 0.0}
    else:
        out = {k: v / n_samples for k, v in sums.items()}
        out["rougeL"] = rsum / n_samples
    out["bertscore_precision"] = bert_p
    out["bertscore_recall"] = bert_r
    out["bertscore_f1"] = bert_f1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="评测指定 checkpoint 在训练集上的生成指标")
    parser.add_argument("--checkpoint", type=str, required=True, help="LoRA checkpoint 目录，如 .../checkpoint-120")
    parser.add_argument("--base_model", type=str, default="", help="可选：显式指定 base 模型目录（不填则从 checkpoint README 推断）")
    parser.add_argument("--data_path", type=str, required=True, help="训练集 JSON（或任意同格式 JSON）")
    parser.add_argument("--image_folder", type=str, required=True, help="图像根目录（相对路径会拼接到这里）")
    parser.add_argument("--image_min_pixels", type=int, default=384 * 32 * 32)
    parser.add_argument("--image_max_pixels", type=int, default=1280 * 32 * 32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=-1, help=">0 时只评前 N 条（调试）")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_flash_attention_2", action="store_true")
    parser.add_argument("--device_map", type=str, default="auto", help='传给 from_pretrained，如 "auto" 或 "cuda:0"')
    parser.add_argument("--bert_model_type", type=str, default="bert-base-chinese", help="BERTScore 使用的模型")
    parser.add_argument("--bert_device", type=str, default="", help='BERTScore 设备，如 "cuda:0"；默认自动')
    parser.add_argument("--output_json", type=str, default="", help="若设置，将写入该 JSON（含逐样本预测）")
    args = parser.parse_args()

    if args.bf16 and args.fp16:
        raise SystemExit("不要同时指定 --bf16 与 --fp16")

    checkpoint = str(Path(args.checkpoint).resolve())
    base_model = (args.base_model or "").strip()
    if not base_model:
        base_model = _infer_base_model_from_checkpoint(checkpoint) or ""
    if not base_model:
        raise SystemExit("无法推断 base_model，请显式传入 --base_model /path/to/base_model")

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

    refs: list[str] = []
    conversations: list[list[dict[str, Any]]] = []
    for item in data:
        convs = item["conversations"]
        human = next(c["value"] for c in convs if c["from"] == "human")
        ref = next(c["value"] for c in convs if c["from"] == "gpt")
        refs.append((ref or "").strip())
        q = extract_question(human or "")
        image_paths = resolve_image_paths(item.get("image") or item.get("images") or "", args.image_folder)
        conversations.append(
            build_conversation_multi(q, image_paths, args.image_min_pixels, args.image_max_pixels)
        )

    print(f"[load] base_model={base_model}", flush=True)
    print(f"[load] checkpoint={checkpoint}", flush=True)
    print(f"[data] samples={len(conversations)}  batch_size={args.batch_size}", flush=True)

    model, processor = load_peft_checkpoint(base_model, checkpoint, attn_impl, torch_dtype, device_map)
    device = next(model.parameters()).device

    preds: list[str] = []
    processed = 0
    for batch in iter_batches(conversations, args.batch_size):
        preds.extend(generate_answers_batch(model, processor, batch, device, args.max_new_tokens))
        processed += len(batch)
        if processed % 20 == 0 or processed == len(conversations):
            print(f"  generated {processed}/{len(conversations)}", flush=True)

    metrics = aggregate_metrics(
        refs,
        preds,
        bert_model_type=args.bert_model_type,
        bert_device=(args.bert_device or None),
    )

    print("\n========== 评测结果（宏平均 sentence BLEU + ROUGE-L F1 + BERTScore）==========")
    print(f"样本数: {len(refs)}")
    for n in range(1, 5):
        print(f"  BLEU-{n}: {metrics[f'bleu{n}']:.4f}")
    print(f"  ROUGE-L: {metrics['rougeL']:.4f}")
    print(f"  BERTScore(P): {metrics['bertscore_precision']:.4f}")
    print(f"  BERTScore(R): {metrics['bertscore_recall']:.4f}")
    print(f"  BERTScore(F1): {metrics['bertscore_f1']:.4f}")
    print("================================================================\n")

    if args.output_json:
        out = {
            "checkpoint": checkpoint,
            "base_model": base_model,
            "data_path": args.data_path,
            "image_folder": args.image_folder,
            "num_samples": len(refs),
            "metrics": metrics,
            "samples": [
                {"index": i, "reference": r, "prediction": p}
                for i, (r, p) in enumerate(zip(refs, preds, strict=True))
            ],
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()

