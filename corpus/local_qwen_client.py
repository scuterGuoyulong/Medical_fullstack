# -*- coding: utf-8 -*-
"""本地 Qwen 推理封装：供 translate_reports.py / rewrite_quality.py 调用。"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_tokenizer(
    model_name_or_path: str,
    tokenizer_path: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    tok_src = tokenizer_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(
        tokenizer, "eos_token", None
    ) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    common = {"trust_remote_code": True}

    if load_in_4bit and load_in_8bit:
        load_in_8bit = False

    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig

        if load_in_4bit:
            qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            qconf = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=qconf,
            device_map="auto",
            **common,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
            **common,
        )
        if not use_cuda:
            model = model.to(torch.device("cpu"))

    model.eval()
    return model, tokenizer


_REDACTED_THINK_BLOCK = re.compile(
    r"<redacted_thinking>[\s\S]*?</redacted_thinking>", re.IGNORECASE
)


def _sanitize_model_reply(text: str) -> str:
    """去掉思考块与常见 think 标签，与 Medical_Qwen 评测侧清洗思路一致。"""
    if not text:
        return ""
    t = text.strip()
    t = _REDACTED_THINK_BLOCK.sub("", t)
    t = re.sub(r"</?think>", "", t, flags=re.IGNORECASE)
    return t.strip()


def _apply_chat_template_str(
    tokenizer,
    messages: list,
    *,
    disable_thinking: bool = True,
) -> str:
    """与 Medical_Qwen/evaluate_sft_qwen.py 中 build_prompt 对齐：关思考并必要时注入空 think 块。"""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not disable_thinking,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if disable_thinking:
            if prompt.endswith("<|im_start|>assistant\n<redacted_thinking>\n"):
                prompt += "\n</redacted_thinking>\n\n"
            elif prompt.endswith("<|im_start|>assistant\n"):
                prompt += "<redacted_thinking>\n\n</redacted_thinking>\n\n"
        return prompt


def _build_prompts(
    tokenizer,
    user_messages: List[str],
    system_text: str,
    *,
    disable_thinking: bool = True,
) -> List[str]:
    prompts: List[str] = []
    for user_content in user_messages:
        msgs = []
        if system_text:
            msgs.append({"role": "system", "content": system_text})
        msgs.append({"role": "user", "content": user_content})
        prompts.append(
            _apply_chat_template_str(tokenizer, msgs, disable_thinking=disable_thinking)
        )
    return prompts


@torch.inference_mode()
def batch_chat_complete(
    model,
    tokenizer,
    user_messages: List[str],
    system_text: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    repetition_penalty: float = 1.0,
    stop_str: str = "</s>",
    disable_thinking: bool = True,
) -> List[str]:
    """对多条 user 文本批量生成回复（左填充下按 attention_mask 截去 prompt）。"""
    device = next(model.parameters()).device
    prompts = _build_prompts(
        tokenizer,
        user_messages,
        system_text,
        disable_thinking=disable_thinking,
    )
    batch = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    gen_kw = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        repetition_penalty=repetition_penalty,
    )
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kw,
    )
    prompt_lens = attention_mask.sum(dim=1).tolist()
    texts: List[str] = []
    for i in range(outputs.size(0)):
        pl = int(prompt_lens[i])
        gen_ids = outputs[i, pl:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pos = gen_text.find(stop_str)
        if pos != -1:
            gen_text = gen_text[:pos]
        texts.append(_sanitize_model_reply(gen_text))
    return texts


@torch.inference_mode()
def chat_complete(
    model,
    tokenizer,
    user: str,
    system_text: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    repetition_penalty: float = 1.0,
    stop_str: str = "</s>",
    disable_thinking: bool = True,
) -> str:
    return batch_chat_complete(
        model,
        tokenizer,
        [user],
        system_text=system_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        stop_str=stop_str,
        disable_thinking=disable_thinking,
    )[0]
