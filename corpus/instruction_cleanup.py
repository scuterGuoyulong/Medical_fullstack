# -*- coding: utf-8 -*-
"""
问诊/问答数据里 instruction（问题）字段清洗：
  - 去掉常见平台模板、重复啰嗦、尾缀「无」等噪声；
  - 可选：本地 Qwen 再提炼为核心问题（延迟 import transformers）。

典型噪声：「曾经的治疗情况和效果：无」「希望得到怎样的帮助：」「在乎怎样的协助：」
以及同一句话复制两遍、问题末尾「，无」等。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# ---------- 规则清洗 ----------

# 从当前位置删到行尾或全文末尾（模板字段常占一段）
_BOILERPLATE_TAIL_PATTERNS = [
    r"曾经的治疗情况和效果\s*[：:].*",
    r"既往治疗(情况)?\s*[：:].*",
    r"治疗情况\s*[：:].*",
    r"想得到什么帮助\s*[：:].*",
    r"希望得到怎样的帮助\s*[：:].*",
    r"希望得到什么样的帮助\s*[：:].*",
    r"想得到怎样的帮助\s*[：:].*",
    r"想得到什么样的帮助\s*[：:].*",
    r"在乎怎样的协助\s*[：:].*",  # 常见错别字模板
    r"想得到怎样协助\s*[：:].*",
    r"希望获得什么帮助\s*[：:].*",
    r"想要咨询的问题\s*[：:].*",
    r"主要想咨询\s*[：:].*",
    r"咨询标题\s*[：:].*",
    r"疾病名称\s*[：:].*",
    r"病情描述\s*[：:]\s*",  # 仅去掉标签，后面常跟正文，用非贪婪要小心
    r"患者\s*[：:]\s*",
    r"性别\s*[：:].*?岁",  # 性别：男 年龄：35岁 类——整段删风险大，改为弱规则
]

# 整段匹配则整段删（整行都是模板）
_FULL_LINE_JUNK = re.compile(
    r"^\s*(曾经的治疗情况和效果|治疗情况|想得到什么帮助|希望得到怎样的帮助)\s*[：:]\s*无?\s*$",
    re.I,
)

# 问题里不应出现的长篇「回答腔」开头（删从此处到结尾）
_ANSWER_LIKE_PREFIX = re.compile(
    r"(您好[,，]|感谢您的信任|根据您的描述|建议如下|治疗方案包括|综上所述)[\s\S]*$"
)

_TRAILING_JUNK = re.compile(
    r"(?:[，,。．、]\s*无\s*|[，,]\s*没有\s*|[，,]\s*暂无\s*|[，,]\s*无\s*)$"
)


def _collapse_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t\u3000]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _strip_boilerplate_tails(s: str) -> str:
    t = s
    for pat in _BOILERPLATE_TAIL_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE | re.DOTALL)
    return t


def _split_sentences(s: str) -> List[str]:
    """按中英文句号、问号、换行切句，保留非空片段。"""
    parts = re.split(r"(?<=[。！？!?])\s*|\n+", s)
    out: List[str] = []
    for p in parts:
        x = p.strip()
        if len(x) >= 4:
            out.append(x)
    return out if out else ([s.strip()] if s.strip() else [])


def _sentence_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 12 and len(b) >= 12:
        if a in b or b in a:
            return 0.92
    return SequenceMatcher(None, a, b).ratio()


def _dedupe_sentences(sents: List[str], *, threshold: float = 0.86) -> List[str]:
    kept: List[str] = []
    for sent in sents:
        if _FULL_LINE_JUNK.match(sent):
            continue
        dup = False
        for prev in kept:
            if _sentence_similarity(sent, prev) >= threshold:
                dup = True
                break
        if not dup:
            kept.append(sent)
    return kept


def _clause_core(t: str) -> str:
    t = re.sub(r"^[您好]+[,，]?\s*", "", (t or "").strip())
    return re.sub(r"[？?！!。．]+$", "", t).strip()


def _dedupe_comma_in_segment(segment: str) -> str:
    """单个小句段内「，」连接的近重复（不含跨问号合并）。"""
    if "，" not in segment and "," not in segment:
        return segment
    sep = "，" if "，" in segment else ","
    raw_chunks = [x.strip() for x in segment.split(sep) if x.strip()]
    if len(raw_chunks) < 2:
        return segment
    result: List[str] = [raw_chunks[0]]
    for ch in raw_chunks[1:]:
        core = _clause_core(ch)
        if not core:
            continue
        prev = result[-1]
        ec = _clause_core(prev)
        if not ec:
            result.append(ch)
            continue
        same = core == ec
        sub = len(core) >= 8 and len(ec) >= 8 and (core in ec or ec in core)
        sim = SequenceMatcher(None, core, ec).ratio() >= 0.82
        if same or sub or sim:
            pick = ch if len(ch) > len(prev) else prev
            if ("？" in ch or "?" in ch) and not re.search(r"[？?]\s*$", pick):
                pick = pick.rstrip("。．") + "？"
            result[-1] = pick
        else:
            result.append(ch)
    return sep.join(result)


def _split_by_question_marks(s: str) -> List[str]:
    """按中文/英文问号切分为小句（保留句末问号）。"""
    s = s.strip()
    if not s:
        return []
    parts = re.split(r"(?<=[？?])\s*", s)
    return [p.strip() for p in parts if p.strip()]


def _dedupe_comma_phrases(text: str) -> str:
    """先按问号分段，再对每段做逗号近重复合并。"""
    segs = _split_by_question_marks(text)
    if not segs:
        return text
    out = [_dedupe_comma_in_segment(seg) for seg in segs]
    return "".join(out) if len(out) > 1 else out[0]


def _fix_truncated_pattern(s: str) -> str:
    """如「对于阳痿是什么病，无」→「阳痿是什么病？」类简化。"""
    t = s
    t = _TRAILING_JUNK.sub("", t)
    # 「对于X是什么病」且无后续 → 保留「X是什么病」
    m = re.match(
        r"^对于(.{2,40}?)是什么病[，,]?\s*无?\s*$",
        t,
    )
    if m:
        core = m.group(1).strip()
        return f"{core}是什么病？"
    return t


def clean_instruction_heuristic(raw: str) -> Tuple[str, bool]:
    """
    规则清洗 instruction。
    返回 (清洗后文本, 是否与原文不同)。
    """
    if not raw or not str(raw).strip():
        return "", False

    original = str(raw).strip()
    t = _collapse_ws(original)
    t = _strip_boilerplate_tails(t)
    t = _collapse_ws(t)
    t = _fix_truncated_pattern(t)
    t = _TRAILING_JUNK.sub("", t).strip()

    # 去掉回答腔尾巴
    t = _ANSWER_LIKE_PREFIX.sub("", t).strip()

    # 患者叙述里误夹的「您好这…」（非医生回复）
    t = re.sub(r"您好[,，]?\s*这", "这", t)

    # 句内逗号分句近重复（问诊平台常见）
    t = _dedupe_comma_phrases(t)
    t = _collapse_ws(t)

    sents = _split_sentences(t)
    sents = _dedupe_sentences(sents)
    t = " ".join(sents) if sents else t
    t = re.sub(r"\s+([，。！？])", r"\1", t)
    t = re.sub(r"([，。！？])\s*", r"\1", t)
    t = _collapse_ws(t)

    # 句末标点：若完全没有问号但像疑问句，不强行加（避免误伤）
    if len(t) < 4:
        return original, False

    changed = t != original
    return t, changed


# ---------- 可选 LLM 精炼 ----------
#
# 「平台模板」指问诊网站/APP 在表单里自动带上的固定栏目文案，不是患者本人病情描述，例如：
# 「曾经的治疗情况和效果：无」「希望得到怎样的帮助：」「在乎怎样的协助：」「病情描述：」等。
# 模型清洗时应删掉这些栏目及其占位内容，但保留患者自述的症状、时间线、顾虑等真正与就医相关的话。

_LLM_SYS = (
    "你是医学问诊文本编辑。用户输入多来自在线问诊平台的复制粘贴，除病情叙述外，常夹杂"
    "「平台模板句」——即网站表单的固定栏目标签及填空（如：曾经的治疗情况和效果、希望得到怎样的帮助、"
    "在乎怎样的协助、病情描述、咨询标题等后面的占位或「无」），以及整段重复拷贝、明显属于医生回复开头的寒暄。"
    "你的任务：输出**清洗后的患者问题文本**，只保留与本次就医咨询直接相关的信息（症状、部位、时长、"
    "诱因、已做检查、具体顾虑、想问医生的点等）；删除上述模板句、重复句、无关客套与明显不属于患者提问的内容。"
    "不要刻意把内容压成一两句话：若患者叙述较长但都与问题相关，应完整保留，仅去掉噪声。"
    "禁止编造或补充原文没有的症状与诊断。不要输出「问题：」「清洗后：」等标签或解释。"
    "只输出清洗后的正文，语序可轻微理顺，标点可规范。"
)


def clean_instruction_llm(
    raw: str,
    model,
    tokenizer,
    *,
    max_new_tokens: int = 768,
    temperature: float = 0.15,
    disable_thinking: bool = True,
) -> str:
    from local_qwen_client import chat_complete

    raw = (raw or "").strip()
    if not raw:
        return ""
    user = f"【患者粘贴原文】\n{raw}\n\n请输出清洗后的患者问题正文（仅正文）："
    out = chat_complete(
        model,
        tokenizer,
        user,
        system_text=_LLM_SYS,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        disable_thinking=disable_thinking,
    )
    return out.strip()


def process_jsonl_row(
    row: Dict[str, Any],
    question_key: str,
    *,
    mode: str,
    model=None,
    tokenizer=None,
    llm_max_new_tokens: int = 768,
    llm_temperature: float = 0.15,
    disable_thinking: bool = True,
) -> Dict[str, Any]:
    """
    mode: none | heuristic | llm（llm 前会先 heuristic）
    """
    q = str(row.get(question_key) or "")
    if mode == "none":
        row["_instruction_cleaned"] = False
        row["_instruction_clean_mode"] = "none"
        return row

    cleaned, changed = clean_instruction_heuristic(q)
    row[question_key] = cleaned
    row["_instruction_heuristic_changed"] = changed

    if mode == "heuristic":
        row["_instruction_cleaned"] = changed
        row["_instruction_clean_mode"] = "heuristic"
        return row

    if mode == "llm":
        if model is None or tokenizer is None:
            raise ValueError("mode=llm 需要 model 与 tokenizer")
        if not cleaned.strip():
            row["_instruction_cleaned"] = changed
            row["_instruction_clean_mode"] = "heuristic_only_empty"
            return row
        refined = clean_instruction_llm(
            cleaned,
            model,
            tokenizer,
            max_new_tokens=llm_max_new_tokens,
            temperature=llm_temperature,
            disable_thinking=disable_thinking,
        )
        if refined:
            row[question_key] = refined
        row["_instruction_cleaned"] = True
        row["_instruction_clean_mode"] = "llm"
        return row

    raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="JSONL：清洗 instruction 字段")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--question_key", default="instruction")
    ap.add_argument(
        "--mode",
        choices=("none", "heuristic", "llm"),
        default="heuristic",
        help="heuristic 仅规则；llm 在规则后再调用本地模型提炼",
    )
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--tokenizer_name_or_path", default=None)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=768,
        help="LLM 清洗 instruction 时的最大生成长度（长叙述勿过小）",
    )
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument(
        "--disable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = ap.parse_args()

    model = tok = None
    if args.mode == "llm":
        if not args.model_name_or_path:
            raise SystemExit("--mode llm 需要 --model_name_or_path")
        from local_qwen_client import load_model_tokenizer

        model, tok = load_model_tokenizer(
            args.model_name_or_path,
            tokenizer_path=args.tokenizer_name_or_path,
            load_in_4bit=args.load_in_4bit,
        )

    n = 0
    n_changed = 0
    with open(args.input, "r", encoding="utf-8") as inf, open(
        args.output, "w", encoding="utf-8"
    ) as out:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            before = str(row.get(args.question_key) or "")
            row = process_jsonl_row(
                row,
                args.question_key,
                mode=args.mode,
                model=model,
                tokenizer=tok,
                llm_max_new_tokens=args.max_new_tokens,
                llm_temperature=args.temperature,
                disable_thinking=args.disable_thinking,
            )
            after = str(row.get(args.question_key) or "")
            if before != after:
                n_changed += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "rows": n,
                "instruction_changed": n_changed,
                "mode": args.mode,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
