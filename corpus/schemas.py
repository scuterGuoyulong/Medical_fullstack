# -*- coding: utf-8 -*-
"""双场景语料结构化字段约定（JSONL 行级）。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class MedicalQARecord:
    """医疗问答：对齐 Medical_Qwen ShareGPT（human/gpt）前的中间结构。"""

    question: str
    answer: str
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_sharegpt(self) -> Dict[str, Any]:
        return {
            "conversations": [
                {"from": "human", "value": self.question.strip()},
                {"from": "gpt", "value": self.answer.strip()},
            ],
            "meta": {"source": self.source, **self.meta},
        }


@dataclass
class MedicalReportRecord:
    """医疗报告生成：可来自影像 findings/impression 或纯文本报告。"""

    instruction: str
    findings_en: Optional[str] = None
    impression_en: Optional[str] = None
    findings_zh: Optional[str] = None
    impression_zh: Optional[str] = None
    full_report_zh: Optional[str] = None
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_alpaca_sft(self) -> Dict[str, Any]:
        """用于报告生成 SFT：instruction + 输入（可为空）+ 中文报告正文。"""
        inp = ""
        out = self.full_report_zh or ""
        if self.findings_zh and self.impression_zh:
            out = f"Findings:\n{self.findings_zh}\n\nImpression:\n{self.impression_zh}"
        return {
            "instruction": self.instruction.strip(),
            "input": inp,
            "output": out.strip(),
            "meta": {"source": self.source, **self.meta},
        }


def dump_jsonl_row(obj: Dict[str, Any]) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def load_jsonl_line(line: str) -> Dict[str, Any]:
    import json

    return json.loads(line)
