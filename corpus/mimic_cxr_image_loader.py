# -*- coding: utf-8 -*-
"""
从本地 load_from_disk 的 MIMIC-CXR 按 row_idx 读取图像（内存中 PIL），**不**再写一份 JPEG 目录。

用途：
- translate_reports 只译文字，本模块不参与；
- 若不想为 Qwen-VL 另存整盘图片，可在自定义 PyTorch Dataset 的 __getitem__ 里调用
  load_pil_image()，用 processor(images=pil, ...) 走训练流程（需自行接 Qwen-VL 训练脚本，
  官方默认的 JSON + image_folder 路径方案仍会写盘）。

注意：训练时仍会把当前 batch 的图载入显存/内存；省的是「重复 JPEG 文件」占用的磁盘，不是显存。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional, Union

try:
    from datasets import Dataset, DatasetDict
except ImportError:
    Dataset = Any  # type: ignore
    DatasetDict = Any  # type: ignore


@lru_cache(maxsize=4)
def _open_split(dataset_root: str, split: str) -> Any:
    from datasets import load_from_disk

    d = load_from_disk(dataset_root)
    if isinstance(d, DatasetDict):
        return d[split]
    return d


def load_pil_image(dataset_root: str, split: str, row_idx: int):
    """
    返回 datasets 解码后的图像（通常为 PIL.Image），不创建任何新文件。
    """
    ds = _open_split(dataset_root, split)
    row = ds[int(row_idx)]
    img = row.get("image")
    if img is None:
        raise KeyError(f"split={split} row_idx={row_idx} 无 image 列")
    return img


def load_findings_impression(
    dataset_root: str, split: str, row_idx: int
) -> tuple[str, str]:
    ds = _open_split(dataset_root, split)
    row = ds[int(row_idx)]
    f = str(row.get("findings") or "").strip()
    i = str(row.get("impression") or "").strip()
    return f, i
