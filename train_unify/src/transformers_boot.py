"""
在导入 Qwen3.5 等模型前调整 Transformers 的可选依赖探测。

部分环境安装了 `fla` 但与当前 `triton` 不兼容，会在 import 阶段崩溃；关闭 FLA 路径后使用标准 PyTorch 算子即可训练/推理。
"""

from __future__ import annotations

import os


def maybe_disable_flash_linear_attention() -> None:
    if (os.environ.get("TRANSFORMERS_USE_FLASH_LINEAR_ATTENTION") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    import transformers.utils.import_utils as import_utils

    import_utils.is_flash_linear_attention_available = lambda: False
