"""
在 SFT 保存 checkpoint 后周期性运行 scripts/eval_medical_bleu_rouge.sh，
将结果写入 output_dir/validation，追加 CSV，并仅保留指标最优的 N 次完整产物目录。
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_main_process(args: TrainingArguments) -> bool:
    lr = getattr(args, "local_rank", -1)
    if lr is None:
        lr = -1
    try:
        lr = int(lr)
    except (TypeError, ValueError):
        lr = -1
    return lr in (-1, 0)


def _parse_metrics_from_json(path: Path) -> tuple[dict[str, float], dict[str, float], int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mb = data.get("metrics_base") or {}
    mf = data.get("metrics_finetuned") or {}
    n = int(data.get("num_samples", 0))
    return mb, mf, n


def _flatten_metrics_row(
    global_step: int,
    checkpoint: str,
    mb: dict[str, float],
    mf: dict[str, float],
    num_samples: int,
    ts: str,
    eval_seconds: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp": ts,
        "global_step": global_step,
        "checkpoint_path": checkpoint,
        "num_samples": num_samples,
        "eval_seconds": eval_seconds,
    }
    for prefix, m in (("base", mb), ("finetuned", mf)):
        for k in ("bleu1", "bleu2", "bleu3", "bleu4", "rougeL"):
            row[f"{k}_{prefix}"] = float(m.get(k, 0.0))
    return row


def _csv_fieldnames() -> list[str]:
    return [
        "timestamp",
        "global_step",
        "checkpoint_path",
        "num_samples",
        "eval_seconds",
        "bleu1_base",
        "bleu2_base",
        "bleu3_base",
        "bleu4_base",
        "rougeL_base",
        "bleu1_finetuned",
        "bleu2_finetuned",
        "bleu3_finetuned",
        "bleu4_finetuned",
        "rougeL_finetuned",
    ]


def _append_csv(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    names = _csv_fieldnames()
    new_file = not csv_path.is_file()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in names})


def _read_csv_scores(csv_path: Path, sort_key: str) -> list[tuple[int, float]]:
    """返回 (global_step, primary_metric) 列表。"""
    if not csv_path.is_file():
        return []
    rows: list[tuple[int, float]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for line in r:
            try:
                step = int(line["global_step"])
                score = float(line[sort_key])
                rows.append((step, score))
            except (KeyError, ValueError):
                continue
    return rows


def _plot_metrics(mb: dict[str, float], mf: dict[str, float], out_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[MedicalBleuRougeEvalCallback] 未安装 matplotlib，跳过图像保存。可: pip install matplotlib",
            flush=True,
        )
        return

    labels = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L"]
    x = range(len(labels))
    base_vals = [float(mb.get(f"bleu{i}", 0.0)) for i in range(1, 5)] + [float(mb.get("rougeL", 0.0))]
    ft_vals = [float(mf.get(f"bleu{i}", 0.0)) for i in range(1, 5)] + [float(mf.get("rougeL", 0.0))]

    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w / 2 for i in x], base_vals, width=w, label="base")
    ax.bar([i + w / 2 for i in x], ft_vals, width=w, label="finetuned")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("score")
    ax.set_title("Medical eval: BLEU / ROUGE-L (macro)")
    ax.legend()
    ax.set_ylim(0, max(1.0, max(base_vals + ft_vals) * 1.1))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _prune_run_dirs(runs_root: Path, csv_path: Path, keep_best_n: int, sort_key: str) -> None:
    if keep_best_n <= 0:
        return
    scores = _read_csv_scores(csv_path, sort_key)
    if not scores:
        return
    # 按分数降序，同分按 step 大者优先
    ranked = sorted(scores, key=lambda t: (t[1], t[0]), reverse=True)
    keep_steps = {s for s, _ in ranked[:keep_best_n]}
    if not runs_root.is_dir():
        return
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        m = re.match(r"^step_(\d+)$", child.name)
        if not m:
            continue
        step = int(m.group(1))
        if step not in keep_steps:
            shutil.rmtree(child, ignore_errors=True)


class MedicalBleuRougeEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        eval_script: str,
        validation_root: str,
        base_model: str,
        data_path: str,
        image_folder: str,
        eval_every_steps: int,
        keep_best_n: int = 3,
        eval_batch_size: int = 8,
        max_new_tokens: int = 1024,
        cuda_visible_devices: Optional[str] = None,
        sort_key: str = "rougeL_finetuned",
    ):
        self.eval_script = Path(eval_script).resolve()
        self.validation_root = Path(validation_root).resolve()
        self.base_model = base_model
        self.data_path = data_path
        self.image_folder = image_folder
        self.eval_every_steps = int(eval_every_steps)
        self.keep_best_n = int(keep_best_n)
        self.eval_batch_size = int(eval_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.cuda_visible_devices = cuda_visible_devices
        self.sort_key = sort_key

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if self.eval_every_steps <= 0:
            return control
        if not _is_main_process(args):
            return control
        step = int(state.global_step)
        if step % self.eval_every_steps != 0:
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
        if not checkpoint_dir.is_dir():
            print(
                f"[MedicalBleuRougeEvalCallback] 未找到 checkpoint 目录: {checkpoint_dir}，跳过评测",
                flush=True,
            )
            return control

        repo = _repo_root()
        runs_root = self.validation_root / "runs"
        run_dir = runs_root / f"step_{step}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out_json = run_dir / "medical_eval_bleu_rouge.json"
        csv_path = self.validation_root / "metrics_history.csv"
        plot_path = run_dir / "metrics_plot.png"

        env = os.environ.copy()
        env["MODEL_DIR"] = self.base_model
        env["FINETUNED_DIR"] = str(checkpoint_dir)
        env["DATA_JSON"] = self.data_path
        env["IMAGE_FOLDER"] = self.image_folder
        env["OUT_JSON"] = str(out_json)
        env["EVAL_BATCH_SIZE"] = str(self.eval_batch_size)
        env["MAX_NEW_TOKENS"] = str(self.max_new_tokens)
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices

        cmd = ["bash", str(self.eval_script)]
        print(
            f"[MedicalBleuRougeEvalCallback] step={step} 启动 BLEU/ROUGE 评测 …\n  FINETUNED_DIR={checkpoint_dir}",
            flush=True,
        )
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo),
                env=env,
                check=False,
                capture_output=False,
                text=True,
            )
        except Exception as e:
            print(f"[MedicalBleuRougeEvalCallback] 子进程异常: {e}", flush=True)
            return control

        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(
                f"[MedicalBleuRougeEvalCallback] 评测失败 (exit={proc.returncode})，耗时 {elapsed:.1f}s",
                flush=True,
            )
            return control

        if not out_json.is_file():
            print(f"[MedicalBleuRougeEvalCallback] 未生成 {out_json}", flush=True)
            return control

        mb, mf, num_samples = _parse_metrics_from_json(out_json)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        row = _flatten_metrics_row(
            step, str(checkpoint_dir), mb, mf, num_samples, ts, round(elapsed, 2)
        )
        _append_csv(csv_path, row)
        _plot_metrics(mb, mf, plot_path)

        meta = {
            "global_step": step,
            "checkpoint_path": str(checkpoint_dir),
            "metrics_base": mb,
            "metrics_finetuned": mf,
            "num_samples": num_samples,
            "eval_seconds": round(elapsed, 2),
            "timestamp": ts,
        }
        with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        _prune_run_dirs(runs_root, csv_path, self.keep_best_n, self.sort_key)
        print(
            f"[MedicalBleuRougeEvalCallback] step={step} 评测完成，ROUGE-L(finetuned)={mf.get('rougeL', 0):.4f}，"
            f"保留最优 {self.keep_best_n} 个 runs/ 子目录",
            flush=True,
        )
        return control
