# -*- coding: utf-8 -*-
"""
从本地 MIMIC-CXR HuggingFace 磁盘数据集导出 findings + impression（英文），供 translate_reports.py。
优先 datasets.load_from_disk；若无 datasets，则用 pyarrow 读 train/data-*.arrow（与当前仓库结构一致）。

itsanmolgupta/mimic-cxr-dataset 等副本里 `image` 多为内嵌 JPEG 字节（struct.path 常为空），
若要得到可供 Qwen-VL `--image_folder` 使用的路径，必须指定 --save_images_dir，将按 row_idx 写出
`000000.jpg` 并在 JSONL 中写入相对文件名。

示例（导出 + 图像路径，再翻译）：
  python export_mimic_cxr_jsonl.py \\
    --dataset_root .../DataSets/mimic-cxr-dataset \\
    --output .../mimic_report_en.jsonl \\
    --save_images_dir .../mimic_cxr_jpeg \\
    --max_rows 100

  python translate_reports.py --input .../mimic_report_en.jsonl --output .../mimic_report_zh.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterator, List, Optional


def _norm_path_cell(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    s = str(v).strip()
    return s or None


def _save_image_to_dir(save_dir: str, row_idx: int, *, pil_img=None, image_bytes: Optional[bytes] = None) -> str:
    """写入 JPEG，返回相对文件名（仅文件名，不含目录）。"""
    os.makedirs(save_dir, exist_ok=True)
    fn = f"{int(row_idx):06d}.jpg"
    out_path = os.path.join(save_dir, fn)
    if pil_img is not None:
        pil_img.save(out_path, format="JPEG", quality=95)
    elif image_bytes:
        with open(out_path, "wb") as fp:
            fp.write(image_bytes)
    else:
        raise ValueError("需要 pil_img 或 image_bytes")
    return fn


def _image_rel_from_row(row: Dict[str, Any], image_column: Optional[str]) -> Optional[str]:
    """仅从字符串列取路径；datasets 中 `image` 常为 PIL，不能当路径。"""
    if image_column and image_column in row:
        v = row.get(image_column)
        if isinstance(v, str):
            return _norm_path_cell(v)
        return None
    for k in ("path", "image_path", "jpg_path", "file_path", "filepath"):
        if k in row and row.get(k) is not None:
            p = _norm_path_cell(row.get(k))
            if p:
                return p
    v = row.get("image")
    if isinstance(v, str):
        return _norm_path_cell(v)
    return None


def _export_via_datasets(
    root: str,
    split: str,
    max_rows: int,
    image_column: Optional[str] = None,
    save_images_dir: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    from datasets import load_from_disk

    ds_dict = load_from_disk(root)
    if split not in ds_dict:
        raise KeyError(f"split {split!r} not in {list(ds_dict.keys())}")
    split_ds = ds_dict[split]
    n = len(split_ds)
    limit = n if max_rows <= 0 else min(n, max_rows)
    for i in range(limit):
        row = split_ds[i]
        rec: Dict[str, Any] = {
            "row_idx": i,
            "findings": row.get("findings") or "",
            "impression": row.get("impression") or "",
        }
        img = _image_rel_from_row(row, image_column)
        if img:
            rec["image"] = img
        elif save_images_dir:
            raw = row.get("image")
            if hasattr(raw, "save"):
                rec["image"] = _save_image_to_dir(save_images_dir, i, pil_img=raw)
            elif isinstance(raw, dict) and raw.get("bytes"):
                rec["image"] = _save_image_to_dir(
                    save_images_dir, i, image_bytes=raw["bytes"]
                )
        yield rec


def _arrow_files_for_split(dataset_root: str, split: str) -> List[str]:
    split_dir = os.path.join(dataset_root, split)
    state_path = os.path.join(split_dir, "state.json")
    if not os.path.isfile(state_path):
        return sorted(
            f
            for f in os.listdir(split_dir)
            if f.startswith("data-") and f.endswith(".arrow")
        )
    with open(state_path, encoding="utf-8") as f:
        st = json.load(f)
    files = [x["filename"] for x in st.get("_data_files", [])]
    return [os.path.join(split_dir, fn) for fn in files]


def _export_via_pyarrow(
    dataset_root: str,
    split: str,
    max_rows: int,
    image_column: Optional[str] = None,
    save_images_dir: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    paths = _arrow_files_for_split(dataset_root, split)
    if not paths:
        raise FileNotFoundError(f"未找到 {split} 下的 arrow 分片: {dataset_root}")

    global_idx = 0
    for path in paths:
        with open(path, "rb") as fp:
            try:
                reader = ipc.open_file(fp)
                table = reader.read_all()
            except Exception:
                fp.seek(0)
                reader = ipc.open_stream(fp)
                table = reader.read_all()
        col_names = set(table.column_names)
        findings_col = table.column("findings").to_pylist()
        impression_col = table.column("impression").to_pylist()
        path_col = None
        path_keys = []
        if image_column and image_column in col_names:
            path_keys = [image_column]
        else:
            path_keys = ["path", "image_path", "jpg_path", "image", "file_path", "filepath"]
        for pk in path_keys:
            if pk not in col_names:
                continue
            col = table.column(pk)
            if pa.types.is_struct(col.type):
                continue
            if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
                path_col = col.to_pylist()
                break
        image_struct_col = None
        if "image" in col_names and pa.types.is_struct(table.column("image").type):
            image_struct_col = table.column("image")

        for j, (fi, im) in enumerate(zip(findings_col, impression_col)):
            f = str(fi).strip() if fi is not None else ""
            imp = str(im).strip() if im is not None else ""
            rid = global_idx
            global_idx += 1
            if not f and not imp:
                continue
            rec: Dict[str, Any] = {"row_idx": rid, "findings": f, "impression": imp}
            if path_col is not None:
                p = _norm_path_cell(path_col[j])
                if p:
                    rec["image"] = p
            elif save_images_dir and image_struct_col is not None:
                st = image_struct_col[j].as_py()
                if isinstance(st, dict):
                    pth = _norm_path_cell(st.get("path"))
                    blob = st.get("bytes")
                    if pth:
                        rec["image"] = pth
                    elif blob:
                        rec["image"] = _save_image_to_dir(
                            save_images_dir, rid, image_bytes=blob
                        )
            yield rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_rows", type=int, default=0, help=">0 时只导出前 N 条非空报告（跨分片累计）")
    ap.add_argument("--prefer_pyarrow", action="store_true", help="跳过 datasets，直接用 pyarrow")
    ap.add_argument(
        "--image_column",
        default="",
        help="若 HF 行中图像路径列名非 path/image_path 等，可显式指定（datasets 模式）",
    )
    ap.add_argument(
        "--save_images_dir",
        default="",
        help="将内嵌 JPEG 解码写入该目录，JSONL 中 image 为相对文件名（如 000042.jpg）；"
        "Qwen-VL 训练时 --image_folder 指向此目录",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    icol = args.image_column.strip() or None
    img_dir = args.save_images_dir.strip() or None
    if img_dir:
        os.makedirs(img_dir, exist_ok=True)

    if args.prefer_pyarrow:
        rows_iter = _export_via_pyarrow(
            args.dataset_root, args.split, args.max_rows, icol, img_dir
        )
    else:
        try:
            rows_iter = _export_via_datasets(
                args.dataset_root, args.split, args.max_rows, icol, img_dir
            )
        except ImportError:
            rows_iter = _export_via_pyarrow(
                args.dataset_root, args.split, args.max_rows, icol, img_dir
            )

    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for rec in rows_iter:
            fi = rec.get("findings") or ""
            im = rec.get("impression") or ""
            if isinstance(fi, bytes):
                fi = fi.decode("utf-8", errors="replace")
            if isinstance(im, bytes):
                im = im.decode("utf-8", errors="replace")
            fi, im = str(fi).strip(), str(im).strip()
            if not fi and not im:
                continue
            obj: Dict[str, Any] = {
                "row_idx": rec.get("row_idx", written),
                "findings": fi,
                "impression": im,
            }
            if rec.get("image"):
                obj["image"] = rec["image"]
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
            if args.max_rows > 0 and written >= args.max_rows:
                break

    print(
        json.dumps(
            {"output": args.output, "written_lines": written, "max_rows_cap": args.max_rows},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
