"""Dataset curation and replay-buffer merging utilities for incremental YOLO training."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict

import yaml

from app.src.config import VALID_IMAGE_EXTS
from app.src.utils.logging_utils import get_logger

logger = get_logger("DatasetBuilder")


def clean_orphan_labels(images_dir: Path, labels_dir: Path) -> int:
    """Delete any .txt label file that no longer has a matching image."""
    valid_stems = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS}
    removed = 0
    for lbl in list(labels_dir.glob("*.txt")):
        if lbl.stem not in valid_stems:
            lbl.unlink()
            removed += 1
    logger.info("Cleaned %d orphan .txt files (kept %d valid pairs).", removed, len(valid_stems))
    return removed


def copy_pair(
    img_path: Path,
    src_labels_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    prefix: str = "",
) -> None:
    """Copy an image and its corresponding YOLO label file (creating an empty label if missing)."""
    stem = f"{prefix}{img_path.stem}"
    shutil.copy2(img_path, dst_images_dir / f"{stem}{img_path.suffix}")
    src_lbl = src_labels_dir / f"{img_path.stem}.txt"
    dst_lbl = dst_labels_dir / f"{stem}.txt"
    if src_lbl.exists():
        shutil.copy2(src_lbl, dst_lbl)
    else:
        dst_lbl.write_text("", encoding="utf-8")


def build_merged_dataset(
    orig_root: Path,
    qwen_root: Path,
    out_root: Path,
    replay_samples: int = 400,
    seed: int = 42,
    yaml_filename: str = "data_merged.yaml",
) -> Path:
    """Merge cleaned Qwen-distilled samples with an original training replay buffer and val/test splits."""
    if out_root.exists():
        shutil.rmtree(out_root)

    train_img_dst = out_root / "train" / "images"
    train_lbl_dst = out_root / "train" / "labels"
    train_img_dst.mkdir(parents=True, exist_ok=True)
    train_lbl_dst.mkdir(parents=True, exist_ok=True)

    qwen_img_src = qwen_root / "images"
    qwen_lbl_src = qwen_root / "labels"
    clean_orphan_labels(qwen_img_src, qwen_lbl_src)

    qwen_imgs = sorted(p for p in qwen_img_src.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS)
    qwen_neg, qwen_pos = 0, 0
    for img_p in qwen_imgs:
        copy_pair(img_p, qwen_lbl_src, train_img_dst, train_lbl_dst, prefix="qwen_")
        lbl_path = qwen_lbl_src / f"{img_p.stem}.txt"
        lbl_txt = lbl_path.read_text(encoding="utf-8").strip() if lbl_path.exists() else ""
        if lbl_txt:
            qwen_pos += 1
        else:
            qwen_neg += 1

    orig_train_imgs = sorted(
        p for p in (orig_root / "train" / "images").iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS
    )
    rng = random.Random(seed)
    rng.shuffle(orig_train_imgs)
    selected_replay = (
        orig_train_imgs
        if (replay_samples <= 0 or replay_samples >= len(orig_train_imgs))
        else orig_train_imgs[:replay_samples]
    )

    for img_p in selected_replay:
        copy_pair(img_p, orig_root / "train" / "labels", train_img_dst, train_lbl_dst, prefix="orig_")

    for split in ("val", "test"):
        src_split = orig_root / split
        if src_split.exists():
            shutil.copytree(src_split, out_root / split)

    names: Dict[int, str] = {0: "Arabian-gazelle", 1: "Arabian-oryx"}
    orig_yaml = orig_root / "data.yaml"
    if orig_yaml.exists():
        cfg = yaml.safe_load(orig_yaml.read_text(encoding="utf-8"))
        if isinstance(cfg, dict) and "names" in cfg:
            names = cfg["names"]

    yaml_data = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": names,
    }
    yaml_path = out_root / yaml_filename
    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")

    logger.info(
        "Built %s -> Train: %d total (%d Qwen Hard Pos + %d Qwen Hard Neg + %d Original Replay)",
        out_root,
        len(qwen_imgs) + len(selected_replay),
        qwen_pos,
        qwen_neg,
        len(selected_replay),
    )
    return yaml_path
