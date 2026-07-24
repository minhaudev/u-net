from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import random

import numpy as np
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_paths: tuple[Path, ...]


def _is_mask(path: Path) -> bool:
    return "_mask" in path.stem.lower()


def discover_busi_samples(data_dir: str | Path, include_normal: bool = False) -> list[Sample]:
    """
    Tìm các cặp ảnh-mask theo cấu trúc BUSI:
      Dataset_BUSI_with_GT/
        benign/
        malignant/
        normal/

    Hỗ trợ một ảnh có nhiều mask:
      benign (1)_mask.png
      benign (1)_mask_1.png
      ...
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_dir}")

    allowed_folders = {"benign", "malignant"}
    if include_normal:
        allowed_folders.add("normal")

    samples: list[Sample] = []

    for folder in sorted(data_dir.rglob("*")):
        if not folder.is_dir() or folder.name.lower() not in allowed_folders:
            continue

        image_files = sorted(
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in VALID_EXTENSIONS
            and not _is_mask(p)
        )

        for image_path in image_files:
            prefix = image_path.stem + "_mask"
            mask_paths = tuple(sorted(
                p for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in VALID_EXTENSIONS
                and p.stem.startswith(prefix)
            ))

            # Ảnh normal có thể không có mask; khi đó dùng mask rỗng.
            if mask_paths or folder.name.lower() == "normal":
                samples.append(Sample(image_path=image_path, mask_paths=mask_paths))

    if not samples:
        raise RuntimeError(
            "Không tìm thấy cặp ảnh-mask. Hãy kiểm tra đường dẫn phải trỏ tới "
            "thư mục chứa các folder benign, malignant và normal."
        )

    return samples


def discover_flat_mask_samples(data_dir: str | Path) -> list[Sample]:
    """
    Find image-mask pairs in a folder recursively.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc du lieu: {data_dir}")

    # Dùng rglob để quét tất cả ảnh kể cả khi nó nằm trong thư mục con (vd: train/benign/...)
    image_files = sorted(
        p for p in data_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VALID_EXTENSIONS
        and not _is_mask(p)
    )

    samples: list[Sample] = []
    missing_masks: list[Path] = []

    for image_path in image_files:
        prefix = image_path.stem + "_mask"
        # Tìm mask nằm cùng thư mục với ảnh gốc
        mask_paths = tuple(sorted(
            p for p in image_path.parent.iterdir()
            if p.is_file()
            and p.suffix.lower() in VALID_EXTENSIONS
            and p.stem.startswith(prefix)
        ))

        # Ảnh normal có thể không có mask, ta kiểm tra nếu parent name là normal
        if mask_paths or image_path.parent.name.lower() == "normal":
            samples.append(Sample(image_path=image_path, mask_paths=mask_paths))
        else:
            missing_masks.append(image_path)

    if missing_masks:
        examples = ", ".join(p.name for p in missing_masks[:5])
        print(f"Cảnh báo: Có {len(missing_masks)} ảnh không có mask trong {data_dir}. Ví dụ: {examples}")

    if not samples:
        raise RuntimeError(
            f"Không tìm thấy cặp ảnh-mask hợp lệ nào trong {data_dir}."
        )

    return samples


def discover_predefined_split_samples(
    data_dir: str | Path,
) -> dict[str, list[Sample]]:
    """
    Load datasets that already contain train/valid/test folders.
    Tự động tìm thư mục train/valid kể cả khi bị lồng 1 cấp thư mục.
    """
    data_dir = Path(data_dir)
    
    # Tìm nhanh thư mục train bên trong data_dir
    train_dir = None
    for p in data_dir.rglob("train"):
        if p.is_dir():
            train_dir = p
            break
            
    if not train_dir:
        return {}
        
    # Từ thư mục cha của train, tìm valid và test
    parent_dir = train_dir.parent
    valid_dir = parent_dir / "valid"
    if not valid_dir.exists():
        valid_dir = parent_dir / "val"
    test_dir = parent_dir / "test"

    if not valid_dir.exists():
        return {}

    splits = {
        "train": discover_flat_mask_samples(train_dir),
        "valid": discover_flat_mask_samples(valid_dir),
    }
    if test_dir.exists():
        splits["test"] = discover_flat_mask_samples(test_dir)

    return splits


def load_merged_mask(mask_paths: Iterable[Path], image_size: tuple[int, int]) -> Image.Image:
    mask_paths = list(mask_paths)
    if not mask_paths:
        return Image.new("L", image_size, color=0)

    merged = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    for mask_path in mask_paths:
        mask = Image.open(mask_path).convert("L")
        if mask.size != image_size:
            mask = mask.resize(image_size, resample=Image.Resampling.NEAREST)
        mask_np = np.asarray(mask)
        merged = np.maximum(merged, (mask_np > 127).astype(np.uint8) * 255)

    return Image.fromarray(merged, mode="L")


def apply_clahe(image: Image.Image) -> Image.Image:
    """
    Áp dụng thuật toán CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lên ảnh siêu âm gốc để làm nổi bật viền khối u.
    """
    img_np = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_np = clahe.apply(img_np)
    return Image.fromarray(clahe_np, mode="L")


class BUSIDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = Image.open(sample.image_path).convert("L")
        clahe_image = apply_clahe(image)
        mask = load_merged_mask(sample.mask_paths, image.size)

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        clahe_image = TF.resize(
            clahe_image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                clahe_image = TF.hflip(clahe_image)
                mask = TF.hflip(mask)

            angle = random.uniform(-12.0, 12.0)
            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            clahe_image = TF.rotate(
                clahe_image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

            if random.random() < 0.5:
                contrast_factor = random.uniform(0.85, 1.15)
                image = TF.adjust_contrast(image, contrast_factor)
                clahe_image = TF.adjust_contrast(clahe_image, contrast_factor)
            if random.random() < 0.3:
                brightness_factor = random.uniform(0.90, 1.10)
                image = TF.adjust_brightness(image, brightness_factor)
                clahe_image = TF.adjust_brightness(clahe_image, brightness_factor)

        image_t = TF.to_tensor(image)  # [1,H,W], 0..1
        clahe_t = TF.to_tensor(clahe_image)
        mask_t = (TF.to_tensor(mask) > 0.5).float()

        return {
            "image": image_t,
            "clahe": clahe_t,
            "mask": mask_t,
            "image_path": str(sample.image_path),
        }
