from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import random

import numpy as np
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
        mask = load_merged_mask(sample.mask_paths, image.size)

        image = TF.resize(
            image,
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
                mask = TF.hflip(mask)

            angle = random.uniform(-12.0, 12.0)
            image = TF.rotate(
                image,
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
                image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))
            if random.random() < 0.3:
                image = TF.adjust_brightness(image, random.uniform(0.90, 1.10))

        image_t = TF.to_tensor(image)  # [1,H,W], 0..1
        mask_t = (TF.to_tensor(mask) > 0.5).float()

        return {
            "image": image_t,
            "mask": mask_t,
            "image_path": str(sample.image_path),
        }
