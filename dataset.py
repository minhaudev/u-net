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
        
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        # Định nghĩa các phép augment mạnh cho dữ liệu Y tế
        if self.augment:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GridDistortion(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                ToTensorV2(),
            ], additional_targets={'clahe': 'image'})
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                ToTensorV2(),
            ], additional_targets={'clahe': 'image'})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = Image.open(sample.image_path).convert("L")
        clahe_image = apply_clahe(image)
        mask = load_merged_mask(sample.mask_paths, image.size)

        # Albumentations yêu cầu numpy array
        image_np = np.array(image)
        clahe_np = np.array(clahe_image)
        mask_np = np.array(mask)

        transformed = self.transform(image=image_np, clahe=clahe_np, mask=mask_np)
        
        # Albumentations trả về tensor. Ảnh xám (H, W) -> cần unsqueeze(0) thành (1, H, W)
        image_t = transformed['image'].float() / 255.0
        if image_t.ndim == 2:
            image_t = image_t.unsqueeze(0)
            
        clahe_t = transformed['clahe'].float() / 255.0
        if clahe_t.ndim == 2:
            clahe_t = clahe_t.unsqueeze(0)
            
        mask_t = transformed['mask'].float()
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)
        
        # Binary mask theo ngưỡng
        mask_t = (mask_t > 0).float()
        if mask_t.max() > 1.0: # Đề phòng mask 0-255
            mask_t = (mask_t > 0.5).float()

        return {
            "image": image_t,
            "clahe": clahe_t,
            "mask": mask_t,
            "image_path": str(sample.image_path),
        }

def discover_any_dataset(data_dir: str | Path) -> list[Sample]:
    """
    Hàm tự động phát hiện mọi cấu trúc dữ liệu (BUSI, BrEaST, BUS-UCLM, v.v.)
    """
    data_dir = Path(data_dir)
    samples: list[Sample] = []
    
    # 1. Kiểm tra cấu trúc BUS-UCLM (Có thư mục images/ và masks/ riêng, tên file giống nhau)
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    if images_dir.exists() and masks_dir.exists():
        for img_path in sorted(images_dir.glob("*.*")):
            if img_path.suffix.lower() not in VALID_EXTENSIONS: continue
            mask_path = masks_dir / img_path.name
            if mask_path.exists():
                samples.append(Sample(image_path=img_path, mask_paths=(mask_path,)))
        if samples: return samples
        
    # 2. Kiểm tra cấu trúc có train/valid/test
    splits = discover_predefined_split_samples(data_dir)
    if splits:
        for split in splits.values():
            samples.extend(split)
        if samples: return samples
        
    # 3. Kiểm tra cấu trúc BUSI (Thư mục benign, malignant, normal chứa chung ảnh và mask _mask)
    try:
        samples = discover_busi_samples(data_dir)
        if samples: return samples
    except Exception:
        pass
        
    # 4. Kiểm tra cấu trúc phẳng như BrEaST (Ảnh gốc và mask nằm chung thư mục nhưng mask có hậu tố _tumor, _other, _mask)
    mask_keywords = ["_mask", "_tumor", "_other", "_gt"]
    image_files = sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        and not any(k in p.stem.lower() for k in mask_keywords)
    )
    for img_path in image_files:
        # Tìm các file mask bắt đầu bằng tên ảnh gốc + "_"
        mask_paths = tuple(sorted(
            p for p in img_path.parent.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
            and p.stem.startswith(img_path.stem + "_")
            and any(k in p.stem.lower() for k in mask_keywords)
        ))
        if mask_paths:
            samples.append(Sample(image_path=img_path, mask_paths=mask_paths))
            
    if not samples:
        raise RuntimeError(f"Không thể tự động nhận diện cấu trúc dataset tại {data_dir}. Vui lòng kiểm tra lại data!")
        
    return samples
