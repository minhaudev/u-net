from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BUSIDataset, discover_busi_samples
from losses import BCEDiceLoss, dice_score_from_logits
from model import UNet, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net trên BUSI")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--include-normal", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_samples(samples: list[Any], val_ratio: float, seed: int):
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_count = max(1, int(len(indices) * val_ratio))
    val_indices = set(indices[:val_count])

    train_samples = [s for i, s in enumerate(samples) if i not in val_indices]
    val_samples = [s for i, s in enumerate(samples) if i in val_indices]
    return train_samples, val_samples


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_dice = 0.0
    total_items = 0

    progress = tqdm(loader, leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.size(0)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=autocast_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, masks)

        if is_train:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        dice = dice_score_from_logits(logits.detach(), masks)

        total_loss += loss.item() * batch_size
        total_dice += dice.item() * batch_size
        total_items += batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_items:.4f}",
            dice=f"{total_dice / total_items:.4f}",
        )

    return total_loss / total_items, total_dice / total_items


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    pin_memory = device.type == "cuda"

    all_samples = discover_busi_samples(
        args.data_dir,
        include_normal=args.include_normal,
    )
    train_samples, val_samples = split_samples(
        all_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_dataset = BUSIDataset(
        train_samples,
        image_size=args.image_size,
        augment=True,
    )
    val_dataset = BUSIDataset(
        val_samples,
        image_size=args.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)

    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda",
    )

    print("=" * 70)
    print(f"Thiết bị             : {device}")
    print(f"Tổng số ảnh          : {len(all_samples)}")
    print(f"Train / Validation   : {len(train_samples)} / {len(val_samples)}")
    print(f"Số tham số mô hình   : {count_parameters(model):,}")
    print(f"Kích thước ảnh       : {args.image_size} x {args.image_size}")
    print("=" * 70)

    history_path = output_dir / "history.csv"
    best_path = output_dir / "best_unet.pt"
    config_path = output_dir / "config.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    best_dice = -1.0
    epochs_without_improvement = 0

    with history_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_dice",
                "val_loss",
                "val_dice",
                "lr",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")

            train_loss, train_dice = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
            )

            with torch.no_grad():
                val_loss, val_dice = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                )

            scheduler.step(val_dice)
            current_lr = optimizer.param_groups[0]["lr"]

            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "lr": current_lr,
            })
            csv_file.flush()

            print(
                f"train_loss={train_loss:.4f} | "
                f"train_dice={train_dice:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_dice={val_dice:.4f}"
            )

            if val_dice > best_dice:
                best_dice = val_dice
                epochs_without_improvement = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "base_channels": args.base_channels,
                        "image_size": args.image_size,
                        "best_val_dice": best_dice,
                    },
                    best_path,
                )
                print(f"Đã lưu mô hình tốt nhất: {best_path}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print("Early stopping.")
                    break

    print("\nHoàn thành.")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Checkpoint: {best_path}")
    print(
        "\nLƯU Ý NGHIÊN CỨU: cách chia ngẫu nhiên trong file này chỉ dùng để "
        "test pipeline. Khi làm bài báo, hãy chia theo bệnh nhân hoặc dùng fold "
        "chính thức của BUS-BRA để tránh rò rỉ dữ liệu."
    )


if __name__ == "__main__":
    main()
