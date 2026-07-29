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

from console_utils import configure_utf8_output
from dataset import (
    BUSIDataset,
    discover_busi_samples,
    discover_predefined_split_samples,
)
from losses import dice_score_from_logits, FocalTverskyLoss, DeepSupervisionLoss
from model import UNet, AttentionUNetFusion, CA_UNet, ECA_UNet, SimAM_UNet, EMA_UNet, Ghost_UNet, SMP_UNet, SMP_MobileNet_Lite, SMP_MobileNetV3_Micro, count_parameters


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
    parser.add_argument("--input-mode", type=str, default="gating", choices=["gating", "orig", "clahe", "fusion", "ca_unet", "eca_unet", "simam_unet", "ema_unet", "ghost_unet", "smp_mobilenet", "smp_lite", "smp_micro"], help="Chọn nhánh để test ablation")
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


def load_data_splits(args: argparse.Namespace):
    predefined_splits = discover_predefined_split_samples(args.data_dir)
    if predefined_splits:
        return (
            predefined_splits["train"],
            predefined_splits["valid"],
            predefined_splits.get("test", []),
            True,
        )

    all_samples = discover_busi_samples(
        args.data_dir,
        include_normal=args.include_normal,
    )
    train_samples, val_samples = split_samples(
        all_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    return train_samples, val_samples, [], False


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
        clahe_images = batch["clahe"].to(device, non_blocking=True)
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
            logits = model(images, clahe_images)
            
            if isinstance(logits, list):
                loss = criterion(logits, masks)
                logits_eval = logits[0]
            else:
                loss = criterion([logits], masks)
                logits_eval = logits

        if is_train:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        dice = dice_score_from_logits(logits_eval.detach(), masks)

        total_loss += loss.item() * batch_size
        total_dice += dice.item() * batch_size
        total_items += batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_items:.4f}",
            dice=f"{total_dice / total_items:.4f}",
        )

    return total_loss / total_items, total_dice / total_items


def main() -> None:
    configure_utf8_output()

    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    pin_memory = device.type == "cuda"

    train_samples, val_samples, test_samples, used_predefined_splits = load_data_splits(args)
    all_samples = train_samples + val_samples + test_samples

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
    test_dataset = BUSIDataset(
        test_samples,
        image_size=args.image_size,
        augment=False,
    ) if test_samples else None

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    ) if test_dataset is not None else None

    if args.input_mode == "fusion":
        model = AttentionUNetFusion(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels,
        ).to(device)
    elif args.input_mode == "ca_unet":
        model = CA_UNet(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels,
        ).to(device)
    elif args.input_mode == "eca_unet":
        model = ECA_UNet(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels,
        ).to(device)
    elif args.input_mode == "simam_unet":
        model = SimAM_UNet(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels,
        ).to(device)
    elif args.input_mode == "ema_unet":
        model = EMA_UNet(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels,
        ).to(device)
    elif args.input_mode == "ghost_unet":
        model = Ghost_UNet(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels,
        ).to(device)
    elif args.input_mode == "smp_mobilenet":
        model = SMP_UNet(
            encoder_name="mobilenet_v2",
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif args.input_mode == "smp_lite":
        model = SMP_MobileNet_Lite(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif args.input_mode == "smp_micro":
        model = SMP_MobileNetV3_Micro(
            in_channels=1,
            out_channels=1,
        ).to(device)

    else:
        model = UNet(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels,
            input_mode=args.input_mode,
        ).to(device)

    criterion = DeepSupervisionLoss(
        base_criterion=FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75),
        weights=(1.0, 0.5, 0.25, 0.125)
    )
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
    print(f"Test                 : {len(test_samples)}")
    print(f"Chế độ chia dữ liệu  : {'train/valid/test có sẵn' if used_predefined_splits else 'chia ngẫu nhiên'}")
    print(f"Chế độ chạy (Input Mode): {args.input_mode.upper()}")
    print(f"Số tham số mô hình   : {count_parameters(model):,}")
    print(f"Kích thước ảnh       : {args.image_size} x {args.image_size}")
    print("=" * 70)

    history_path = output_dir / f"history_{args.input_mode}.csv"
    best_path = output_dir / f"best_unet_{args.input_mode}.pt"
    config_path = output_dir / f"config_{args.input_mode}.json"
    test_metrics_path = output_dir / f"test_metrics_{args.input_mode}.json"

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
                        "input_mode": args.input_mode,
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
    if test_loader is not None and best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print("\nĐang tự động tính toán B-IoU, HD95, IoU, Dice trên tập Test...")
        try:
            from evaluate_metrics import calculate_iou, calculate_hd95, calculate_boundary_iou, calculate_dice
            
            metrics = {"dice": [], "iou": [], "hd95": [], "biou": []}
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Test Metrics"):
                    images = batch["image"].to(device, non_blocking=True)
                    clahe_images = batch["clahe"].to(device, non_blocking=True)
                    masks = batch["mask"].cpu().numpy()
                    
                    logits = model(images, clahe_images)
                    if isinstance(logits, list):
                        logits_eval = logits[0]
                    else:
                        logits_eval = logits
                        
                    probs = torch.sigmoid(logits_eval).cpu().numpy()
                    preds = (probs >= 0.5).astype(np.uint8)
                    
                    for i in range(len(preds)):
                        pred_i = preds[i].squeeze()
                        mask_i = masks[i].squeeze()
                        
                        metrics["dice"].append(calculate_dice(pred_i, mask_i))
                        metrics["iou"].append(calculate_iou(pred_i, mask_i))
                        
                        hd95_val = calculate_hd95(pred_i, mask_i)
                        if not np.isnan(hd95_val):
                            metrics["hd95"].append(hd95_val)
                            
                        metrics["biou"].append(calculate_boundary_iou(pred_i, mask_i))
                        
            test_dice = float(np.mean(metrics["dice"]))
            test_iou = float(np.mean(metrics["iou"]))
            test_biou = float(np.mean(metrics["biou"]))
            test_hd95 = float(np.mean(metrics["hd95"]))
            
            with test_metrics_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "test_dice": test_dice,
                        "test_iou": test_iou,
                        "test_biou": test_biou,
                        "test_hd95": test_hd95,
                        "checkpoint": str(best_path),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"\nTest Dice  : {test_dice:.4f}")
            print(f"Test IoU   : {test_iou:.4f}")
            print(f"Test B-IoU : {test_biou:.4f}")
            print(f"Test HD95  : {test_hd95:.4f}")
            print(f"Đã lưu Test metrics vào: {test_metrics_path}")

        except ImportError:
            print("Không tìm thấy evaluate_metrics.py, quay về tính Test Loss/Dice mặc định.")
            with torch.no_grad():
                test_loss, test_dice = run_epoch(
                    model,
                    test_loader,
                    criterion,
                    device,
                )
            print(f"Test loss: {test_loss:.4f}")
            print(f"Test Dice: {test_dice:.4f}")
    if used_predefined_splits:
        print(
            "\nLƯU Ý NGHIÊN CỨU: đang dùng split train/valid/test có sẵn. "
            "Dùng validation để chọn mô hình và chỉ báo cáo test độc lập sau cùng."
        )
    else:
        print(
            "\nLƯU Ý NGHIÊN CỨU: cách chia ngẫu nhiên trong file này chỉ dùng để "
            "test pipeline. Khi làm bài báo, hãy chia theo bệnh nhân hoặc dùng fold "
            "chính thức của BUS-BRA để tránh rò rỉ dữ liệu."
        )


if __name__ == "__main__":
    main()
