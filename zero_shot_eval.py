import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BUSIDataset, discover_any_dataset
from model import UNet, AttentionUNetFusion, CA_UNet, ECA_UNet, SimAM_UNet, EMA_UNet, Ghost_UNet, SMP_UNet, SMP_MobileNet_Lite, SMP_MobileNetV3_Micro, Ghost_CA_UNet

# Import các hàm tính toán từ file evaluate_metrics vừa tạo
from evaluate_metrics import calculate_iou, calculate_hd95, calculate_boundary_iou, calculate_dice

def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot Evaluation U-Net (Full Metrics)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn đến file .pt của mô hình đã train")
    parser.add_argument("--target-dir", type=str, required=True, help="Thư mục dataset mới để test zero-shot (ví dụ Dataset_BUSI)")
    parser.add_argument("--baseline-dice", type=float, default=None, help="Dice score trên tập gốc để tính Retention Rate")
    parser.add_argument("--batch-size", type=int, default=1) # Sửa mặc định thành 1 để tính HD95 chính xác từng ảnh
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    
    print("=" * 60)
    print("ZERO-SHOT EVALUATION (TEST KHẢ NĂNG TỔNG QUÁT HÓA)")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Target Data: {args.target_dir}")
    print("=" * 60)

    # 1. Load Checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        base_channels = checkpoint.get("base_channels", 16)
        image_size = checkpoint.get("image_size", 256)
        input_mode = checkpoint.get("input_mode", "orig")
    except Exception as e:
        print(f"Lỗi khi load checkpoint: {e}")
        return
    
    # 2. Setup Model
    if input_mode == "fusion":
        model = AttentionUNetFusion(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    elif input_mode == "ca_unet":
        model = CA_UNet(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    elif input_mode == "eca_unet":
        model = ECA_UNet(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    elif input_mode == "simam_unet":
        model = SimAM_UNet(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    elif input_mode == "ema_unet":
        model = EMA_UNet(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    elif input_mode == "ghost_unet":
        model = Ghost_UNet(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    elif input_mode == "ghost_ca_unet":
        model = Ghost_CA_UNet(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    elif input_mode == "smp_mobilenet":
        model = SMP_UNet(encoder_name="mobilenet_v2", in_channels=1, out_channels=1).to(device)
    elif input_mode == "smp_lite":
        model = SMP_MobileNet_Lite(in_channels=1, out_channels=1).to(device)
    elif input_mode == "smp_micro":
        model = SMP_MobileNetV3_Micro(in_channels=1, out_channels=1).to(device)

    else:
        model = UNet(in_channels=1, out_channels=1, base_channels=base_channels, input_mode=input_mode).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Load Target Data
    # Đã là zero-shot thì gom toàn bộ dữ liệu của target dataset vào để test.
    # Sử dụng hàm đa năng để tự động phát hiện cấu trúc (BUSI, BrEaST, BUS-UCLM, v.v.)
    all_samples = discover_any_dataset(args.target_dir)

    print(f"Đã tìm thấy {len(all_samples)} ảnh trong tập Zero-Shot Target.")
    
    dataset = BUSIDataset(all_samples, image_size=image_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 4. Evaluation
    metrics = {"dice": [], "iou": [], "hd95": [], "biou": []}
    
    progress = tqdm(loader, desc="Evaluating Zero-Shot")
    with torch.no_grad():
        for batch in progress:
            images = batch["image"].to(device)
            clahe_images = batch["clahe"].to(device)
            masks = batch["mask"].cpu().numpy() # Đưa mask về numpy ngay để tính toán
            
            # Forward
            logits = model(images, clahe_images)
            
            if isinstance(logits, list):
                logits_eval = logits[0]
            else:
                logits_eval = logits
                
            probs = torch.sigmoid(logits_eval).cpu().numpy()
            preds = (probs >= 0.5).astype(np.uint8)
            
            # Tính metrics cho từng ảnh
            for i in range(len(preds)):
                pred_i = preds[i].squeeze()
                mask_i = masks[i].squeeze()
                
                dice_score = calculate_dice(pred_i, mask_i)
                iou_score = calculate_iou(pred_i, mask_i)
                hd95_score = calculate_hd95(pred_i, mask_i)
                biou_score = calculate_boundary_iou(pred_i, mask_i)
                
                metrics["dice"].append(dice_score)
                metrics["iou"].append(iou_score)
                if not np.isnan(hd95_score):
                    metrics["hd95"].append(hd95_score)
                metrics["biou"].append(biou_score)

    avg_dice = np.mean(metrics['dice'])
    
    print("\n" + "=" * 60)
    print("KẾT QUẢ ZERO-SHOT (TRÊN DỮ LIỆU LẠ):")
    print(f"Zero-Shot Dice Score : {avg_dice:.4f}")
    print(f"Zero-Shot IoU Score  : {np.mean(metrics['iou']):.4f}")
    print(f"Zero-Shot B-IoU Score: {np.mean(metrics['biou']):.4f}")
    print(f"Zero-Shot HD95       : {np.mean(metrics['hd95']):.4f} pixels")
    print("-" * 60)
    
    if args.baseline_dice is not None:
        retention_rate = (avg_dice / args.baseline_dice) * 100
        print(f"Baseline Dice       : {args.baseline_dice:.4f}")
        print(f"Retention Rate      : {retention_rate:.2f}% (Tỷ lệ giữ vững hiệu năng)")
        
        # Đánh giá mức độ giữ hiệu năng
        if retention_rate >= 90:
            print("Đánh giá: Rất xuất sắc (Mô hình giữ được >= 90% hiệu năng trên dữ liệu khác viện).")
        elif retention_rate >= 75:
            print("Đánh giá: Khá tốt (Mô hình có khả năng tổng quát hóa ổn định).")
        else:
            print("Đánh giá: Cần cải thiện (Mô hình bị giảm hiệu năng đáng kể do khác biệt ảnh chụp).")
    else:
        print("Mẹo: Chạy lại với cờ --baseline-dice <giá_trị> để script tự động tính Retention Rate.")
    print("=" * 60)

if __name__ == "__main__":
    main()
