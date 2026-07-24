import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BUSIDataset, discover_busi_samples, discover_predefined_split_samples
from model import UNet
from losses import dice_score_from_logits

def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot Evaluation U-Net")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn đến file .pt của mô hình đã train")
    parser.add_argument("--target-dir", type=str, required=True, help="Thư mục dataset mới để test zero-shot (vd: Dataset_BUSI_70_15_15)")
    parser.add_argument("--baseline-dice", type=float, default=None, help="Dice score trên tập gốc để tính Retention Rate")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    
    print("=" * 60)
    print("ZERO-SHOT EVALUATION")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Target Data: {args.target_dir}")
    print("=" * 60)

    # 1. Load Checkpoint to get base_channels and image_size
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        base_channels = checkpoint.get("base_channels", 16)
        image_size = checkpoint.get("image_size", 256)
    except Exception as e:
        print(f"Lỗi khi load checkpoint: {e}")
        return
    
    # 2. Setup Model
    model = UNet(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Load Target Data
    # Thu thập tất cả các ảnh từ dataset mục tiêu
    predefined_splits = discover_predefined_split_samples(args.target_dir)
    if predefined_splits:
        all_samples = predefined_splits.get("train", []) + predefined_splits.get("valid", []) + predefined_splits.get("test", [])
    else:
        all_samples = discover_busi_samples(args.target_dir, include_normal=False)

    print(f"Đã tìm thấy {len(all_samples)} ảnh trong tập target.")
    
    dataset = BUSIDataset(all_samples, image_size=image_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 4. Evaluation
    total_dice = 0.0
    total_items = 0
    
    progress = tqdm(loader, desc="Evaluating")
    with torch.no_grad():
        for batch in progress:
            images = batch["image"].to(device)
            clahe_images = batch["clahe"].to(device)
            masks = batch["mask"].to(device)
            batch_size = images.size(0)

            # Forward
            logits = model(images, clahe_images)
            
            # Xử lý output nếu mô hình trả về list (Deep Supervision)
            if isinstance(logits, list):
                logits_eval = logits[0]
            else:
                logits_eval = logits
                
            dice = dice_score_from_logits(logits_eval, masks)
            
            total_dice += dice.item() * batch_size
            total_items += batch_size

    avg_dice = total_dice / total_items
    print("\n" + "=" * 60)
    print(f"Zero-Shot Dice Score: {avg_dice:.4f}")
    
    if args.baseline_dice is not None:
        retention_rate = (avg_dice / args.baseline_dice) * 100
        print(f"Baseline Dice       : {args.baseline_dice:.4f}")
        print(f"Retention Rate      : {retention_rate:.2f}%")
        
        # Đánh giá mức độ giữ hiệu năng
        if retention_rate >= 90:
            print("Đánh giá: Rất xuất sắc (Mô hình giữ được >= 90% hiệu năng trên dữ liệu mới).")
        elif retention_rate >= 75:
            print("Đánh giá: Khá tốt (Mô hình có khả năng tổng quát hóa ổn định).")
        else:
            print("Đánh giá: Cần cải thiện (Mô hình bị giảm hiệu năng đáng kể, có thể do domain shift).")
    else:
        print("Mẹo: Chạy lại với cờ --baseline-dice <giá trị> để tính Retention Rate.")
    print("=" * 60)

if __name__ == "__main__":
    main()
