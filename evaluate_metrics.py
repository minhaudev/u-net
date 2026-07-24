import argparse
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

from torch.utils.data import DataLoader
from dataset import BUSIDataset, discover_predefined_split_samples, discover_busi_samples
from model import UNet

def calculate_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if np.sum(pred) == 0 and np.sum(gt) == 0 else 0.0
    return intersection / union

def calculate_hd95(pred, gt):
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 0.0
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return np.nan # Cannot compute distance if one is completely empty
        
    # Get edge boundaries
    pred_edges = cv2.Canny((pred * 255).astype(np.uint8), 0, 1) > 0
    gt_edges = cv2.Canny((gt * 255).astype(np.uint8), 0, 1) > 0
    
    # Distance transform
    dt_gt = distance_transform_edt(~gt_edges)
    dt_pred = distance_transform_edt(~pred_edges)
    
    # Get distances at boundary points
    dists_pred_to_gt = dt_gt[pred_edges]
    dists_gt_to_pred = dt_pred[gt_edges]
    
    if len(dists_pred_to_gt) == 0 or len(dists_gt_to_pred) == 0:
        return np.nan
        
    # 95th percentile
    hd95 = max(np.percentile(dists_pred_to_gt, 95), np.percentile(dists_gt_to_pred, 95))
    return hd95

def calculate_boundary_iou(pred, gt, dilation_ratio=0.02):
    img_diag = np.sqrt(pred.shape[0]**2 + pred.shape[1]**2)
    d = int(round(dilation_ratio * img_diag))
    if d < 1: d = 1
    
    kernel = np.ones((d, d), np.uint8)
    
    pred_uint8 = (pred > 0).astype(np.uint8)
    gt_uint8 = (gt > 0).astype(np.uint8)
    
    pred_boundary = cv2.dilate(pred_uint8, kernel) - cv2.erode(pred_uint8, kernel)
    gt_boundary = cv2.dilate(gt_uint8, kernel) - cv2.erode(gt_uint8, kernel)
    
    intersection = np.logical_and(pred_boundary, gt_boundary).sum()
    union = np.logical_or(pred_boundary, gt_boundary).sum()
    if union == 0:
         # Nếu cả 2 đều không có ranh giới, coi như match 100% nếu chúng đều trống
        return 1.0 if np.sum(pred) == 0 and np.sum(gt) == 0 else 0.0
    return intersection / union

def calculate_dice(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    denominator = np.sum(pred) + np.sum(gt)
    if denominator == 0:
        return 1.0
    return 2.0 * intersection / denominator

def parse_args():
    parser = argparse.ArgumentParser(description="Compute All Test Metrics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn đến file .pt")
    parser.add_argument("--data-dir", type=str, required=True, help="Thư mục dataset (ví dụ: Dataset_BUSBRA_70_15_15)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    base_channels = checkpoint.get("base_channels", 16)
    image_size = checkpoint.get("image_size", 256)
    input_mode = checkpoint.get("input_mode", "orig")
    best_val_dice = checkpoint.get("best_val_dice", "Không rõ")
    
    print(f"Best Validation Dice (từ file pt): {best_val_dice}")
    print(f"Input mode của mô hình: {input_mode}")
    
    model = UNet(in_channels=1, out_channels=1, base_channels=base_channels, input_mode=input_mode).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test split
    predefined_splits = discover_predefined_split_samples(args.data_dir)
    if predefined_splits and "test" in predefined_splits and len(predefined_splits["test"]) > 0:
        test_samples = predefined_splits["test"]
        print(f"Đã tìm thấy {len(test_samples)} ảnh trong tập Test.")
    else:
        print("CẢNH BÁO: Không tìm thấy tập 'test' được chia sẵn. Sẽ chạy trên toàn bộ dataset (không khuyến khích để báo cáo).")
        test_samples = discover_busi_samples(args.data_dir, include_normal=False)
        
    dataset = BUSIDataset(test_samples, image_size=image_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics = {"dice": [], "iou": [], "hd95": [], "biou": []}
    
    progress = tqdm(loader, desc="Evaluating Metrics")
    with torch.no_grad():
        for batch in progress:
            images = batch["image"].to(device)
            clahe_images = batch["clahe"].to(device)
            masks = batch["mask"].cpu().numpy() # Ground truth mask
            
            # Mô hình dự đoán
            logits = model(images, clahe_images)
            if isinstance(logits, list):
                logits_eval = logits[0]
            else:
                logits_eval = logits
                
            probs = torch.sigmoid(logits_eval).cpu().numpy()
            preds = (probs >= 0.5).astype(np.uint8) # Threshold 0.5
            
            # Tính metrics cho từng ảnh trong batch
            for i in range(len(preds)):
                pred_i = preds[i].squeeze()
                mask_i = masks[i].squeeze()
                
                dice_score = calculate_dice(pred_i, mask_i)
                iou_score = calculate_iou(pred_i, mask_i)
                hd95_score = calculate_hd95(pred_i, mask_i)
                biou_score = calculate_boundary_iou(pred_i, mask_i)
                
                metrics["dice"].append(dice_score)
                metrics["iou"].append(iou_score)
                if not np.isnan(hd95_score): # Bỏ qua NaN (khi một trong hai trống trơn)
                    metrics["hd95"].append(hd95_score)
                metrics["biou"].append(biou_score)

    # In kết quả
    print("\n" + "="*50)
    print("KẾT QUẢ TRÊN TẬP TEST:")
    print(f"Test Dice  : {np.mean(metrics['dice']):.4f}")
    print(f"Test IoU   : {np.mean(metrics['iou']):.4f}")
    print(f"Test B-IoU : {np.mean(metrics['biou']):.4f}")
    print(f"Test HD95  : {np.mean(metrics['hd95']):.4f} pixels")
    print("="*50)
    
if __name__ == "__main__":
    main()
