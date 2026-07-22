# U-Net BUSI Starter

Bộ mã dùng để:

1. Kiểm tra U-Net chạy được trên máy.
2. Kiểm tra cấu trúc bộ dữ liệu BUSI.
3. Huấn luyện U-Net cơ bản.
4. Dự đoán mask trên một ảnh mới.
5. Chuyển cùng quy trình lên Kaggle.

## 1. Cấu trúc dự án

```text
unet_busi_starter/
├── model.py
├── dataset.py
├── losses.py
├── smoke_test.py
├── inspect_dataset.py
├── train.py
├── predict.py
├── requirements.txt
└── kaggle_unet_busi.ipynb
```

## 2. Chuẩn bị trên Windows

Khuyến nghị dùng Python 3.11.

Mở CMD hoặc PowerShell tại thư mục dự án:

```powershell
py -3.11 -m venv .venv
```

Kích hoạt môi trường:

### CMD

```bat
.venv\Scripts\activate.bat
```

### PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script, dùng CMD để kích hoạt.

Nâng cấp pip:

```powershell
python -m pip install --upgrade pip
```

Cài PyTorch và thư viện:

```powershell
pip install torch torchvision
pip install -r requirements.txt
```

## 3. Kiểm tra U-Net chưa cần dataset

```powershell
python smoke_test.py
```

Kết quả đúng phải có dạng:

```text
U-Net smoke test thành công
Input shape  : (2, 1, 256, 256)
Output shape : (2, 1, 256, 256)
```

## 4. Chuẩn bị BUSI

Sau khi tải và giải nén, cấu trúc nên gần giống:

```text
data/
└── Dataset_BUSI_with_GT/
    ├── benign/
    ├── malignant/
    └── normal/
```

Trong mỗi thư mục:

```text
benign (1).png
benign (1)_mask.png
```

Một số ảnh có nhiều mask:

```text
benign (1)_mask.png
benign (1)_mask_1.png
```

Code sẽ tự hợp nhất các mask đó.

## 5. Kiểm tra dataset

```powershell
python inspect_dataset.py --data-dir "data\Dataset_BUSI_with_GT"
```

Giai đoạn đầu chỉ dùng benign và malignant.

Muốn đưa cả normal vào:

```powershell
python inspect_dataset.py --data-dir "data\Dataset_BUSI_with_GT" --include-normal
```

## 6. Train thử nhanh trên máy

Chạy 2 epoch, ảnh 128 x 128 để kiểm tra pipeline:

```powershell
python train.py ^
  --data-dir "data\Dataset_BUSI_with_GT" ^
  --epochs 2 ^
  --image-size 128 ^
  --batch-size 2 ^
  --output-dir outputs_test
```

Nếu dùng PowerShell, có thể viết trên một dòng:

```powershell
python train.py --data-dir "data\Dataset_BUSI_with_GT" --epochs 2 --image-size 128 --batch-size 2 --output-dir outputs_test
```

Nếu máy yếu hoặc gặp lỗi GPU:

```powershell
python train.py --data-dir "data\Dataset_BUSI_with_GT" --epochs 2 --image-size 128 --batch-size 2 --cpu
```

## 7. Train cấu hình cơ sở

```powershell
python train.py --data-dir "data\Dataset_BUSI_with_GT" --epochs 30 --image-size 256 --batch-size 8 --base-channels 16 --output-dir outputs_busi
```

Nếu thiếu RAM/VRAM, giảm:

```text
batch-size: 8 -> 4 -> 2
image-size: 256 -> 192 -> 128
```

## 8. Dự đoán một ảnh

```powershell
python predict.py ^
  --checkpoint "outputs_busi\best_unet.pt" ^
  --image "data\Dataset_BUSI_with_GT\benign\benign (1).png" ^
  --output-dir prediction
```

Kết quả:

```text
prediction/predicted_mask.png
prediction/overlay.png
```

## 9. Chạy trên Kaggle

1. Tạo Notebook mới trên Kaggle.
2. Chọn **Add Input** và thêm dataset BUSI.
3. Trong phần Notebook settings, bật GPU.
4. Upload file `kaggle_unet_busi.ipynb` hoặc import notebook này.
5. Chạy từng cell từ trên xuống.
6. Notebook tự tìm thư mục `Dataset_BUSI_with_GT`.
7. Mô hình tốt nhất được lưu ở:

```text
/kaggle/working/best_unet_busi.pt
```

## 10. Lưu ý nghiên cứu

Pipeline hiện chia dữ liệu ngẫu nhiên theo ảnh để kiểm tra kỹ thuật.

Khi thực hiện bài báo:

- Dùng phân chia theo bệnh nhân nếu có patient ID.
- Hoặc dùng các fold chính thức của BUS-BRA.
- Không chọn cấu hình dựa trên test set.
- Báo cáo Dice, IoU, Sensitivity và hiệu quả tính toán.
