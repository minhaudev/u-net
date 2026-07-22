# U-Net BUSI Starter

Code huan luyen va du doan segmentation anh sieu am vu bang U-Net tren bo du lieu BUSI.

## Cau truc chinh

- `model.py`: mo hinh U-Net.
- `dataset.py`: doc anh BUSI va ghep nhieu mask neu co.
- `losses.py`: loss BCE + Dice va Dice score.
- `train.py`: huan luyen mo hinh.
- `predict.py`: du doan mask cho mot anh.
- `inspect_dataset.py`: kiem tra cau truc dataset.
- `smoke_test.py`: kiem tra nhanh model khong can dataset.
- `kaggle_unet_busi.ipynb`: notebook de chay tren Kaggle.

## Chuan bi moi truong

Khuyen nghi dung Python 3.11 tren Windows.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

Neu PowerShell chan script activate, co the dung CMD:

```bat
.venv\Scripts\activate.bat
```

## Chuan bi dataset

Dataset khong duoc day len GitHub. Hay dat BUSI o mot trong hai cach sau:

```text
data/Dataset_BUSI_with_GT/
```

hoac:

```text
Dataset_BUSI_with_GT/
```

Ben trong dataset can co cac thu muc:

```text
benign/
malignant/
normal/
```

Vi du file:

```text
benign (1).png
benign (1)_mask.png
benign (1)_mask_1.png
```

## Cach chay

Kiem tra U-Net khong can dataset:

```powershell
python smoke_test.py
```

Kiem tra dataset:

```powershell
python inspect_dataset.py --data-dir "data\Dataset_BUSI_with_GT"
```

Neu muon tinh ca lop `normal`:

```powershell
python inspect_dataset.py --data-dir "data\Dataset_BUSI_with_GT" --include-normal
```

Train thu nhanh:

```powershell
python train.py --data-dir "data\Dataset_BUSI_with_GT" --epochs 2 --image-size 128 --batch-size 2 --output-dir outputs_test
```

Train cau hinh co so:

```powershell
python train.py --data-dir "data\Dataset_BUSI_with_GT" --epochs 30 --image-size 256 --batch-size 8 --base-channels 16 --output-dir outputs_busi
```

Neu may yeu hoac thieu GPU/VRAM:

```powershell
python train.py --data-dir "data\Dataset_BUSI_with_GT" --epochs 2 --image-size 128 --batch-size 2 --cpu
```

Du doan mot anh sau khi train:

```powershell
python predict.py --checkpoint "outputs_busi\best_unet.pt" --image "data\Dataset_BUSI_with_GT\benign\benign (1).png" --output-dir prediction
```

Ket qua du doan nam trong:

```text
prediction/predicted_mask.png
prediction/overlay.png
```
