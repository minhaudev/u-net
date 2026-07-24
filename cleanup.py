import os
import shutil

# Danh sách các file đơn lẻ cần xóa
files_to_delete = [
    "kaggle_unet_busi.ipynb",
    "archive.zip",
    "check_names.py",
    "split_busbra.py",
    "format_busbra_to_busi.py",
    "split_dataset.py"
]

# Danh sách các thư mục chứa dữ liệu thô dư thừa (vì ta đã có bản 70_15_15 hoàn chỉnh)
dirs_to_delete = [
    "dataset",              # Thư mục rác từ roboflow
    "archive",              # Dữ liệu thô BUSBRA
    "Dataset_BUSI_with_GT", # Dữ liệu thô BUSI
    "splits"                # Chứa các file CSV nháp
]

print("Bắt đầu dọn dẹp không gian làm việc...")

for f in files_to_delete:
    if os.path.exists(f):
        try:
            os.remove(f)
            print(f" Đã xóa file: {f}")
        except Exception as e:
            print(f" Lỗi khi xóa {f}: {e}")

for d in dirs_to_delete:
    if os.path.exists(d):
        try:
            shutil.rmtree(d)
            print(f" Đã xóa thư mục: {d}/")
        except Exception as e:
            print(f" Lỗi khi xóa {d}: {e}")

print("Dọn dẹp hoàn tất! Workspace bây giờ cực kỳ gọn gàng.")
