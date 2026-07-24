import torch
from dataset import BUSIDataset, discover_predefined_split_samples
from pathlib import Path

def main():
    data_dir = "Dataset_BUSI_70_15_15"
    splits = discover_predefined_split_samples(data_dir)
    train_samples = splits["train"]
    print(f"Loaded {len(train_samples)} samples from train split.")

    dataset = BUSIDataset(train_samples, image_size=256, augment=True)
    batch = dataset[0]
    
    print("Keys in batch:", batch.keys())
    print("Image shape:", batch["image"].shape)
    print("CLAHE shape:", batch["clahe"].shape)
    print("Mask shape:", batch["mask"].shape)

    # Check if shapes match
    assert batch["image"].shape == batch["clahe"].shape == batch["mask"].shape
    print("Shapes match perfectly!")

if __name__ == "__main__":
    main()
