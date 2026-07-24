import argparse
from collections import Counter
from pathlib import Path

from console_utils import configure_utf8_output
from dataset import discover_busi_samples, discover_predefined_split_samples


def main() -> None:
    configure_utf8_output()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--include-normal", action="store_true")
    args = parser.parse_args()

    split_samples = discover_predefined_split_samples(args.data_dir)
    if split_samples:
        print("Phat hien dataset co san train/valid/test:")
        for split_name, samples in split_samples.items():
            multi_masks = sum(len(s.mask_paths) > 1 for s in samples)
            print(f"  {split_name}: {len(samples)} anh, {multi_masks} anh co nhieu mask")
        samples = [sample for split in split_samples.values() for sample in split]
    else:
        samples = discover_busi_samples(args.data_dir, include_normal=args.include_normal)

    groups = Counter(s.image_path.parent.name for s in samples)
    multi_masks = sum(len(s.mask_paths) > 1 for s in samples)

    print(f"Tổng số ảnh hợp lệ: {len(samples)}")
    for group, count in sorted(groups.items()):
        print(f"  {group}: {count}")
    print(f"Ảnh có nhiều mask: {multi_masks}")
    print("\n5 mẫu đầu:")
    for sample in samples[:5]:
        print(f"- {sample.image_path.name} -> {len(sample.mask_paths)} mask")


if __name__ == "__main__":
    main()
