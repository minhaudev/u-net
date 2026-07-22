from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from console_utils import configure_utf8_output
from model import UNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="prediction")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_utf8_output()

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    base_channels = int(checkpoint.get("base_channels", 16))
    image_size = int(checkpoint.get("image_size", 256))

    model = UNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    original = Image.open(args.image).convert("L")
    resized = TF.resize(
        original,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
    )
    image_t = TF.to_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        probability = torch.sigmoid(model(image_t))[0, 0].cpu().numpy()

    mask = (probability >= args.threshold).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask, mode="L")
    mask_image.save(output_dir / "predicted_mask.png")

    # Overlay đỏ trên ảnh grayscale.
    base = np.asarray(resized.convert("RGB")).copy()
    region = mask > 0
    overlay = base.copy()
    overlay[region, 0] = 255
    overlay[region, 1] = (overlay[region, 1] * 0.35).astype(np.uint8)
    overlay[region, 2] = (overlay[region, 2] * 0.35).astype(np.uint8)

    blended = (0.65 * base + 0.35 * overlay).astype(np.uint8)
    Image.fromarray(blended).save(output_dir / "overlay.png")

    print(f"Đã lưu: {output_dir / 'predicted_mask.png'}")
    print(f"Đã lưu: {output_dir / 'overlay.png'}")


if __name__ == "__main__":
    main()
