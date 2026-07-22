import torch

from model import UNet, count_parameters


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1, base_channels=16).to(device)

    x = torch.randn(2, 1, 256, 256, device=device)
    with torch.no_grad():
        y = model(x)

    print("=" * 55)
    print("U-Net smoke test thành công")
    print(f"Thiết bị       : {device}")
    print(f"Input shape    : {tuple(x.shape)}")
    print(f"Output shape   : {tuple(y.shape)}")
    print(f"Số tham số     : {count_parameters(model):,}")
    print(f"CUDA khả dụng  : {torch.cuda.is_available()}")
    print("=" * 55)

    assert y.shape == (2, 1, 256, 256)


if __name__ == "__main__":
    main()
