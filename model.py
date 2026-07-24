import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Depthwise
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        # Pointwise
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DoubleDSConv(nn.Module):
    """Hai lớp DSConv-BatchNorm-ReLU liên tiếp thay vì Conv chuẩn để giảm tham số."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialGatingFusion(nn.Module):
    """
    Trộn 2 tensor (gốc và CLAHE).
    - Concat 2 tensor
    - Qua Conv1x1 -> Sigmoid ra gating map (attention map 2D)
    - Nhân chéo để trộn: gate * x_orig + (1 - gate) * x_clahe
    """
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        # 2 inputs được nối lại nên số kênh nhân đôi
        self.conv = nn.Conv2d(in_channels * 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_orig: torch.Tensor, x_clahe: torch.Tensor) -> torch.Tensor:
        # Nối 2 tensor lại theo chiều channel (dim=1)
        x_cat = torch.cat([x_orig, x_clahe], dim=1)
        # Bản đồ Gating 2D
        gate = self.sigmoid(self.conv(x_cat))
        # Trộn lại
        out = gate * x_orig + (1.0 - gate) * x_clahe
        return out


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleDSConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleDSConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Phòng trường hợp kích thước lẻ làm lệch 1 pixel.
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net 2D siêu nhẹ cho binary segmentation.
    Sử dụng Depthwise Separable Convolution và Spatial Gating Fusion.

    Input : x (Original B x 1 x H x W), x_clahe (CLAHE B x 1 x H x W)
    Output: B x 1 x H x W (logits)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        input_mode: str = "gating",
    ) -> None:
        super().__init__()
        c = base_channels
        self.input_mode = input_mode
        
        # Spatial Gating Fusion
        self.fusion = SpatialGatingFusion(in_channels=in_channels)

        # Sử dụng DoubleDSConv thay cho DoubleConv
        self.inc = DoubleDSConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)

        self.up1 = Up(c * 16, c * 8, c * 8)
        self.up2 = Up(c * 8, c * 4, c * 4)
        self.up3 = Up(c * 4, c * 2, c * 2)
        self.up4 = Up(c * 2, c, c)

        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)
        
        # Deep supervision layers
        self.outc_ds1 = nn.Conv2d(c * 2, out_channels, kernel_size=1)
        self.outc_ds2 = nn.Conv2d(c * 4, out_channels, kernel_size=1)
        self.outc_ds3 = nn.Conv2d(c * 8, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, x_clahe: torch.Tensor):
        # Tùy chọn nhánh Ablation
        if self.input_mode == "orig":
            fused_x = x
        elif self.input_mode == "clahe":
            fused_x = x_clahe
        else:
            # Bước trộn ảnh ngay đầu vào Encoder (Spatial Gating)
            fused_x = self.fusion(x, x_clahe)
        
        x1 = self.inc(fused_x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)
        
        final_out = self.outc(d4)
        
        if self.training:
            out_ds1 = self.outc_ds1(d3)
            out_ds2 = self.outc_ds2(d2)
            out_ds3 = self.outc_ds3(d1)
            return [final_out, out_ds1, out_ds2, out_ds3]
        else:
            return final_out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
