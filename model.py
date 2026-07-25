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


# --- CÁC COMPONENT MỚI (TỐI ƯU HÓA) KHÔNG ẢNH HƯỞNG CODE CŨ ---

class ResidualDSConv(nn.Module):
    """Giống DoubleDSConv nhưng có thêm Residual connection"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut để khớp số kênh nếu thay đổi
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.block(x)
        out += identity
        out = self.relu(out)
        return out


class AttentionGate(nn.Module):
    """Attention Gate sử dụng trong skip connection (từ Attention U-Net)"""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ECABlock(nn.Module):
    """Efficient Channel Attention (ECA) - Cơ chế Attention kênh siêu nhẹ 1D"""
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x) # (B, C, 1, 1)
        # Reshape sang 1D conv
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CBAMSpatial(nn.Module):
    """Spatial Attention siêu nhẹ từ CBAM"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        sa = self.sigmoid(self.conv(x_cat))
        return x * sa


class FeatureFusionModule(nn.Module):
    """Trộn đặc trưng sau lớp Conv đầu tiên sử dụng ECA (Channel) & CBAM (Spatial)"""
    def __init__(self, channels: int):
        super().__init__()
        self.eca = ECABlock(k_size=3)
        self.spatial = CBAMSpatial(kernel_size=7)
        self.fuse_conv = ResidualDSConv(channels * 2, channels)

    def forward(self, f_orig: torch.Tensor, f_clahe: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f_orig, f_clahe], dim=1)
        x = self.eca(x)
        x = self.spatial(x)
        return self.fuse_conv(x)


class AttDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualDSConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttUp(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        # Attention Gate cho skip connection
        self.ag = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=skip_channels // 2)
        self.conv = ResidualDSConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
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

        # Lọc skip feature qua Attention Gate
        skip = self.ag(g=x, x=skip)
        
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNetFusion(nn.Module):
    """
    Phiên bản U-Net Tối Ưu (Feature Fusion + Attention Gate + Residual)
    Giữ nguyên độ nhẹ (lightweight) cho bài toán Segmentation y tế.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        c = base_channels
        
        # 1. Trích xuất đặc trưng chung (Shared Encoder khối đầu)
        self.shared_inc = ResidualDSConv(in_channels, c)
        
        # 2. Module trộn đặc trưng (Feature-level Fusion)
        self.fusion = FeatureFusionModule(channels=c)
        
        # 3. Encoder
        self.down1 = AttDown(c, c * 2)
        self.down2 = AttDown(c * 2, c * 4)
        self.down3 = AttDown(c * 4, c * 8)
        self.down4 = AttDown(c * 8, c * 16)

        # 4. Decoder với Attention Gate
        self.up1 = AttUp(c * 16, c * 8, c * 8)
        self.up2 = AttUp(c * 8, c * 4, c * 4)
        self.up3 = AttUp(c * 4, c * 2, c * 2)
        self.up4 = AttUp(c * 2, c, c)

        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)
        
        # Deep supervision layers
        self.outc_ds1 = nn.Conv2d(c * 2, out_channels, kernel_size=1)
        self.outc_ds2 = nn.Conv2d(c * 4, out_channels, kernel_size=1)
        self.outc_ds3 = nn.Conv2d(c * 8, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, x_clahe: torch.Tensor):
        # Trích xuất đặc trưng bậc thấp từ cả 2 nhánh (sử dụng chung 1 layer để bớt tham số)
        f1_orig = self.shared_inc(x)
        f1_clahe = self.shared_inc(x_clahe)
        
        # Trộn đặc trưng bằng Attention
        x1 = self.fusion(f1_orig, f1_clahe)
        
        # Các bước tiếp theo như U-Net bình thường
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1) # skip x1 đã là kết quả từ fusion
        
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
