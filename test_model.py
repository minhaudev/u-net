import torch
from model import UNet, AttentionUNetFusion, CA_UNet, ECA_UNet, SimAM_UNet, EMA_UNet, Ghost_UNet, SMP_UNet, SMP_MobileNet_Lite, SMP_MobileNetV3_Micro, count_parameters

def main():
    print("=== Testing Ultra-lightweight U-Net (Baseline) ===")
    model_base = UNet(in_channels=1, out_channels=1, base_channels=16)
    
    # 2 inputs: original image and CLAHE image
    x_orig = torch.randn(2, 1, 256, 256)
    x_clahe = torch.randn(2, 1, 256, 256)
    
    print(f"Total Trainable Parameters (Baseline): {count_parameters(model_base):,}")
    
    print("\n=== Testing Attention U-Net Fusion (New) ===")
    model_new = AttentionUNetFusion(in_channels=1, out_channels=1, base_channels=16)
    print(f"Total Trainable Parameters (New): {count_parameters(model_new):,}")

    # Forward pass (Train mode - Deep Supervision)
    model_new.train()
    outputs_train = model_new(x_orig, x_clahe)
    print(f"Outputs in Train Mode (List length): {len(outputs_train)}")
    for i, out in enumerate(outputs_train):
        print(f"  - Output {i} Shape: {out.shape}")
        
    # Forward pass (Eval mode - Single output)
    model_new.eval()
    with torch.no_grad():
        output_eval = model_new(x_orig, x_clahe)
    print(f"Output Shape in Eval Mode: {output_eval.shape}")
    
    assert isinstance(outputs_train, list) and len(outputs_train) == 4, "Train mode should return 4 outputs"
    assert output_eval.shape == (2, 1, 256, 256), "Eval output shape mismatch!"
    
    print("\n=== Testing CA-UNet (Coordinate Attention + ASPP) ===")
    model_ca = CA_UNet(in_channels=1, out_channels=1, base_channels=16)
    print(f"Total Trainable Parameters (CA-UNet): {count_parameters(model_ca):,}")
    
    model_ca.train()
    outputs_ca_train = model_ca(x_orig)
    print(f"Outputs in Train Mode (List length): {len(outputs_ca_train)}")
    
    model_ca.eval()
    with torch.no_grad():
        output_ca_eval = model_ca(x_orig)
    print(f"Output Shape in Eval Mode: {output_ca_eval.shape}")
    
    assert isinstance(outputs_ca_train, list) and len(outputs_ca_train) == 4, "Train mode should return 4 outputs"
    assert output_ca_eval.shape == (2, 1, 256, 256), "Eval output shape mismatch!"
    
    print("\n=== Testing ECA-UNet (Efficient Channel Attention) ===")
    model_eca = ECA_UNet(in_channels=1, out_channels=1, base_channels=16)
    print(f"Total Trainable Parameters (ECA-UNet): {count_parameters(model_eca):,}")
    model_eca.train()
    outputs_eca = model_eca(x_orig)
    assert len(outputs_eca) == 4
    model_eca.eval()
    with torch.no_grad():
        out_eca = model_eca(x_orig)
    assert out_eca.shape == (2, 1, 256, 256)

    print("\n=== Testing SimAM-UNet (Parameter-Free Attention) ===")
    model_simam = SimAM_UNet(in_channels=1, out_channels=1, base_channels=16)
    print(f"Total Trainable Parameters (SimAM-UNet): {count_parameters(model_simam):,}")
    model_simam.train()
    outputs_simam = model_simam(x_orig)
    assert len(outputs_simam) == 4
    model_simam.eval()
    with torch.no_grad():
        out_simam = model_simam(x_orig)
    assert out_simam.shape == (2, 1, 256, 256)


    print("\n=== Testing EMA-UNet (Efficient Multi-Scale Attention) ===")
    model_ema = EMA_UNet(in_channels=1, out_channels=1, base_channels=16)
    print(f"Total Trainable Parameters (EMA-UNet): {count_parameters(model_ema):,}")
    model_ema.train()
    outputs_ema = model_ema(x_orig)
    assert len(outputs_ema) == 4
    model_ema.eval()
    with torch.no_grad():
        out_ema = model_ema(x_orig)
    assert out_ema.shape == (2, 1, 256, 256)

    print("\n=== Testing Ghost-UNet (50% Params Reduction) ===")
    model_ghost = Ghost_UNet(in_channels=1, out_channels=1, base_channels=16)
    print(f"Total Trainable Parameters (Ghost-UNet): {count_parameters(model_ghost):,}")
    model_ghost.train()
    outputs_ghost = model_ghost(x_orig)
    assert len(outputs_ghost) == 4
    model_ghost.eval()
    with torch.no_grad():
        out_ghost = model_ghost(x_orig)
    assert out_ghost.shape == (2, 1, 256, 256)


    print("\n=== Testing Pre-trained SMP UNet (MobileNetV2) ===")
    model_smp = SMP_UNet(encoder_name="mobilenet_v2")
    print(f"Total Trainable Parameters (SMP MobileNetV2): {count_parameters(model_smp):,}")
    model_smp.train()
    outputs_smp = model_smp(x_orig)
    assert len(outputs_smp) == 4
    model_smp.eval()
    with torch.no_grad():
        out_smp = model_smp(x_orig)
    assert out_smp.shape == (2, 1, 256, 256)

    print("\n=== Testing Pre-trained SMP MobileNet Lite (50% Decoder) ===")
    model_smp_lite = SMP_MobileNet_Lite()
    print(f"Total Trainable Parameters (SMP Lite): {count_parameters(model_smp_lite):,}")
    model_smp_lite.train()
    outputs_smp_lite = model_smp_lite(x_orig)
    assert len(outputs_smp_lite) == 4
    model_smp_lite.eval()
    with torch.no_grad():
        out_smp_lite = model_smp_lite(x_orig)
    assert out_smp_lite.shape == (2, 1, 256, 256)

    print("\n=== Testing Pre-trained SMP MobileNetV3 Micro (Small Encoder + 25% Decoder) ===")
    model_smp_micro = SMP_MobileNetV3_Micro()
    print(f"Total Trainable Parameters (SMP Micro): {count_parameters(model_smp_micro):,}")
    model_smp_micro.train()
    outputs_smp_micro = model_smp_micro(x_orig)
    assert len(outputs_smp_micro) == 4
    model_smp_micro.eval()
    with torch.no_grad():
        out_smp_micro = model_smp_micro(x_orig)
    assert out_smp_micro.shape == (2, 1, 256, 256)

    print("All tests passed successfully!")

if __name__ == "__main__":
    main()
