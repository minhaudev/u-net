import torch
from model import UNet, AttentionUNetFusion, CA_UNet, CAMLP_UNet, count_parameters

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
    
    print("\n=== Testing CAMLP-UNet (Coordinate Attention + Tokenized MLP) ===")
    model_camlp = CAMLP_UNet(in_channels=1, out_channels=1, base_channels=16)
    print(f"Total Trainable Parameters (CAMLP-UNet): {count_parameters(model_camlp):,}")
    
    model_camlp.train()
    outputs_camlp_train = model_camlp(x_orig)
    print(f"Outputs in Train Mode (List length): {len(outputs_camlp_train)}")
    
    model_camlp.eval()
    with torch.no_grad():
        output_camlp_eval = model_camlp(x_orig)
    print(f"Output Shape in Eval Mode: {output_camlp_eval.shape}")
    
    assert isinstance(outputs_camlp_train, list) and len(outputs_camlp_train) == 4, "Train mode should return 4 outputs"
    assert output_camlp_eval.shape == (2, 1, 256, 256), "Eval output shape mismatch!"
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    main()
