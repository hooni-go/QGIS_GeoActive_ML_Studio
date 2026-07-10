import os
import sys
import torch
import numpy as np

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from models.model_builder import build_model

def test_models():
    print("=== Testing Model Instantiation ===")
    models = ["mask2former", "unet", "deeplabv3plus", "segformer"]
    num_classes = 4
    is_16bit = False
    
    for arch in models:
        try:
            print(f"Testing {arch}...")
            model = build_model(num_classes, is_16bit=is_16bit, backbone="swin-tiny", model_arch=arch)
            
            # Test forward pass with dummy data
            dummy_input = torch.randn(2, 3, 256, 256)
            
            with torch.no_grad():
                if arch == "mask2former":
                    out = model(pixel_values=dummy_input)
                    if hasattr(out, "class_queries_logits") and hasattr(out, "masks_queries_logits"):
                        print(f"  [OK] {arch} forward pass successful. Output dict keys available.")
                    else:
                        print(f"  [FAIL] {arch} missing expected outputs.")
                elif arch == "segformer":
                    out = model(pixel_values=dummy_input)
                    if hasattr(out, "logits"):
                        print(f"  [OK] {arch} forward pass successful. Logits shape: {out.logits.shape}")
                    else:
                        print(f"  [FAIL] {arch} missing logits.")
                else: # SMP models (UNet, DeepLabV3+)
                    out = model(dummy_input)
                    print(f"  [OK] {arch} forward pass successful. Logits shape: {out.shape}")
                    
        except Exception as e:
            print(f"  [ERROR] Failed to instantiate or run {arch}: {e}")

if __name__ == "__main__":
    test_models()
