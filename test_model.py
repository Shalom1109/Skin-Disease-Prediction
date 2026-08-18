import torch
import sys
import os

# Add current directory to path so we can import backend.model
sys.path.append(os.getcwd())

try:
    from backend.model import SkinSenseEnsemble
    print("Initializing Triple Ensemble (ConvNeXt + ViT + MaxViT)...")
    model = SkinSenseEnsemble(num_classes=7)
    
    # Simulate a small batch input (Batch Size = 1) memory test
    dummy_input = torch.randn(1, 3, 384, 384)
    print("Running forward pass (CPU/RAM test)...")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Success! Output shape: {output.shape}")
    print("The code has been successfully generated and tested!")

except Exception as e:
    print(f"Error during dry run: {e}")
