#!/usr/bin/env python3
"""
Download EfficientNet-B3 ImageNet weights for offline use

Run this on a machine with internet, then copy efficientnet_b3_imagenet_weights.pth
to your offline machine in the same directory as dashboard.py
"""

import torch
import os
from torchvision import models

print("Downloading EfficientNet-B3 ImageNet pretrained weights...")
print("This may take a few minutes...")

# Download weights
weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
model = models.efficientnet_b3(weights=weights)

# Save the weights to a file
output_file = "efficientnet_b3_imagenet_weights.pth"
torch.save(model.state_dict(), output_file)

print(f"\n[OK] Weights saved to: {output_file}")
print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
print("\nNext steps:")
print("1. Copy this file to your offline machine")
print("2. Place it in the same directory as dashboard.py")
print("3. The dashboard will automatically use it instead of downloading")

