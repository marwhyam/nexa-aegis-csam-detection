#!/usr/bin/env python3
"""
Check what's inside phase2_best_model.pt
Shows all keys and structure of the saved model
"""

import torch
import os

MODEL_PATH = r"C:\age-detection\phase2_best_model.pt"

print("="*60)
print("CHECKING MODEL CHECKPOINT")
print("="*60)
print(f"Model path: {MODEL_PATH}")
print(f"File exists: {os.path.exists(MODEL_PATH)}")

if not os.path.exists(MODEL_PATH):
    print("\n[ERROR] Model file not found!")
    exit(1)

# Get file size
file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
print(f"File size: {file_size:.2f} MB")
print()

# Load the checkpoint
print("Loading checkpoint...")
try:
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
except Exception as e:
    print(f"[ERROR] Failed to load: {e}")
    exit(1)

print("[OK] Checkpoint loaded successfully")
print()

# Check what type of object it is
print("="*60)
print("CHECKPOINT TYPE")
print("="*60)
print(f"Type: {type(checkpoint)}")
print()

# If it's a dictionary, show keys
if isinstance(checkpoint, dict):
    print("="*60)
    print("CHECKPOINT KEYS")
    print("="*60)
    for key in checkpoint.keys():
        print(f"  - {key}")
    print()
    
    # Check if it has model weights
    if 'model_state_dict' in checkpoint:
        print("="*60)
        print("MODEL_STATE_DICT KEYS (First 20)")
        print("="*60)
        state_dict = checkpoint['model_state_dict']
        keys = list(state_dict.keys())
        for i, key in enumerate(keys[:20]):
            print(f"  {i+1}. {key}")
        if len(keys) > 20:
            print(f"  ... and {len(keys) - 20} more keys")
        print()
        print(f"Total keys in model_state_dict: {len(keys)}")
        
        # Check if it has backbone layers (EfficientNet layers)
        backbone_keys = [k for k in keys if 'features' in k or 'classifier' not in k]
        classifier_keys = [k for k in keys if 'classifier' in k]
        
        print()
        print("="*60)
        print("MODEL STRUCTURE ANALYSIS")
        print("="*60)
        print(f"Backbone layers (features): {len(backbone_keys)}")
        print(f"Classifier layers: {len(classifier_keys)}")
        print()
        
        if len(backbone_keys) > 0:
            print("[OK] Contains backbone weights (full model)")
            print("Sample backbone keys:")
            for k in backbone_keys[:5]:
                print(f"  - {k}")
        else:
            print("[WARNING] No backbone weights found (only classifier?)")
        
        print()
        if len(classifier_keys) > 0:
            print("[OK] Contains classifier weights")
            print("Classifier keys:")
            for k in classifier_keys:
                print(f"  - {k}")
        
    elif 'state_dict' in checkpoint:
        print("="*60)
        print("STATE_DICT KEYS (First 20)")
        print("="*60)
        state_dict = checkpoint['state_dict']
        keys = list(state_dict.keys())
        for i, key in enumerate(keys[:20]):
            print(f"  {i+1}. {key}")
        if len(keys) > 20:
            print(f"  ... and {len(keys) - 20} more keys")
        print()
        print(f"Total keys in state_dict: {len(keys)}")
        
        # Check if it has backbone layers
        backbone_keys = [k for k in keys if 'features' in k or 'classifier' not in k]
        classifier_keys = [k for k in keys if 'classifier' in k]
        
        print()
        print("="*60)
        print("MODEL STRUCTURE ANALYSIS")
        print("="*60)
        print(f"Backbone layers (features): {len(backbone_keys)}")
        print(f"Classifier layers: {len(classifier_keys)}")
        print()
        
        if len(backbone_keys) > 0:
            print("[OK] Contains backbone weights (full model)")
            print("Sample backbone keys:")
            for k in backbone_keys[:5]:
                print(f"  - {k}")
        else:
            print("[WARNING] No backbone weights found (only classifier?)")
        
        print()
        if len(classifier_keys) > 0:
            print("[OK] Contains classifier weights")
            print("Classifier keys:")
            for k in classifier_keys:
                print(f"  - {k}")
    
    else:
        # Might be a direct state_dict
        print("="*60)
        print("DIRECT STATE_DICT (First 20 keys)")
        print("="*60)
        keys = list(checkpoint.keys())
        for i, key in enumerate(keys[:20]):
            print(f"  {i+1}. {key}")
        if len(keys) > 20:
            print(f"  ... and {len(keys) - 20} more keys")
        print()
        print(f"Total keys: {len(keys)}")
        
        # Check if it has backbone layers
        backbone_keys = [k for k in keys if 'features' in k or 'classifier' not in k]
        classifier_keys = [k for k in keys if 'classifier' in k]
        
        print()
        print("="*60)
        print("MODEL STRUCTURE ANALYSIS")
        print("="*60)
        print(f"Backbone layers (features): {len(backbone_keys)}")
        print(f"Classifier layers: {len(classifier_keys)}")
        print()
        
        if len(backbone_keys) > 0:
            print("[OK] Contains backbone weights (full model)")
            print("Sample backbone keys:")
            for k in backbone_keys[:5]:
                print(f"  - {k}")
        else:
            print("[WARNING] No backbone weights found (only classifier?)")
        
        print()
        if len(classifier_keys) > 0:
            print("[OK] Contains classifier weights")
            print("Classifier keys:")
            for k in classifier_keys:
                print(f"  - {k}")

else:
    # It's a direct state_dict (not wrapped in dict)
    print("="*60)
    print("DIRECT STATE_DICT (Not wrapped in dictionary)")
    print("="*60)
    keys = list(checkpoint.keys())
    print(f"Total keys: {len(keys)}")
    print("\nFirst 20 keys:")
    for i, key in enumerate(keys[:20]):
        print(f"  {i+1}. {key}")
    if len(keys) > 20:
        print(f"  ... and {len(keys) - 20} more keys")

print()
print("="*60)
print("SUMMARY")
print("="*60)
print("If you see 'features' keys, the model contains the full EfficientNet backbone.")
print("If you only see 'classifier' keys, you'll need ImageNet weights to initialize the backbone.")
print("="*60)

