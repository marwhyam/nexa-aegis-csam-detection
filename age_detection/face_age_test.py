#!/usr/bin/env python3
"""
Face Cropping & Age Detection Test Script
Simple script to test face detection, cropping, and age classification

================================================================================
HOW TO USE:
================================================================================
1. Edit the paths below (TEST_FOLDERS and TEST_IMAGES)
2. Run: python face_age_test.py
3. Cropped faces saved to: cropped_faces/ folder
4. Results saved to: face_age_results_TIMESTAMP.json

Example:
    TEST_FOLDERS = [
        "C:/images/folder1/",
    ]
    
    TEST_IMAGES = [
        "C:/images/test1.jpg",
    ]

================================================================================
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================

# Model paths
FACE_MODEL_PATH = "yolov8l-face.pt"  # YOLOv8 face detection model
AGE_MODEL_PATH = "phase2_best_model.pt"  # Phase 2 age detection model

# Folders to test - ADD YOUR FOLDER PATHS HERE
TEST_FOLDERS = [
    "C:/fyp_data/yolo/final_balanced/images/test",
    # Add more folders here...
]

# Single images to test (optional)
TEST_IMAGES = [
    # "path/to/image1.jpg",
    # Add more images here...
]

# Age detection settings
IMAGE_SIZE = 584
CLASS_NAMES = ['under18', 'adult']

# Confidence thresholds
FACE_CONF_THRESHOLD = 0.25  # Face detection confidence
AGE_CONF_THRESHOLD = 0.6  # Age classification confidence for Under-18

# Output directory for cropped faces
OUTPUT_DIR = "cropped_faces"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_age_model(model_path):
    """Load age detection model (Phase 2 EfficientNet-B3)"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Age model not found: {model_path}")
    
    # Build EfficientNet-B3 architecture (without downloading ImageNet weights)
    model = models.efficientnet_b3(weights=None)  # No download needed
    
    # Replace classifier (same as training)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=False),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(512, 2)  # Under-18 vs Adult
    )
    
    # Load weights (your phase2 model contains full model including backbone)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if isinstance(state_dict, dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
    model.load_state_dict(state_dict, strict=False)  # strict=False allows missing keys
    model.to(DEVICE)
    model.eval()
    print(f"[OK] Age detection model loaded: {model_path}")
    return model

def load_face_model(model_path):
    """Load YOLOv8 face detection model"""
    if not os.path.exists(model_path):
        print(f"[WARNING] Face model not found: {model_path}. Will try to download...")
        model = YOLO('yolov8l.pt')  # Downloads automatically
    else:
        model = YOLO(model_path)
    print(f"[OK] Face detection model loaded: {model_path}")
    return model

# ============================================================================
# PREPROCESSING
# ============================================================================

# Age detection transforms
age_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def detect_and_crop_faces(image_path, face_model):
    """
    Detect and crop faces from image using YOLOv8
    Returns: (cropped_faces, bounding_boxes, original_image)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return [], [], None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Run YOLOv8 detection
        results = face_model(img_rgb, conf=FACE_CONF_THRESHOLD, verbose=False)
        
        cropped_faces = []
        bounding_boxes = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Filter: Only keep detections that look like faces (rough aspect ratio check)
                width = x2 - x1
                height = y2 - y1
                if height > 0:
                    aspect_ratio = width / height
                    # Keep if aspect ratio suggests face-like region
                    if 0.5 <= aspect_ratio <= 1.5:
                        # Add 10% padding
                        padding_x = int(width * 0.1)
                        padding_y = int(height * 0.1)
                        
                        x1 = max(0, x1 - padding_x)
                        y1 = max(0, y1 - padding_y)
                        x2 = min(w, x2 + padding_x)
                        y2 = min(h, y2 + padding_y)
                        
                        face_crop = img_rgb[y1:y2, x1:x2]
                        
                        # Ensure minimum size (at least 32x32 pixels)
                        if face_crop.size > 0 and face_crop.shape[0] >= 32 and face_crop.shape[1] >= 32:
                            face_pil = Image.fromarray(face_crop)
                            cropped_faces.append(face_pil)
                            bounding_boxes.append((x1, y1, x2, y2))
        
        return cropped_faces, bounding_boxes, img_rgb
    except Exception as e:
        print(f"  [ERROR] Face detection failed: {e}")
        return [], [], None

def classify_age(face_image, age_model):
    """Classify age of face (Under-18 vs Adult)"""
    try:
        img_tensor = age_transform(face_image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = age_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            under18_prob = probs[0].item()
            adult_prob = probs[1].item()
        
        predicted_class = 0 if under18_prob > adult_prob else 1
        confidence = under18_prob if predicted_class == 0 else adult_prob
        
        return predicted_class, confidence, under18_prob
    except Exception as e:
        print(f"  [ERROR] Age classification failed: {e}")
        return 1, 0.0, 0.0  # Default to adult if error

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_image(image_path, face_model, age_model, save_crops=True):
    """
    Process single image: Detect faces → Crop → Classify age
    
    Returns:
        dict with results
    """
    image_path = Path(image_path)
    results = {
        'image_path': str(image_path),
        'timestamp': datetime.now().isoformat(),
        'faces_detected': 0,
        'under18_faces': 0,
        'face_details': []
    }
    
    print(f"\nProcessing: {image_path.name}")
    
    # Step 1: Detect and crop faces
    faces, boxes, original_img = detect_and_crop_faces(image_path, face_model)
    results['faces_detected'] = len(faces)
    print(f"  [FACES] Detected {len(faces)} face(s)")
    
    if len(faces) == 0:
        print("  [RESULT] No faces detected")
        return results
    
    # Create output directory for this image
    if save_crops:
        img_output_dir = Path(OUTPUT_DIR) / image_path.stem
        img_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 2: Classify age for each face
    under18_count = 0
    for i, (face, box) in enumerate(zip(faces, boxes)):
        age_class, confidence, under18_prob = classify_age(face, age_model)
        age_label = CLASS_NAMES[age_class]
        
        face_result = {
            'face_id': i + 1,
            'bounding_box': box,
            'age_class': age_label,
            'under18_probability': under18_prob,
            'confidence': confidence,
            'is_under18': age_class == 0
        }
        
        results['face_details'].append(face_result)
        
        print(f"  [FACE {i+1}] {age_label.upper()} (Under-18 prob: {under18_prob:.2%}, Confidence: {confidence:.2%})")
        print(f"            Bounding box: {box}")
        
        # Save cropped face
        if save_crops:
            crop_filename = f"face_{i+1}_{age_label}_{under18_prob:.2f}.jpg"
            crop_path = img_output_dir / crop_filename
            face.save(crop_path, quality=95)
            face_result['cropped_face_path'] = str(crop_path)
            print(f"            Saved to: {crop_path}")
        
        # Check if Under-18 with sufficient confidence
        if age_class == 0 and under18_prob >= AGE_CONF_THRESHOLD:
            under18_count += 1
    
    results['under18_faces'] = under18_count
    
    # Draw bounding boxes on original image and save
    if save_crops and original_img is not None:
        annotated_img = original_img.copy()
        for i, (box, face_detail) in enumerate(zip(boxes, results['face_details'])):
            x1, y1, x2, y2 = box
            age_label = face_detail['age_class']
            is_under18 = face_detail['is_under18']
            
            # Color: Red for Under-18, Green for Adult
            color = (255, 0, 0) if is_under18 else (0, 255, 0)
            thickness = 3
            
            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"Face {i+1}: {age_label.upper()} ({face_detail['under18_probability']:.2%})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1 - 10, label_size[1] + 10)
            cv2.rectangle(annotated_img, (x1, label_y - label_size[1] - 5), 
                         (x1 + label_size[0], label_y + 5), color, -1)
            cv2.putText(annotated_img, label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        annotated_path = img_output_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        results['annotated_image_path'] = str(annotated_path)
        print(f"  [SAVED] Annotated image: {annotated_path}")
    
    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Load models
    print("="*60)
    print("LOADING MODELS")
    print("="*60)
    
    face_model = load_face_model(FACE_MODEL_PATH)
    
    try:
        age_model = load_age_model(AGE_MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load age model: {e}")
        return
    
    # Collect all images to process
    all_image_paths = []
    
    # Add single images
    for img_path in TEST_IMAGES:
        if os.path.exists(img_path):
            all_image_paths.append(Path(img_path))
        else:
            print(f"[WARNING] Image not found: {img_path}")
    
    # Add images from folders
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
    for folder_path in TEST_FOLDERS:
        folder = Path(folder_path)
        if folder.exists() and folder.is_dir():
            image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
            all_image_paths.extend(image_files)
            print(f"[OK] Found {len(image_files)} images in {folder_path}")
        else:
            print(f"[WARNING] Folder not found: {folder_path}")
    
    if len(all_image_paths) == 0:
        print("\n[ERROR] No images found to process!")
        print("Please add folder paths to TEST_FOLDERS or image paths to TEST_IMAGES at the top of this script.")
        return
    
    # Process all images
    print("\n" + "="*60)
    print(f"PROCESSING {len(all_image_paths)} IMAGES")
    print("="*60)
    
    all_results = []
    for i, img_path in enumerate(all_image_paths, 1):
        print(f"[{i}/{len(all_image_paths)}] ", end="")
        result = process_image(img_path, face_model, age_model, save_crops=True)
        all_results.append(result)
    
    # Save results
    output_file = f"face_age_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    total = len(all_results)
    total_faces = sum(r['faces_detected'] for r in all_results)
    total_under18 = sum(r['under18_faces'] for r in all_results)
    
    print(f"Total images processed: {total}")
    print(f"Total faces detected: {total_faces}")
    print(f"Under-18 faces detected: {total_under18}")
    print(f"Adult faces detected: {total_faces - total_under18}")
    
    print(f"\nCropped faces saved to: {OUTPUT_DIR}/")
    print(f"Detailed results saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()

