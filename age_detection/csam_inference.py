#!/usr/bin/env python3
"""
Complete CSAM Detection Pipeline
NSFW Detection (YOLO11) → Face Detection (YOLOv8) → Age Classification → CSAM Flagging

================================================================================
HOW TO USE:
================================================================================
1. Edit the paths below (TEST_FOLDERS and TEST_IMAGES)
2. Run: python csam_inference.py
3. Results saved to: csam_results_TIMESTAMP.json

Example:
    TEST_FOLDERS = [
        "C:/images/folder1/",
        "D:/test_images/",
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

# Model paths - UPDATE THESE TO YOUR MODEL LOCATIONS
NSFW_MODEL_PATH = "nsfw.pt"  # YOLO11 NSFW segmentation model (detects body parts)
FACE_MODEL_PATH = "yolov8l.pt"  # YOLOv8 face detection model
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

# Videos to test (optional)
# Videos are processed frame-by-frame (configurable stride below)
TEST_VIDEOS = [
    # "path/to/video1.mp4",
    # Add more videos here...
]

# Video processing settings
VIDEO_FRAME_STRIDE = 5  # process every Nth frame to save time
VIDEO_MAX_FRAMES = None  # set to an int to cap frames per video, or None for all

# Age detection settings
IMAGE_SIZE = 584
CLASS_NAMES = ['under18', 'adult']

# Confidence thresholds
NSFW_CONF_THRESHOLD = 0.25  # NSFW body part detection confidence
FACE_CONF_THRESHOLD = 0.25  # Face detection confidence
AGE_CONF_THRESHOLD = 0.6  # Age classification confidence for Under-18

# NSFW classes (from YOLO11 model - body parts)
NSFW_CLASSES = {
    0: 'anus',
    1: 'breast',
    2: 'female_genital',
    3: 'male_genital'
}

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_nsfw_model(model_path):
    """Load YOLO11 NSFW segmentation model"""
    if not os.path.exists(model_path):
        print(f"[WARNING] NSFW model not found: {model_path}")
        print("  Continuing without NSFW detection (will process all images)")
        return None
    
    try:
        model = YOLO(model_path)
        print(f"[OK] NSFW model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load NSFW model: {e}")
        return None

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

def detect_nsfw(image_path, nsfw_model, image_np=None):
    """
    Detect NSFW content using YOLO11 segmentation model.
    Returns: (is_nsfw, confidence, detected_classes, blurred_rgb_image)
    If NSFW is detected, the detected regions are blurred and the blurred RGB image is returned.
    """
    if nsfw_model is None:
        return True, 0.0, [], None  # Skip NSFW check if model not available
    
    try:
        # Prepare input for YOLO11: either path or numpy image (BGR)
        if image_np is not None:
            results = nsfw_model(image_np, conf=NSFW_CONF_THRESHOLD, verbose=False)
        else:
            results = nsfw_model(str(image_path), conf=NSFW_CONF_THRESHOLD, verbose=False)
        
        if len(results) == 0 or results[0].boxes is None:
            return False, 0.0, [], None
        
        boxes = results[0].boxes
        detected_classes = []
        max_conf = 0.0
        
        # Load image once for blurring
        if image_np is not None:
            img_bgr = image_np.copy()
        else:
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                return False, 0.0, [], None
        
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            if cls in NSFW_CLASSES:
                detected_classes.append(NSFW_CLASSES[cls])
                max_conf = max(max_conf, conf)
                
                # Blur the detected NSFW region
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_bgr.shape[1], x2); y2 = min(img_bgr.shape[0], y2)
                roi = img_bgr[y1:y2, x1:x2]
                if roi.size > 0:
                    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                    img_bgr[y1:y2, x1:x2] = blurred_roi
        
        is_nsfw = len(detected_classes) > 0
        blurred_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if is_nsfw else None
        return is_nsfw, max_conf, detected_classes, blurred_rgb
        
    except Exception as e:
        print(f"  [ERROR] NSFW detection failed: {e}")
        return False, 0.0, [], None

def detect_faces(image_path, face_model, override_image=None):
    """Detect and crop faces from image using YOLOv8"""
    try:
        if override_image is not None:
            # override_image is expected in RGB
            img_rgb = override_image.copy()
            h, w = img_rgb.shape[:2]
        else:
            img = cv2.imread(str(image_path))
            if img is None:
                return [], []
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
                # Faces are typically wider than tall, aspect ratio ~0.7-1.3
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
        
        return cropped_faces, bounding_boxes
    except Exception as e:
        print(f"  [ERROR] Face detection failed: {e}")
        return [], []

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
# MAIN PIPELINE
# ============================================================================

def process_image(image_path, nsfw_model, face_model, age_model):
    """
    Complete pipeline: NSFW → Face Detection → Age Classification → CSAM
    
    Returns:
        dict with results
    """
    image_path = Path(image_path)
    results = {
        'media_type': 'image',
        'image_path': str(image_path),
        'timestamp': datetime.now().isoformat(),
        'nsfw_detected': False,
        'nsfw_confidence': 0.0,
        'nsfw_classes': [],
        'faces_detected': 0,
        'under18_faces': 0,
        'csam_detected': False,
        'face_details': []
    }
    
    # Step 1: NSFW Detection
    print(f"\nProcessing: {image_path.name}")
    is_nsfw, nsfw_conf, nsfw_classes, blurred_rgb = detect_nsfw(image_path, nsfw_model)
    results['nsfw_detected'] = is_nsfw
    results['nsfw_confidence'] = nsfw_conf
    results['nsfw_classes'] = nsfw_classes
    
    if nsfw_model is None:
        print(f"  [NSFW] Check skipped (model not available)")
    elif is_nsfw:
        print(f"  [NSFW] Detected: {', '.join(nsfw_classes)} (confidence: {nsfw_conf:.2%})")
    else:
        print(f"  [NSFW] Not detected (confidence: {nsfw_conf:.2%})")
        # If not NSFW, no need to check for CSAM (only if NSFW model is available)
        if nsfw_model is not None:
            return results
    
    # Step 2: Face Detection (only if NSFW or NSFW model not available)
    # If NSFW detected and we have a blurred image, run face detection on blurred image
    faces, boxes = detect_faces(image_path, face_model, override_image=blurred_rgb if is_nsfw else None)
    results['faces_detected'] = len(faces)
    print(f"  [FACES] Detected {len(faces)} face(s)")
    
    if len(faces) == 0:
        print("  [RESULT] No faces detected - Cannot determine CSAM")
        return results
    
    # Step 3: Age Classification for each face
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
        
        # Check if Under-18 with sufficient confidence
        if age_class == 0 and under18_prob >= AGE_CONF_THRESHOLD:
            under18_count += 1
    
    results['under18_faces'] = under18_count
    
    # Step 4: CSAM Decision
    if under18_count > 0:
        results['csam_detected'] = True
        print(f"  [CSAM] DETECTED - {under18_count} Under-18 face(s) found in NSFW content")
    else:
        print(f"  [CSAM] Not detected - All faces are adults or low confidence")
    
    return results


def process_video(video_path, nsfw_model, face_model, age_model, frame_stride=VIDEO_FRAME_STRIDE, max_frames=VIDEO_MAX_FRAMES):
    """
    Process a video frame-by-frame.
    Returns a summary dict with per-frame counts and flagged frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    processed = 0
    nsfw_frames = 0
    csam_frames = 0
    under18_frames = 0
    frame_results = []
    
    print(f"\nProcessing video: {Path(video_path).name} | {total_frames} frames | {width}x{height} | {fps:.1f} fps")
    
    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        # Apply stride
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        
        if max_frames is not None and processed >= max_frames:
            break
        
        # Run pipeline on this frame (without disk I/O)
        is_nsfw, nsfw_conf, nsfw_classes, blurred_rgb = detect_nsfw(
            video_path, nsfw_model, image_np=frame_bgr)
        
        # If NSFW model is missing, treat as not-checked and continue to faces
        if nsfw_model is None:
            is_nsfw = True  # force face + age check when NSFW model not available
        
        if is_nsfw:
            nsfw_frames += 1
        
        faces, boxes = detect_faces(
            video_path, face_model, override_image=blurred_rgb if is_nsfw else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        under18_count = 0
        face_details = []
        for i, (face, box) in enumerate(zip(faces, boxes)):
            age_class, confidence, under18_prob = classify_age(face, age_model)
            age_label = CLASS_NAMES[age_class]
            face_details.append({
                'face_id': i + 1,
                'bounding_box': box,
                'age_class': age_label,
                'under18_probability': under18_prob,
                'confidence': confidence,
                'is_under18': age_class == 0
            })
            if age_class == 0 and under18_prob >= AGE_CONF_THRESHOLD:
                under18_count += 1
        
        if under18_count > 0:
            under18_frames += 1
        if under18_count > 0 and is_nsfw:
            csam_frames += 1
        
        frame_results.append({
            'frame_index': frame_idx,
            'nsfw_detected': bool(is_nsfw),
            'nsfw_confidence': nsfw_conf,
            'nsfw_classes': nsfw_classes,
            'faces_detected': len(faces),
            'under18_faces': under18_count,
            'csam_detected': bool(under18_count > 0 and is_nsfw),
            'face_details': face_details,
        })
        
        processed += 1
        frame_idx += 1
    
    cap.release()
    
    summary = {
        'media_type': 'video',
        'video_path': str(video_path),
        'timestamp': datetime.now().isoformat(),
        'total_frames': total_frames,
        'processed_frames': processed,
        'frame_stride': frame_stride,
        'fps': fps,
        'width': width,
        'height': height,
        'nsfw_frames': nsfw_frames,
        'under18_frames': under18_frames,
        'csam_frames': csam_frames,
        'frames': frame_results
    }
    
    print(f"  [VIDEO SUMMARY] processed={processed}/{total_frames}, nsfw_frames={nsfw_frames}, under18_frames={under18_frames}, csam_frames={csam_frames}")
    return summary

# ============================================================================
# MAIN - NO NEED TO EDIT BELOW THIS LINE
# ============================================================================

def main():
    # Load models
    print("="*60)
    print("LOADING MODELS")
    print("="*60)
    
    nsfw_model = load_nsfw_model(NSFW_MODEL_PATH)
    face_model = load_face_model(FACE_MODEL_PATH)
    
    try:
        age_model = load_age_model(AGE_MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load age model: {e}")
        return
    
    # Collect media to process
    all_image_paths = []
    all_video_paths = []
    
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
    
    # Add videos
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    for vid_path in TEST_VIDEOS:
        if os.path.exists(vid_path) and Path(vid_path).suffix.lower() in video_extensions:
            all_video_paths.append(Path(vid_path))
        else:
            print(f"[WARNING] Video not found or unsupported: {vid_path}")
    
    if len(all_image_paths) == 0 and len(all_video_paths) == 0:
        print("\n[ERROR] No media found to process!")
        print("Please add paths to TEST_FOLDERS / TEST_IMAGES / TEST_VIDEOS at the top of this script.")
        return
    
    all_results = []
    
    # Process images
    if len(all_image_paths) > 0:
        print("\n" + "="*60)
        print(f"PROCESSING {len(all_image_paths)} IMAGES")
        print("="*60)
        for i, img_path in enumerate(all_image_paths, 1):
            print(f"[IMG {i}/{len(all_image_paths)}] ", end="")
            result = process_image(img_path, nsfw_model, face_model, age_model)
            all_results.append(result)
    
    # Process videos
    if len(all_video_paths) > 0:
        print("\n" + "="*60)
        print(f"PROCESSING {len(all_video_paths)} VIDEOS")
        print("="*60)
        for i, vid_path in enumerate(all_video_paths, 1):
            print(f"[VID {i}/{len(all_video_paths)}] ", end="")
            vid_result = process_video(vid_path, nsfw_model, face_model, age_model)
            if vid_result is not None:
                all_results.append(vid_result)
    
    # Save results
    output_file = f"csam_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary (images + videos)
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    img_results = [r for r in all_results if r.get('media_type') == 'image']
    vid_results = [r for r in all_results if r.get('media_type') == 'video']
    
    # Images summary
    if img_results:
        total = len(img_results)
        nsfw_count = sum(1 for r in img_results if r['nsfw_detected'])
        csam_count = sum(1 for r in img_results if r['csam_detected'])
        total_faces = sum(r['faces_detected'] for r in img_results)
        total_under18 = sum(r['under18_faces'] for r in img_results)
        
        print(f"[IMAGES] Total: {total}, NSFW: {nsfw_count} ({nsfw_count/total*100:.1f}%), Faces: {total_faces}, Under-18: {total_under18}, CSAM: {csam_count}")
        if csam_count > 0:
            print("  [ALERT] CSAM in images:")
            for r in img_results:
                if r['csam_detected']:
                    print(f"    - {r['image_path']} ({r['under18_faces']} Under-18 face(s))")
    
    # Videos summary
    if vid_results:
        total_v = len(vid_results)
        csam_v = sum(1 for r in vid_results if r['csam_frames'] > 0)
        print(f"[VIDEOS] Total: {total_v}, CSAM flagged videos: {csam_v}")
        for r in vid_results:
            print(f"  - {r['video_path']}: frames processed={r['processed_frames']}, nsfw_frames={r['nsfw_frames']}, under18_frames={r['under18_frames']}, csam_frames={r['csam_frames']}")
    
    print(f"\nDetailed results saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()
