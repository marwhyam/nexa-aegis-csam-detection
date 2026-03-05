#!/usr/bin/env python3
"""
CSAM Detection Dashboard
Simple web interface for NSFW detection, face detection, and age classification

Run: streamlit run dashboard.py
"""

import streamlit as st
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

# ============================================================================
# CONFIGURATION
# ============================================================================

NSFW_MODEL_PATH = "nsfw.pt"  # YOLO11 NSFW segmentation model
FACE_MODEL_PATH = "C:/age-detection/yolov8l-face.pt"  # YOLOv8 face detection model
AGE_MODEL_PATH = "phase2_best_model.pt"  # Phase 2 age detection model
EFFICIENTNET_WEIGHTS_PATH = "efficientnet_b3_imagenet_weights.pth"  # Local ImageNet weights (optional)

IMAGE_SIZE = 584
CLASS_NAMES = ['under18', 'adult']

# Confidence thresholds
NSFW_CONF_THRESHOLD = 0.25
FACE_CONF_THRESHOLD = 0.25
AGE_CONF_THRESHOLD = 0.6
VIDEO_FRAME_STRIDE = 5  # process every Nth frame
VIDEO_MAX_FRAMES = None  # set an int to cap frames, or None for all

# NSFW classes
NSFW_CLASSES = {
    0: 'anus',
    1: 'breast',
    2: 'female_genital',
    3: 'male_genital'
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_nsfw_model(model_path):
    """Load YOLO11 NSFW segmentation model"""
    if not os.path.exists(model_path):
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load NSFW model: {e}")
        return None

@st.cache_resource
def load_age_model(model_path):
    """Load age detection model (Phase 2 EfficientNet-B3)"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Age model not found: {model_path}")
    
    # Build model architecture (without downloading ImageNet weights)
    model = models.efficientnet_b3(weights=None)  # No download needed
    
    # Replace classifier (same as training)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=False),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(512, 2)
    )
    
    # Load your trained weights (should contain full model including backbone)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if isinstance(state_dict, dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
    # Load the weights (your phase2 model should have all weights)
    model.load_state_dict(state_dict, strict=False)  # strict=False allows missing keys
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_face_model(model_path):
    """Load YOLOv8 face detection model"""
    if not os.path.exists(model_path):
        st.warning(f"Face model not found: {model_path}. Using default YOLOv8.")
        model = YOLO('yolov8l.pt')
    else:
        model = YOLO(model_path)
    return model

# ============================================================================
# PREPROCESSING
# ============================================================================

age_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def detect_nsfw(image, nsfw_model, return_blurred=True):
    """
    Detect NSFW content using YOLO11 segmentation model.
    image: PIL image (RGB) or numpy array (RGB or BGR).
    Returns: (is_nsfw, max_conf, detected_classes, blurred_rgb or None)
    """
    if nsfw_model is None:
        return True, 0.0, [], None  # Skip check if model not available
    
    try:
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image.copy()
            if img_bgr.shape[2] == 3:
                pass
        results = nsfw_model(img_bgr, conf=NSFW_CONF_THRESHOLD, verbose=False)
        
        if len(results) == 0 or results[0].boxes is None:
            return False, 0.0, [], None
        
        boxes = results[0].boxes
        detected_classes = []
        max_conf = 0.0
        
        blurred_bgr = img_bgr.copy()
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            if cls in NSFW_CLASSES:
                detected_classes.append(NSFW_CLASSES[cls])
                max_conf = max(max_conf, conf)
                
                if return_blurred:
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(blurred_bgr.shape[1], x2); y2 = min(blurred_bgr.shape[0], y2)
                    roi = blurred_bgr[y1:y2, x1:x2]
                    if roi.size > 0:
                        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                        blurred_bgr[y1:y2, x1:x2] = blurred_roi
        
        is_nsfw = len(detected_classes) > 0
        blurred_rgb = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB) if (return_blurred and is_nsfw) else None
        return is_nsfw, max_conf, detected_classes, blurred_rgb
        
    except Exception as e:
        st.error(f"NSFW detection failed: {e}")
        return False, 0.0, [], None

def detect_faces(image, face_model, override_image=None):
    """Detect and crop faces from image using YOLOv8"""
    try:
        if override_image is not None:
            img_rgb = override_image.copy()
        else:
            img_array = np.array(image)
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) if len(img_array.shape) == 3 else img_array
        h, w = img_rgb.shape[:2]
        
        results = face_model(img_rgb, conf=FACE_CONF_THRESHOLD, verbose=False)
        
        cropped_faces = []
        bounding_boxes = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                width = x2 - x1
                height = y2 - y1
                if height > 0:
                    aspect_ratio = width / height
                    if 0.5 <= aspect_ratio <= 1.5:
                        padding_x = int(width * 0.1)
                        padding_y = int(height * 0.1)
                        
                        x1 = max(0, x1 - padding_x)
                        y1 = max(0, y1 - padding_y)
                        x2 = min(w, x2 + padding_x)
                        y2 = min(h, y2 + padding_y)
                        
                        face_crop = img_rgb[y1:y2, x1:x2]
                        
                        if face_crop.size > 0 and face_crop.shape[0] >= 32 and face_crop.shape[1] >= 32:
                            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            face_pil = Image.fromarray(face_rgb)
                            cropped_faces.append(face_pil)
                            bounding_boxes.append((x1, y1, x2, y2))
        
        return cropped_faces, bounding_boxes
    except Exception as e:
        st.error(f"Face detection failed: {e}")
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
        st.error(f"Age classification failed: {e}")
        return 1, 0.0, 0.0

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="CSAM Detection Dashboard",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 CSAM Detection Dashboard")
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading models..."):
        nsfw_model = load_nsfw_model(NSFW_MODEL_PATH)
        face_model = load_face_model(FACE_MODEL_PATH)
        age_model = load_age_model(AGE_MODEL_PATH)
    
    st.success("Models loaded successfully!")
    
    tab_image, tab_video = st.tabs(["Image", "Video"])
    
    # ---------------------------------------------------------------------
    # IMAGE TAB
    # ---------------------------------------------------------------------
    with tab_image:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to analyze for NSFW content and CSAM detection"
        )
    
    if uploaded_file is not None and uploaded_file.type.startswith("image"):
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            
            # Initialize variables
            faces = []
            boxes = []
            face_results = []
            under18_count = 0
            csam_detected = False
            
            # Step 1: NSFW Detection
            with st.spinner("Checking for NSFW content..."):
                is_nsfw, nsfw_conf, nsfw_classes, blurred_rgb = detect_nsfw(image, nsfw_model, return_blurred=True)
            
            if nsfw_model is None:
                st.warning("⚠️ NSFW model not available. Skipping NSFW check.")
                is_nsfw = True  # Process anyway
            elif is_nsfw:
                st.error(f"🚨 **NSFW DETECTED**")
                st.write(f"**Confidence:** {nsfw_conf:.2%}")
                st.write(f"**Detected classes:** {', '.join(nsfw_classes)}")
            else:
                st.success("✅ **Not NSFW**")
                st.write(f"**Confidence:** {(1-nsfw_conf):.2%}")
            
            st.markdown("---")

            # Show blurred NSFW view if applicable
            if is_nsfw and blurred_rgb is not None:
                st.subheader("Blurred NSFW View")
                st.image(blurred_rgb, use_container_width=True, caption="Sensitive regions are blurred")
                st.markdown("---")
            
            # Step 2: Face Detection (only if NSFW or NSFW model not available)
            if is_nsfw or nsfw_model is None:
                with st.spinner("Detecting faces..."):
                    faces, boxes = detect_faces(image, face_model, override_image=blurred_rgb)
                
                if len(faces) == 0:
                    st.warning("⚠️ No faces detected")
                    st.info("**Result:** Cannot determine CSAM (no faces found)")
                else:
                    st.success(f"✅ **{len(faces)} face(s) detected**")
                    
                    # Step 3: Age Classification
                    st.markdown("### Face Analysis")
                    
                    for i, (face, box) in enumerate(zip(faces, boxes)):
                        with st.spinner(f"Analyzing face {i+1}/{len(faces)}..."):
                            age_class, confidence, under18_prob = classify_age(face, age_model)
                        
                        age_label = CLASS_NAMES[age_class].upper()
                        is_under18 = age_class == 0
                        
                        # Store result
                        face_results.append({
                            'age_class': age_class,
                            'under18_prob': under18_prob,
                            'is_under18': is_under18
                        })
                        
                        # Display face crop
                        st.image(face, caption=f"Face {i+1}: {age_label}", width=200)
                        
                        # Display results
                        if is_under18:
                            st.write(f"**Face {i+1}:** 🚨 **UNDER-18**")
                            st.write(f"- Under-18 probability: {under18_prob:.2%}")
                            st.write(f"- Confidence: {confidence:.2%}")
                            if under18_prob >= AGE_CONF_THRESHOLD:
                                under18_count += 1
                        else:
                            st.write(f"**Face {i+1}:** ✅ **ADULT**")
                            st.write(f"- Under-18 probability: {under18_prob:.2%}")
                            st.write(f"- Confidence: {confidence:.2%}")
                        
                        st.write(f"- Bounding box: {box}")
                        st.markdown("---")
                    
                    # Final CSAM Decision
                    st.markdown("### Final Result")
                    if under18_count > 0:
                        csam_detected = True
                        st.error("🚨 **CSAM DETECTED**")
                        st.write(f"**{under18_count} Under-18 face(s)** found in NSFW content")
                    else:
                        st.success("✅ **Not CSAM**")
                        st.write("All detected faces are adults or low confidence")
            else:
                st.info("ℹ️ **Not NSFW** - No further analysis needed")
        
        # Draw annotated image (only if we have face results)
        if len(face_results) > 0 and len(boxes) > 0:
            st.markdown("---")
            st.subheader("Annotated Image")
            
            # Draw bounding boxes on blurred image if NSFW, else original
            if is_nsfw and blurred_rgb is not None:
                annotated_img = blurred_rgb.copy()
            else:
                annotated_img = np.array(image).copy()
            
            for i, (box, face_result) in enumerate(zip(boxes, face_results)):
                x1, y1, x2, y2 = box
                
                is_under18 = face_result['is_under18']
                under18_prob = face_result['under18_prob']
                
                color = (255, 0, 0) if is_under18 else (0, 255, 0)
                thickness = 3
                
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
                
                label = f"Face {i+1}: {'UNDER-18' if is_under18 else 'ADULT'} ({under18_prob:.2%})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y1 - 10, label_size[1] + 10)
                cv2.rectangle(annotated_img, (x1, label_y - label_size[1] - 5), 
                             (x1 + label_size[0], label_y + 5), color, -1)
                cv2.putText(annotated_img, label, (x1, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            st.image(annotated_img, use_container_width=True)
    
    else:
        st.info("👆 Please upload an image to begin analysis")
    
    # ---------------------------------------------------------------------
    # VIDEO TAB
    # ---------------------------------------------------------------------
    with tab_video:
        st.subheader("Upload Video")
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'],
            help="Upload a video to analyze frame-by-frame"
        )
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            run_video = st.button("Analyze Video")
            
            if run_video:
                # Save to temp file for OpenCV
                with st.spinner("Saving video..."):
                    tmp_path = Path("tmp_upload_video.mp4")
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded_video.read())
                
                cap = cv2.VideoCapture(str(tmp_path))
                if not cap.isOpened():
                    st.error("Failed to open video.")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    nsfw_frames = 0
                    csam_frames = 0
                    under18_frames = 0
                    processed = 0
                    
                    annotated_sample = None
                    
                    progress = st.progress(0.0)
                    status = st.empty()
                    
                    frame_idx = 0
                    max_frames = VIDEO_MAX_FRAMES if VIDEO_MAX_FRAMES is not None else total_frames
                    
                    while True:
                        ret, frame_bgr = cap.read()
                        if not ret:
                            break
                        
                        if frame_idx % VIDEO_FRAME_STRIDE != 0:
                            frame_idx += 1
                            continue
                        if processed >= max_frames:
                            break
                        
                        is_nsfw, nsfw_conf, nsfw_classes, blurred_rgb = detect_nsfw(frame_bgr, nsfw_model, return_blurred=True)
                        if nsfw_model is None:
                            is_nsfw = True  # force downstream processing
                        
                        faces, boxes = detect_faces(frame_bgr, face_model, override_image=blurred_rgb if is_nsfw else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                        
                        under18_count = 0
                        face_details = []
                        for i, (face, box) in enumerate(zip(faces, boxes)):
                            age_class, confidence, under18_prob = classify_age(face, age_model)
                            is_under18 = age_class == 0 and under18_prob >= AGE_CONF_THRESHOLD
                            if is_under18:
                                under18_count += 1
                            face_details.append((box, is_under18, under18_prob))
                        
                        if is_nsfw:
                            nsfw_frames += 1
                        if under18_count > 0:
                            under18_frames += 1
                        if under18_count > 0 and is_nsfw:
                            csam_frames += 1
                        
                        # store first annotated frame
                        if annotated_sample is None and len(faces) > 0:
                            ann = blurred_rgb if (is_nsfw and blurred_rgb is not None) else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                            for box, is_under18, prob in face_details:
                                x1, y1, x2, y2 = box
                                color = (255, 0, 0) if is_under18 else (0, 255, 0)
                                cv2.rectangle(ann, (x1, y1), (x2, y2), color, 3)
                                label = f"{'U18' if is_under18 else 'Adult'} ({prob:.2%})"
                                cv2.putText(ann, label, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            annotated_sample = ann
                        
                        processed += 1
                        frame_idx += 1
                        progress.progress(min(frame_idx / max(total_frames, 1), 1.0))
                        status.text(f"Processed {processed} frames (stride {VIDEO_FRAME_STRIDE})")
                    
                    cap.release()
                    tmp_path.unlink(missing_ok=True)
                    
                    st.markdown("### Video Summary")
                    st.write(f"Frames processed: {processed}/{total_frames}")
                    st.write(f"NSFW frames: {nsfw_frames}")
                    st.write(f"Under-18 frames: {under18_frames}")
                    st.write(f"CSAM frames: {csam_frames}")
                    
                    if annotated_sample is not None:
                        st.markdown("#### Sample annotated frame")
                        st.image(annotated_sample, use_container_width=True)
    
    # Sidebar info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This dashboard performs:
        1. **NSFW Detection** - Detects adult content
        2. **Face Detection** - Finds faces in images
        3. **Age Classification** - Classifies faces as Under-18 or Adult
        4. **CSAM Detection** - Flags if Under-18 faces found in NSFW content
        """)
        
        st.header("Settings")
        st.write(f"**Device:** {DEVICE}")
        st.write(f"**NSFW Model:** {'Loaded' if nsfw_model else 'Not available'}")
        st.write(f"**Face Model:** Loaded")
        st.write(f"**Age Model:** Loaded")
        
        st.header("Thresholds")
        st.write(f"NSFW Confidence: {NSFW_CONF_THRESHOLD}")
        st.write(f"Face Confidence: {FACE_CONF_THRESHOLD}")
        st.write(f"Age Confidence: {AGE_CONF_THRESHOLD}")

if __name__ == "__main__":
    main()

