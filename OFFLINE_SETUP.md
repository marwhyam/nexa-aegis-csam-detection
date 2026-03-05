# Offline Setup Guide for CSAM Detection System

## Files to Download for Offline Device

### 1. Model Files (Required)
Download these model files and place them in the same directory as the scripts:

- **NSFW Model**: `nsfw.pt`
  - Your NSFW detection model weights
  - Place in project root directory

- **Face Detection Model**: `yolov8n.pt`
  - YOLOv8 nano model (smallest, fastest)
  - Can download from: https://github.com/ultralytics/assets/releases
  - Or let the script download automatically on first run (requires internet)

- **Age Detection Model**: `phase2_best_model.pt`
  - Your Phase 2 fine-tuned age detection model
  - From: `/kaggle/working/phase2/final_export/phase2_best_model.pt`
  - Place in project root directory

### 2. Python Packages (Install with Internet, then copy to offline device)

#### Option A: Using pip download (Recommended)
```bash
# On internet-connected machine:
mkdir packages
pip download -r requirements.txt -d packages

# Copy packages/ folder to offline device
# On offline device:
pip install --no-index --find-links packages -r requirements.txt
```

#### Option B: Using conda-pack (if using conda)
```bash
# On internet machine:
conda pack -n your_env_name -o environment.tar.gz

# Copy to offline device and extract:
mkdir offline_env
tar -xzf environment.tar.gz -C offline_env
source offline_env/bin/activate
```

### 3. Requirements File
Create `requirements.txt` with:
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
```

### 4. Directory Structure
```
project/
├── csam_inference.py          # Main inference script
├── nexaaegis.ipynb            # Face detection notebook
├── nsfw.pt                    # NSFW model
├── yolov8n.pt                 # Face detection model
├── phase2_best_model.pt       # Age detection model
├── requirements.txt           # Python dependencies
└── packages/                  # Downloaded packages (for offline install)
```

## Quick Start (Offline)

1. **Install Python packages** (if not already installed):
   ```bash
   pip install --no-index --find-links packages -r requirements.txt
   ```

2. **Run inference on single image**:
   ```bash
   python csam_inference.py --image path/to/image.jpg
   ```

3. **Run inference on folder**:
   ```bash
   python csam_inference.py --folder path/to/images/
   ```

4. **Save cropped faces**:
   ```bash
   python csam_inference.py --image image.jpg --save-crops
   ```

## Model Paths Configuration

Edit `csam_inference.py` and update these paths:
```python
NSFW_MODEL_PATH = "nsfw.pt"
FACE_MODEL_PATH = "yolov8n.pt"
AGE_MODEL_PATH = "phase2_best_model.pt"
```

## Notes

- **YOLOv8**: If `yolov8n.pt` is not found, the script will try to download it (requires internet). For fully offline use, download it first.
- **CUDA**: If you have NVIDIA GPU, install CUDA-enabled PyTorch for faster inference
- **Memory**: Models require ~2-3GB RAM total
- **Speed**: 
  - CPU: ~2-5 seconds per image
  - GPU: ~0.5-1 second per image

## Troubleshooting

1. **Model not found errors**: Check file paths in `csam_inference.py`
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Import errors**: Install missing packages from `packages/` folder

