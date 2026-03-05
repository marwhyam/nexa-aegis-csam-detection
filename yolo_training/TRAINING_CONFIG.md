# YOLO11 Training Configuration Summary

## Hardware Setup
- **GPUs**: 3x RTX A2000 (12GB VRAM each)
- **Total VRAM**: 36GB
- **Multi-GPU**: Enabled (automatic distribution)

## Model Configuration
- **Model**: YOLO11-detect (yolo11n.pt or yolo11s.pt)
- **Task**: Detection only (boxes, no masks)
- **Image Size**: 640x640 (optimal for 3x A2000)
- **Total Batch Size**: 24 (8 per GPU × 3 GPUs)
- **Workers**: 6 per GPU (18 total)

## Training Parameters
- **Epochs**: 100
- **Early Stopping**: 20 epochs patience
- **Learning Rate**: 0.01 (initial)
- **Optimizer**: SGD
- **Momentum**: 0.937
- **Weight Decay**: 0.0005

## Why These Settings?

### Batch Size (24 total, 8 per GPU)
- RTX A2000 12GB can handle ~8-12 images per batch at 640px
- With 3 GPUs, total batch of 24 is optimal
- Larger batches = more stable training, faster convergence
- If OOM occurs, reduce to 18 (6 per GPU) or 12 (4 per GPU)

### Image Size (640)
- Good balance between accuracy and speed
- Fits comfortably in 12GB VRAM with batch size 8
- Can increase to 800 if you have headroom, but slower
- 1024 is possible but may need smaller batch size

### Multi-GPU Setup
- YOLO11 automatically distributes batches across GPUs
- Device: [0, 1, 2] uses all 3 GPUs
- DataParallel is handled internally by Ultralytics
- Each GPU processes batch_size/3 images

## Grad-CAM Configuration
- **Enabled**: Yes
- **Target Layer**: model.22 (YOLO11 detection head)
- **Samples per Epoch**: 4
- **Save Directory**: runs/gradcam/

## Performance Expectations

### Training Speed
- **With 3x A2000**: ~2-4 hours for 100 epochs (depends on dataset size)
- **Per epoch**: ~2-3 minutes (with 31,928 training images)
- **Batch processing**: ~0.5-1 second per batch

### Memory Usage
- **Per GPU**: ~8-10GB VRAM (with batch 8, image 640)
- **Total**: ~24-30GB across 3 GPUs
- **Headroom**: ~6GB total for safety

## Optimization Tips

### If You Get OOM (Out of Memory)
1. Reduce batch size: 24 → 18 → 12
2. Reduce image size: 640 → 512
3. Reduce workers: 6 → 4 per GPU
4. Use gradient accumulation (not in current config)

### For Better Accuracy
1. Use yolo11s.pt instead of yolo11n.pt
2. Increase image size: 640 → 800 (reduce batch to 18)
3. Train for more epochs: 100 → 150
4. Increase augmentation

### For Faster Training
1. Use yolo11n.pt (smaller model)
2. Keep image size at 640
3. Increase batch size if possible: 24 → 30
4. Reduce workers if CPU-bound: 6 → 4

## Comparison with Mask R-CNN

| Feature | YOLO11-detect | Mask R-CNN |
|---------|---------------|------------|
| Training Speed | ✅ Much Faster | Slower |
| Inference Speed | ✅ Real-time | Slower |
| Memory Usage | ✅ Lower | Higher |
| Accuracy | Good | Slightly Better |
| Masks | ❌ Boxes only | ✅ Pixel masks |
| Multi-GPU | ✅ Automatic | Manual setup |
| Deployment | ✅ Easier | More complex |

## Files Modified for Multi-GPU

1. **config.py**: 
   - `device: [0, 1, 2]` (3 GPUs)
   - `batch: 24` (total, auto-split)
   - `workers: 6` (per GPU)

2. **train.py**: 
   - Automatically handles multi-GPU
   - Prints per-GPU batch size

3. **gradcam.py**: 
   - New file for Grad-CAM visualization
   - Works with YOLO11 architecture

4. **evaluate.py**: 
   - Integrated Grad-CAM generation
   - Saves visualizations automatically

## Next Steps

1. ✅ Config optimized for 3x A2000
2. ✅ Grad-CAM added
3. ✅ Multi-GPU enabled
4. ⏭️ Run: `python train.py`
5. ⏭️ Monitor GPU usage: `nvidia-smi`
6. ⏭️ Check Grad-CAM outputs in `runs/gradcam/`

