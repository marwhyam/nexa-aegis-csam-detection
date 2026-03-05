"""
Utility functions for YOLO11 training
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.nn.parallel import DataParallel
from datetime import datetime

from config import DATASET_YAML, CLASS_NAMES, OUTPUT_CONFIG, MODEL_CONFIG


def check_dataset_structure(dataset_dir: Path) -> Dict:
    """
    Check and verify dataset structure.
    
    Args:
        dataset_dir: Path to dataset directory
    
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'exists': False,
        'yaml_exists': False,
        'splits': {},
        'total_images': 0,
        'total_labels': 0,
        'classes': CLASS_NAMES
    }
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return stats
    
    stats['exists'] = True
    
    # Check YAML file
    yaml_path = dataset_dir / "dataset.yaml"
    if yaml_path.exists():
        stats['yaml_exists'] = True
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            stats['yaml_data'] = yaml_data
    else:
        print(f"⚠ Warning: dataset.yaml not found at {yaml_path}")
    
    # Check splits
    for split in ['train', 'val', 'test']:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        
        split_stats = {
            'images_exist': images_dir.exists(),
            'labels_exist': labels_dir.exists(),
            'image_count': 0,
            'label_count': 0
        }
        
        if images_dir.exists():
            # Count images
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            for ext in image_extensions:
                split_stats['image_count'] += len(list(images_dir.glob(f"*{ext}")))
        
        if labels_dir.exists():
            # Count labels
            split_stats['label_count'] += len(list(labels_dir.glob("*.txt")))
        
        stats['splits'][split] = split_stats
        stats['total_images'] += split_stats['image_count']
        stats['total_labels'] += split_stats['label_count']
    
    return stats


def print_dataset_info(stats: Dict):
    """Print dataset information."""
    print("=" * 80)
    print("Dataset Structure Check")
    print("=" * 80)
    print()
    
    if not stats['exists']:
        print("❌ Dataset directory does not exist!")
        return
    
    print(f"✓ Dataset directory exists")
    print(f"✓ YAML file exists: {stats['yaml_exists']}")
    print()
    
    if stats['yaml_exists']:
        yaml_data = stats.get('yaml_data', {})
        print("Dataset Configuration:")
        print(f"  Path: {yaml_data.get('path', 'N/A')}")
        print(f"  Classes: {yaml_data.get('nc', 'N/A')}")
        print(f"  Class names: {yaml_data.get('names', {})}")
        print()
    
    print("Splits:")
    for split, split_stats in stats['splits'].items():
        print(f"  {split.upper()}:")
        print(f"    Images: {split_stats['image_count']:,} ({'✓' if split_stats['images_exist'] else '❌'})")
        print(f"    Labels: {split_stats['label_count']:,} ({'✓' if split_stats['labels_exist'] else '❌'})")
    
    print()
    print(f"Total: {stats['total_images']:,} images, {stats['total_labels']:,} labels")
    print("=" * 80)


def check_gpu_availability() -> Dict:
    """Check GPU availability and return info."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': []
    }
    
    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            device_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                'memory_allocated_gb': torch.cuda.memory_allocated(i) / (1024**3),
                'memory_reserved_gb': torch.cuda.memory_reserved(i) / (1024**3)
            }
            info['devices'].append(device_info)
    
    return info


def print_gpu_info(info: Dict):
    """Print GPU information."""
    print("=" * 80)
    print("GPU Information")
    print("=" * 80)
    print()
    
    if not info['cuda_available']:
        print("❌ CUDA not available. Training will use CPU (very slow!)")
        return
    
    print(f"✓ CUDA available")
    print(f"✓ Number of GPUs: {info['device_count']}")
    print()
    
    for device in info['devices']:
        print(f"GPU {device['id']}: {device['name']}")
        print(f"  Total Memory: {device['memory_total_gb']:.2f} GB")
        print(f"  Allocated: {device['memory_allocated_gb']:.2f} GB")
        print(f"  Reserved: {device['memory_reserved_gb']:.2f} GB")
        print()
    
    print("=" * 80)


def save_training_config(config_dict: Dict, save_path: Path):
    """Save training configuration to JSON."""
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"✓ Training configuration saved to: {save_path}")


def load_training_config(load_path: Path) -> Dict:
    """Load training configuration from JSON."""
    with open(load_path, 'r') as f:
        config = json.load(f)
    print(f"✓ Training configuration loaded from: {load_path}")
    return config


def create_output_directories(base_dir: Path, run_name: str) -> Dict[str, Path]:
    """Create output directories for training run."""
    dirs = {
        'base': base_dir / run_name,
        'weights': base_dir / run_name / "weights",
        'plots': base_dir / run_name / "plots",
        'logs': base_dir / run_name / "logs",
        'predictions': base_dir / run_name / "predictions",
        'analysis': base_dir / run_name / "analysis"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


class AttributeAccessDataParallel(DataParallel):
    """DataParallel wrapper that forwards missing attrs to the underlying module."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = super().__getattr__('module')
            return getattr(module, name)


def resolve_data_parallel_device_ids() -> List[int]:
    """Return sanitized GPU ids for DataParallel based on config and availability."""
    if not MODEL_CONFIG.get('use_data_parallel', False):
        return []
    if not torch.cuda.is_available():
        print("⚠ DataParallel requested but CUDA is unavailable. Falling back to CPU.")
        return []

    configured = MODEL_CONFIG.get('data_parallel_devices')
    if not configured:
        configured = list(range(torch.cuda.device_count()))

    resolved = []
    for dev in configured:
        try:
            idx = int(dev)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < torch.cuda.device_count():
            resolved.append(idx)

    # Remove duplicates while preserving order
    seen = set()
    filtered = []
    for idx in resolved:
        if idx not in seen:
            seen.add(idx)
            filtered.append(idx)

    if len(filtered) < 2:
        print("⚠ DataParallel requested but requires at least 2 GPUs. Running on a single GPU.")
        return []

    return filtered


def apply_data_parallel_to_module(module: nn.Module, device_ids: List[int] | None = None):
    """
    Wrap a PyTorch module with DataParallel using the provided (or resolved) devices.

    Args:
        module: The nn.Module to wrap.
        device_ids: Optional pre-resolved list of GPU ids.

    Returns:
        Tuple of (possibly wrapped module, active device list).
    """
    resolved_devices = device_ids if device_ids is not None else resolve_data_parallel_device_ids()
    if not resolved_devices:
        return module, []

    if isinstance(module, DataParallel):
        active_devices = list(module.device_ids)
        return module, active_devices

    primary_device = resolved_devices[0]
    module.to(torch.device(f'cuda:{primary_device}'))
    dp_model = AttributeAccessDataParallel(module, device_ids=resolved_devices, output_device=primary_device)
    return dp_model, resolved_devices


def enable_data_parallel_if_configured(yolo_wrapper):
    """
    Wrap the underlying PyTorch model with DataParallel when requested.

    Args:
        yolo_wrapper: Ultralytics YOLO wrapper instance.

    Returns:
        Tuple of (YOLO wrapper, list of active GPU ids used for DataParallel).
    """
    module = getattr(yolo_wrapper, 'model', None)
    if module is None:
        raise AttributeError("YOLO wrapper is missing the underlying 'model' attribute needed for DataParallel.")

    dp_model, active_devices = apply_data_parallel_to_module(module)
    if active_devices:
        yolo_wrapper.model = dp_model
        setattr(yolo_wrapper, 'data_parallel_device_ids', active_devices)
        print(f"✓ Wrapped YOLO backbone with DataParallel on GPUs: {active_devices}")
    else:
        setattr(yolo_wrapper, 'data_parallel_device_ids', [])

    return yolo_wrapper, active_devices


if __name__ == "__main__":
    # Test functions
    print("Testing utility functions...")
    print()
    
    # Check dataset
    from config import DATASET_DIR
    stats = check_dataset_structure(DATASET_DIR)
    print_dataset_info(stats)
    print()
    
    # Check GPU
    gpu_info = check_gpu_availability()
    print_gpu_info(gpu_info)

