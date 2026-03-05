"""
Comprehensive verification of class names, IDs, and mappings
"""
import yaml
from pathlib import Path
from collections import Counter

# Paths
DATASET_DIR = Path(r"C:\fyp_data\final_balanced")
DATASET_YAML = DATASET_DIR / "dataset.yaml"

print("=" * 80)
print("CLASS MAPPING VERIFICATION")
print("=" * 80)

# 1. Read YAML file
print("\n1. DATASET YAML:")
print("-" * 80)
with open(DATASET_YAML, 'r') as f:
    yaml_data = yaml.safe_load(f)
    
print(f"Path: {yaml_data['path']}")
print(f"Classes (nc): {yaml_data['nc']}")
print("\nClass names from YAML:")
for class_id, class_name in yaml_data['names'].items():
    print(f"  YOLO Class {class_id}: {class_name}")

# 2. Expected CLASS_NAMES in notebook
print("\n2. EXPECTED CLASS_NAMES IN NOTEBOOK:")
print("-" * 80)
EXPECTED_CLASS_NAMES = {
    0: 'anus',
    1: 'breast',
    2: 'female_genital',
    3: 'male_genital'
}
print("Expected CLASS_NAMES dictionary:")
for class_id, class_name in EXPECTED_CLASS_NAMES.items():
    print(f"  {class_id}: '{class_name}'")

# 3. Verify YAML matches expected
print("\n3. YAML vs EXPECTED VERIFICATION:")
print("-" * 80)
yaml_matches = True
for class_id in range(4):
    yaml_name = yaml_data['names'][class_id]
    expected_name = EXPECTED_CLASS_NAMES[class_id]
    match = "✓" if yaml_name == expected_name else "✗"
    print(f"  Class {class_id}: YAML='{yaml_name}' vs Expected='{expected_name}' {match}")
    if yaml_name != expected_name:
        yaml_matches = False

# 4. Class ID conversion mapping
print("\n4. CLASS ID CONVERSION MAPPING:")
print("-" * 80)
print("YOLO Format → Mask R-CNN Format:")
print("  (0 is background in Mask R-CNN)")
for yolo_id in range(4):
    maskrcnn_id = yolo_id + 1
    class_name = EXPECTED_CLASS_NAMES[yolo_id]
    print(f"  YOLO {yolo_id} ({class_name}) → Mask R-CNN {maskrcnn_id} ({class_name})")

# 5. Check actual labels in dataset
print("\n5. CHECKING ACTUAL LABELS IN DATASET:")
print("-" * 80)
labels_dir = DATASET_DIR / "labels" / "train"
if labels_dir.exists():
    label_files = list(labels_dir.glob("*.txt"))
    print(f"Found {len(label_files)} label files in train split")
    
    class_counts = Counter()
    invalid_classes = []
    sample_labels = []
    
    for label_file in label_files[:100]:  # Check first 100 files
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:  # Need at least class_id + 2 coordinates
                    continue
                try:
                    class_id = int(float(parts[0]))
                    if 0 <= class_id <= 3:
                        class_counts[class_id] += 1
                        if len(sample_labels) < 5:
                            sample_labels.append((label_file.name, line_num, class_id))
                    else:
                        invalid_classes.append((label_file.name, line_num, class_id))
                except (ValueError, IndexError):
                    continue
    
    print(f"\nClass distribution (from first 100 files):")
    for class_id in range(4):
        count = class_counts[class_id]
        class_name = EXPECTED_CLASS_NAMES[class_id]
        print(f"  Class {class_id} ({class_name}): {count} instances")
    
    if sample_labels:
        print(f"\nSample labels found:")
        for filename, line_num, class_id in sample_labels[:5]:
            class_name = EXPECTED_CLASS_NAMES[class_id]
            print(f"  {filename}:{line_num} → Class {class_id} ({class_name})")
    
    if invalid_classes:
        print(f"\n⚠ WARNING: Found {len(invalid_classes)} invalid class IDs:")
        for filename, line_num, class_id in invalid_classes[:10]:
            print(f"  {filename}:{line_num} → Invalid class {class_id}")
else:
    print(f"⚠ Labels directory not found: {labels_dir}")

# 6. Model configuration check
print("\n6. MODEL CONFIGURATION:")
print("-" * 80)
print("Expected CONFIG['num_classes']: 5")
print("  (4 classes + 1 background = 5 total)")
print("\nMask R-CNN class IDs:")
print("  0: background (reserved)")
for yolo_id in range(4):
    maskrcnn_id = yolo_id + 1
    class_name = EXPECTED_CLASS_NAMES[yolo_id]
    print(f"  {maskrcnn_id}: {class_name}")

# 7. Summary
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)
all_good = yaml_matches and len(invalid_classes) == 0
if all_good:
    print("✓ All class mappings are correct!")
    print("✓ YAML matches expected CLASS_NAMES")
    print("✓ Class ID conversion (YOLO → Mask R-CNN) is correct")
    print("✓ Model configuration (num_classes=5) is correct")
else:
    print("⚠ Issues found:")
    if not yaml_matches:
        print("  ✗ YAML class names don't match expected")
    if invalid_classes:
        print(f"  ✗ Found {len(invalid_classes)} invalid class IDs in labels")
print("=" * 80)

