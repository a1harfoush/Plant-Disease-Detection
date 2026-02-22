"""
Dataset Organization Script for PlantVillage
Organizes downloaded PlantVillage images into train/val/test splits
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# Set seed for reproducibility
random.seed(42)

# Configuration
SOURCE_DIR = "plantvillage_data"  # Adjust based on extracted folder name
TARGET_DIR = "data"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Class mapping - adjust based on your PlantVillage structure
# This maps PlantVillage folder names to our 4 categories
CLASS_MAPPING = {
    "Healthy": [
        "Tomato_healthy",
        "Potato_healthy",
        "Pepper_bell_healthy",
    ],
    "Bacterial_Blight": [
        "Tomato_Bacterial_spot",
        "Pepper_bell_Bacterial_spot",
    ],
    "Leaf_Spot": [
        "Tomato_Septoria_leaf_spot",
        "Tomato_Leaf_Mold",
    ],
    "Rust": [
        "Tomato_Early_blight",
        "Tomato_Late_blight",
    ]
}


def create_directory_structure():
    """Create train/val/test directories for each class"""
    for split in ['train', 'val', 'test']:
        for category in CLASS_MAPPING.keys():
            path = Path(TARGET_DIR) / split / category
            path.mkdir(parents=True, exist_ok=True)
    print("✓ Directory structure created")


def organize_images():
    """Organize images from PlantVillage into our structure"""
    source_path = Path(SOURCE_DIR)
    
    if not source_path.exists():
        print(f"Error: {SOURCE_DIR} not found!")
        print("Please download and extract the PlantVillage dataset first.")
        return
    
    for target_class, source_classes in CLASS_MAPPING.items():
        all_images = []
        
        # Collect all images for this target class
        for source_class in source_classes:
            source_class_path = source_path / source_class
            if source_class_path.exists():
                images = list(source_class_path.glob("*.jpg")) + \
                        list(source_class_path.glob("*.JPG")) + \
                        list(source_class_path.glob("*.png"))
                all_images.extend(images)
                print(f"  Found {len(images)} images in {source_class}")
        
        if not all_images:
            print(f"Warning: No images found for {target_class}")
            continue
        
        # Split into train/val/test
        train_imgs, temp_imgs = train_test_split(
            all_images, 
            test_size=(VAL_RATIO + TEST_RATIO),
            random_state=42
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=TEST_RATIO/(VAL_RATIO + TEST_RATIO),
            random_state=42
        )
        
        # Copy images to target directories
        for split, images in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            target_dir = Path(TARGET_DIR) / split / target_class
            for img_path in images:
                shutil.copy2(img_path, target_dir / img_path.name)
        
        print(f"✓ {target_class}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")


def print_summary():
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        total = 0
        for category in CLASS_MAPPING.keys():
            path = Path(TARGET_DIR) / split / category
            count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.png")))
            print(f"  {category:20s}: {count:4d} images")
            total += count
        print(f"  {'TOTAL':20s}: {total:4d} images")


if __name__ == "__main__":
    print("="*60)
    print("PlantVillage Dataset Organization")
    print("="*60)
    print(f"\nSource: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    print(f"Split: {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / {TEST_RATIO:.0%} test\n")
    
    create_directory_structure()
    organize_images()
    print_summary()
    
    print("\n✅ Dataset organization complete!")
    print("You can now run the training notebook/script.")
