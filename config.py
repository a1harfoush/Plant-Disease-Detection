"""
Configuration file for Plant Disease Detection
Modify these settings to customize your training
"""

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_CONFIG = {
    # Number of output classes
    "num_classes": 4,
    
    # Class names (must match folder names in data/)
    "class_names": ["Healthy", "Bacterial_Blight", "Leaf_Spot", "Rust"],
    
    # Which ResNet layer to start unfreezing from
    # Options: "layer4" (default), "layer3" (more trainable), "layer2" (even more)
    "unfreeze_layer": "layer4",
    
    # Dropout rate in the classification head
    "dropout_rate": 0.4,
}

# ============================================================
# TRAINING CONFIGURATION
# ============================================================
TRAINING_CONFIG = {
    # Number of training epochs
    "num_epochs": 15,
    
    # Batch size (reduce if CUDA out of memory)
    "batch_size": 32,
    
    # Learning rate for the new classification head
    "lr_head": 1e-3,
    
    # Learning rate for unfrozen backbone layers (should be smaller)
    "lr_backbone": 1e-5,
    
    # Weight decay for regularization
    "weight_decay": 1e-4,
    
    # Label smoothing (0.0 = no smoothing, 0.1 = default)
    "label_smoothing": 0.1,
    
    # Gradient clipping max norm
    "grad_clip_norm": 1.0,
}

# ============================================================
# DATA CONFIGURATION
# ============================================================
DATA_CONFIG = {
    # Input image size (ResNet expects 224x224)
    "img_size": 224,
    
    # Data directory paths
    "train_dir": "data/train",
    "val_dir": "data/val",
    "test_dir": "data/test",
    
    # Number of data loading workers (0 = main process only)
    "num_workers": 0,
    
    # Pin memory for faster GPU transfer
    "pin_memory": False,
}

# ============================================================
# AUGMENTATION CONFIGURATION
# ============================================================
AUGMENTATION_CONFIG = {
    # Random horizontal flip probability
    "horizontal_flip": True,
    
    # Random vertical flip probability
    "vertical_flip": True,
    
    # Random rotation range in degrees
    "rotation_degrees": 30,
    
    # Color jitter parameters
    "brightness": 0.3,
    "contrast": 0.3,
    "saturation": 0.2,
    "hue": 0.1,
    
    # Random erasing probability
    "random_erasing_prob": 0.2,
}

# ============================================================
# OUTPUT CONFIGURATION
# ============================================================
OUTPUT_CONFIG = {
    # Directory to save outputs
    "output_dir": "outputs",
    
    # Model checkpoint filename
    "model_filename": "best_model.pth",
    
    # Plot filenames
    "training_plot": "training_results.png",
    "predictions_plot": "sample_predictions.png",
    
    # DPI for saved plots
    "plot_dpi": 150,
}

# ============================================================
# DEVICE CONFIGURATION
# ============================================================
DEVICE_CONFIG = {
    # Device to use ("cuda", "cpu", or "auto")
    "device": "auto",  # auto = use CUDA if available
    
    # Random seed for reproducibility
    "seed": 42,
    
    # Enable cudnn deterministic mode
    "deterministic": True,
}

# ============================================================
# HELPER FUNCTION
# ============================================================
def get_full_config():
    """Merge all configs into a single dictionary"""
    config = {}
    config.update(MODEL_CONFIG)
    config.update(TRAINING_CONFIG)
    config.update(DATA_CONFIG)
    config.update(AUGMENTATION_CONFIG)
    config.update(OUTPUT_CONFIG)
    config.update(DEVICE_CONFIG)
    return config


# ============================================================
# PRESET CONFIGURATIONS
# ============================================================

# Fast training preset (for testing)
FAST_CONFIG = {
    "num_epochs": 5,
    "batch_size": 64,
    "img_size": 128,
}

# High accuracy preset (longer training)
HIGH_ACCURACY_CONFIG = {
    "num_epochs": 30,
    "unfreeze_layer": "layer3",
    "lr_head": 5e-4,
    "dropout_rate": 0.5,
}

# Low memory preset (for limited GPU)
LOW_MEMORY_CONFIG = {
    "batch_size": 8,
    "img_size": 224,
    "num_workers": 0,
}


if __name__ == "__main__":
    # Print current configuration
    print("="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    
    config = get_full_config()
    for key, value in config.items():
        print(f"{key:25s}: {value}")
