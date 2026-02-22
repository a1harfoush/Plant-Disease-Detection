# Complete Guide: Plant Disease Detection Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dataset Setup](#dataset-setup)
4. [Running the Project](#running-the-project)
5. [Understanding the Code](#understanding-the-code)
6. [Customization](#customization)
7. [Results Interpretation](#results-interpretation)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)

---

## Project Overview

### What Does This Project Do?
This project trains a deep learning model to classify plant diseases from leaf images. It uses transfer learning with a pre-trained ResNet50 model, fine-tuning it for 4 disease categories.

### Why Transfer Learning?
- Faster training (10-15 min vs hours)
- Better accuracy with limited data
- Leverages features learned from millions of ImageNet images
- Only requires training ~10% of parameters

### Project Files Explained

| File | Purpose |
|------|---------|
| `plant_disease_finetuning.ipynb` | Main notebook - run this! |
| `plant_disease_finetuning.py` | Script version (same as notebook) |
| `organize_dataset.py` | Prepares dataset structure |
| `config.py` | Easy configuration editing |
| `setup.bat` / `setup.sh` | Automated setup scripts |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |
| `QUICKSTART.md` | Fast setup guide |

---

## Installation

### Prerequisites
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- GPU with CUDA (optional but recommended)
- 5GB free disk space

### Step-by-Step Installation

#### 1. Install Python Packages
```bash
pip install -r requirements.txt
```

This installs:
- PyTorch (deep learning framework)
- torchvision (computer vision utilities)
- numpy (numerical computing)
- matplotlib (visualization)
- scikit-learn (metrics)
- kaggle (dataset download)
- jupyter (notebook interface)

#### 2. Setup Kaggle API

**Why?** To download the PlantVillage dataset.

**Steps:**
1. Create account at [kaggle.com](https://www.kaggle.com)
2. Go to: Account → API → "Create New API Token"
3. Download `kaggle.json`
4. Place it in:
   - Windows: `C:\Users\<YourUsername>\.kaggle\`
   - Linux/Mac: `~/.kaggle/`
5. Linux/Mac only: `chmod 600 ~/.kaggle/kaggle.json`

#### 3. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
kaggle --version
```

---

## Dataset Setup

### Option 1: Automated Setup (Recommended)

**Windows:**
```cmd
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

This will:
1. Download PlantVillage dataset (~1.5GB)
2. Extract images
3. Organize into train/val/test splits
4. Create proper folder structure

### Option 2: Manual Setup

#### Download Dataset
```bash
kaggle datasets download -d abdallahalidev/plantvillage-dataset
```

#### Extract
```bash
# Windows PowerShell
Expand-Archive plantvillage-dataset.zip -DestinationPath plantvillage_data

# Linux/Mac
unzip plantvillage-dataset.zip -d plantvillage_data
```

#### Organize Dataset
```bash
python organize_dataset.py
```

### Expected Folder Structure
```
data/
├── train/
│   ├── Healthy/
│   ├── Bacterial_Blight/
│   ├── Leaf_Spot/
│   └── Rust/
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

### Verify Dataset
```bash
python -c "from pathlib import Path; print(f'Train images: {len(list(Path('data/train').rglob('*.jpg')))}')"
```

---

## Running the Project

### Method 1: Jupyter Notebook (Recommended for Learning)

```bash
jupyter notebook plant_disease_finetuning.ipynb
```

**Advantages:**
- Interactive execution
- See outputs immediately
- Easy to modify and experiment
- Great for learning

**How to use:**
1. Click on each cell
2. Press Shift+Enter to run
3. Wait for output before moving to next cell
4. Read the markdown explanations

### Method 2: Python Script (For Production)

```bash
python plant_disease_finetuning.py
```

**Advantages:**
- Runs start to finish automatically
- No browser needed
- Easy to schedule/automate
- Better for remote servers

### What Happens During Training?

1. **Initialization** (1 min)
   - Loads pre-trained ResNet50
   - Freezes early layers
   - Replaces classification head

2. **Training Loop** (10-15 min with GPU)
   - 15 epochs
   - Each epoch: train on all training data, validate
   - Saves best model based on validation accuracy

3. **Evaluation** (1 min)
   - Tests on held-out test set
   - Generates classification report
   - Creates visualizations

4. **Output Generation**
   - Saves model weights
   - Creates training curves
   - Generates sample predictions

---

## Understanding the Code

### Key Concepts

#### 1. Transfer Learning
```python
# Load pre-trained model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Freeze early layers (keep ImageNet features)
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 (adapt to our task)
for param in model.layer4.parameters():
    param.requires_grad = True
```

#### 2. Custom Classification Head
```python
model.fc = nn.Sequential(
    nn.Linear(2048, 512),      # Reduce dimensions
    nn.BatchNorm1d(512),       # Normalize
    nn.ReLU(),                 # Activation
    nn.Dropout(0.4),           # Regularization
    nn.Linear(512, 128),       # Further reduction
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 4),         # Final output (4 classes)
)
```

#### 3. Differential Learning Rates
```python
optimizer = optim.AdamW([
    {"params": backbone_params, "lr": 1e-5},  # Small lr for backbone
    {"params": head_params, "lr": 1e-3},      # Larger lr for new head
])
```

Why? The backbone already knows good features, so we update it slowly. The new head needs to learn from scratch, so we use a larger learning rate.

#### 4. Data Augmentation
```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),     # Mirror image
    transforms.RandomRotation(30),         # Rotate ±30°
    transforms.ColorJitter(...),           # Vary colors
    transforms.RandomErasing(),            # Occlude parts
    transforms.Normalize(...),             # Standardize
])
```

Why? Creates variations of training images, helping the model generalize better.

### Training Loop Explained

```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()          # Reset gradients
        outputs = model(images)        # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                # Backward pass
        optimizer.step()               # Update weights
    
    # Validation phase
    model.eval()
    with torch.no_grad():              # No gradient computation
        for images, labels in val_loader:
            outputs = model(images)
            # Compute accuracy
```

---

## Customization

### Easy Modifications

#### Change Number of Epochs
Edit `config.py`:
```python
TRAINING_CONFIG = {
    "num_epochs": 20,  # Increase for better accuracy
    ...
}
```

#### Reduce Batch Size (if out of memory)
```python
TRAINING_CONFIG = {
    "batch_size": 16,  # Or 8 for very limited memory
    ...
}
```

#### Unfreeze More Layers
```python
MODEL_CONFIG = {
    "unfreeze_layer": "layer3",  # More trainable parameters
    ...
}
```

### Advanced Modifications

#### Try Different Architecture
```python
# In the notebook/script
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
# Adjust the head accordingly
```

#### Add More Classes
1. Organize data with new class folders
2. Update `config.py`:
```python
MODEL_CONFIG = {
    "num_classes": 6,
    "class_names": ["Healthy", "Blight", "Spot", "Rust", "Mildew", "Mosaic"],
}
```

#### Implement Early Stopping
```python
patience = 5
best_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    # ... training code ...
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

---

## Results Interpretation

### Training Curves

#### Loss Curve
- Should decrease over time
- Training loss < validation loss is normal
- Large gap = overfitting (add regularization)

#### Accuracy Curve
- Should increase over time
- Validation accuracy plateaus = model converged
- Oscillating = learning rate too high

### Confusion Matrix

```
              Predicted
           H   B   L   R
Actual H  [90  2   1   0]
       B  [1  85  3   2]
       L  [0   4  88  1]
       R  [0   1   2  89]
```

- Diagonal = correct predictions
- Off-diagonal = misclassifications
- Look for patterns (which classes confuse the model?)

### Classification Report

```
                  precision  recall  f1-score
Healthy              0.96     0.97     0.96
Bacterial_Blight     0.92     0.93     0.93
Leaf_Spot            0.94     0.95     0.94
Rust                 0.97     0.97     0.97
```

- Precision: Of predicted X, how many were actually X?
- Recall: Of actual X, how many did we find?
- F1-score: Harmonic mean of precision and recall

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce batch_size to 16 or 8
- Reduce img_size to 128
- Close other GPU applications
- Use CPU (slower but works)

#### 2. Kaggle Authentication Error
```
401 - Unauthorized
```
**Solutions:**
- Check kaggle.json location
- Verify file permissions (Linux/Mac)
- Re-download API token from Kaggle

#### 3. No Images Found
```
FileNotFoundError: data/train not found
```
**Solutions:**
- Run `organize_dataset.py`
- Check PlantVillage extraction path
- Verify folder structure matches expected

#### 4. Low Accuracy (<70%)
**Possible causes:**
- Not enough training data
- Learning rate too high/low
- Insufficient epochs
- Poor data quality

**Solutions:**
- Increase num_epochs to 25-30
- Unfreeze more layers (layer3)
- Adjust learning rates
- Add more data augmentation

#### 5. Training Too Slow
**Solutions:**
- Use GPU instead of CPU
- Increase batch_size (if memory allows)
- Reduce img_size to 128
- Use fewer epochs for testing

---

## Advanced Topics

### 1. Mixed Precision Training
Faster training with less memory:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in train_loader:
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. Learning Rate Finder
Find optimal learning rate:
```python
from torch_lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot()
```

### 3. Grad-CAM Visualization
See what the model looks at:
```python
from pytorch_grad_cam import GradCAM

cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
grayscale_cam = cam(input_tensor=image)
# Overlay on original image
```

### 4. Model Deployment
Convert to ONNX for production:
```python
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")
```

### 5. Ensemble Methods
Combine multiple models:
```python
models = [model1, model2, model3]
predictions = []

for model in models:
    pred = model(image)
    predictions.append(pred)

final_pred = torch.stack(predictions).mean(dim=0)
```

---

## Additional Resources

### Documentation
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

### Papers
- ResNet: [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- Transfer Learning: [How transferable are features?](https://arxiv.org/abs/1411.1792)

### Datasets
- [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- [Plant Pathology 2021](https://www.kaggle.com/c/plant-pathology-2021-fgvc8)

---

## Summary Checklist

- [ ] Python 3.8+ installed
- [ ] All packages installed (`pip install -r requirements.txt`)
- [ ] Kaggle API configured
- [ ] Dataset downloaded and organized
- [ ] Training completed successfully
- [ ] Results visualized and analyzed
- [ ] Model saved in outputs/

**Congratulations!** You've completed the plant disease detection project. 🎉

For questions or issues, refer to the troubleshooting section or check the README.md.
