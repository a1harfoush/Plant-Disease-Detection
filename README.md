# 🌱 Plant Disease Detection - Fine-Tuning ResNet50

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project for plant disease classification using transfer learning with ResNet50. Classifies leaf images into 4 disease categories with 85-95% accuracy.

## 🎯 Overview

This project fine-tunes a pre-trained ResNet50 model to classify plant leaf images into:
- **Healthy** - No disease
- **Bacterial Blight** - Bacterial infection
- **Leaf Spot** - Fungal spots
- **Rust** - Rust disease

## ✨ Features

- 🚀 Transfer learning with ResNet50 (ImageNet pre-trained)
- 📓 Interactive Jupyter notebook with step-by-step training
- 📊 Comprehensive visualizations (training curves, confusion matrix, predictions)
- ⚙️ Easy configuration via `config.py`
- 🔧 Automated dataset organization
- 📈 Real-time training progress tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 8GB RAM (16GB recommended)
- GPU with CUDA (optional but recommended)
- 5GB free disk space

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Setup Kaggle API:**
   - Create account at [kaggle.com](https://www.kaggle.com)
   - Go to Account → API → "Create New API Token"
   - Place `kaggle.json` in:
     - Windows: `C:\Users\<YourUsername>\.kaggle\`
     - Linux/Mac: `~/.kaggle/`

3. **Download and organize dataset:**
```bash
kaggle datasets download -d abdallahalidev/plantvillage-dataset
python organize_dataset.py
```

4. **Run training:**
```bash
jupyter notebook plant_disease_finetuning.ipynb
```

## 📊 Results

After training (15 epochs):
- **Validation Accuracy**: 85-95%
- **Training Time**: 10-15 min (GPU) / 2-3 hours (CPU)
- **Model Size**: ~90 MB

### Output Files
- `outputs/best_model.pth` - Trained model weights
- `outputs/training_results.png` - Loss/accuracy curves + confusion matrix
- `outputs/sample_predictions.png` - Visual predictions

## 🏗️ Model Architecture

- **Base**: ResNet50 (ImageNet pre-trained)
- **Frozen Layers**: conv1, bn1, layer1, layer2, layer3
- **Trainable Layers**: layer4 + custom classification head
- **Custom Head**: 2048 → 512 → 128 → 4 classes

### Training Configuration
- **Optimizer**: AdamW with differential learning rates
- **Scheduler**: Cosine Annealing
- **Loss**: CrossEntropyLoss with label smoothing
- **Regularization**: Dropout, BatchNorm, Data Augmentation

## 🔧 Customization

Edit `config.py` to modify:
```python
TRAINING_CONFIG = {
    "num_epochs": 15,      # Number of training epochs
    "batch_size": 32,      # Batch size
    "lr_head": 1e-3,       # Learning rate for classification head
    "lr_backbone": 1e-5,   # Learning rate for backbone
}
```

## 📁 Project Structure

```
Plant-Disease-Detection/
├── plant_disease_finetuning.ipynb  # Main notebook
├── plant_disease_finetuning.py     # Python script version
├── organize_dataset.py             # Dataset preparation
├── config.py                       # Configuration
├── requirements.txt                # Dependencies
├── setup.bat                       # Windows setup
├── setup.sh                        # Linux/Mac setup
├── setup_kaggle.bat                # Kaggle API setup
├── README.md                       # This file
├── data/                           # Dataset (train/val/test)
└── outputs/                        # Results & model
```

## 🎓 What You'll Learn

- Transfer learning concepts
- Fine-tuning pre-trained models
- PyTorch workflows
- Data augmentation strategies
- Model evaluation techniques
- Hyperparameter tuning

## 🐛 Troubleshooting

### CUDA Out of Memory
Reduce batch size in `config.py`:
```python
TRAINING_CONFIG = {"batch_size": 16}
```

### Kaggle API Error
- Verify `kaggle.json` location
- Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`
- Run `setup_kaggle.bat` (Windows)

### Low Accuracy
- Increase epochs: `"num_epochs": 25`
- Unfreeze more layers: `"unfreeze_layer": "layer3"`
- Adjust learning rates
- Add more training data

## 📈 Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | 85-95% |
| Training Time (GPU) | 10-15 min |
| Training Time (CPU) | 2-3 hours |
| Model Size | ~90 MB |
| Inference Time | <50ms per image |

## 📖 References

- [ResNet Paper](https://arxiv.org/abs/1512.03385) - Deep Residual Learning
- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## 🤝 Contributing

Contributions are welcome! Feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- PlantVillage dataset contributors
- PyTorch and torchvision teams
- ResNet authors (He et al.)

---

**Made with ❤️ for the agricultural AI community**
