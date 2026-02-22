# Quick Start Guide

## Complete Setup in 5 Steps

### Step 1: Install Dependencies
```bash
pip install torch torchvision numpy matplotlib scikit-learn kaggle jupyter
```

### Step 2: Setup Kaggle API
1. Go to [kaggle.com](https://www.kaggle.com) → Account → API → Create New Token
2. Place `kaggle.json` in:
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\`
   - **Linux/Mac**: `~/.kaggle/`
3. **Linux/Mac only**: `chmod 600 ~/.kaggle/kaggle.json`

### Step 3: Download Dataset
```bash
# Download PlantVillage dataset
kaggle datasets download -d abdallahalidev/plantvillage-dataset

# Extract it
# Windows (PowerShell):
Expand-Archive plantvillage-dataset.zip -DestinationPath plantvillage_data

# Linux/Mac:
unzip plantvillage-dataset.zip -d plantvillage_data
```

### Step 4: Organize Dataset
```bash
python organize_dataset.py
```

This will create the proper folder structure:
```
data/
  train/
    Healthy/
    Bacterial_Blight/
    Leaf_Spot/
    Rust/
  val/ ...
  test/ ...
```

### Step 5: Run Training

#### Option A: Jupyter Notebook (Recommended)
```bash
jupyter notebook plant_disease_finetuning.ipynb
```
Then run all cells sequentially.

#### Option B: Python Script
```bash
python plant_disease_finetuning.py
```

## Expected Output

After training completes, you'll find in the `outputs/` folder:
- `best_model.pth` - Trained model weights
- `training_results.png` - Loss/accuracy curves and confusion matrix
- `sample_predictions.png` - Visual predictions on test images

## Training Time Estimates

- **With GPU (CUDA)**: ~10-15 minutes for 15 epochs
- **CPU only**: ~2-3 hours for 15 epochs

## Troubleshooting

### "kaggle: command not found"
```bash
pip install --upgrade kaggle
```

### "401 Unauthorized" from Kaggle
- Verify `kaggle.json` is in the correct location
- Check file permissions (Linux/Mac)

### "CUDA out of memory"
Edit the CONFIG in the notebook/script:
```python
CONFIG = {
    ...
    "batch_size": 16,  # Reduce from 32
    ...
}
```

### No images found
- Check that `plantvillage_data` folder exists
- Verify the folder structure matches what's in `organize_dataset.py`
- You may need to adjust `CLASS_MAPPING` in `organize_dataset.py`

## Quick Test (Without Real Data)

Want to test the pipeline first? The original Python script includes synthetic data generation. Just run:
```bash
python plant_disease_finetuning.py
```

It will generate synthetic leaf images automatically - no dataset download needed!

## Next Steps

After successful training:
1. Check `outputs/training_results.png` for model performance
2. Review the classification report in the notebook output
3. Try predicting on your own leaf images (see README.md for code)
4. Experiment with hyperparameters to improve accuracy

## Need Help?

- Check the full README.md for detailed documentation
- Review the notebook comments for step-by-step explanations
- Ensure all dependencies are installed correctly
