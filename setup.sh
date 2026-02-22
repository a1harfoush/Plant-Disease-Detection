#!/bin/bash

echo "============================================================"
echo "Plant Disease Detection - Automated Setup"
echo "============================================================"
echo ""

echo "Step 1: Installing Python dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi
echo "Done!"
echo ""

echo "Step 2: Checking Kaggle API setup..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "WARNING: kaggle.json not found!"
    echo "Please:"
    echo "  1. Go to kaggle.com -> Account -> API -> Create New Token"
    echo "  2. Place kaggle.json in: ~/.kaggle/"
    echo "  3. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo "  4. Run this script again"
    exit 1
fi
chmod 600 ~/.kaggle/kaggle.json
echo "Kaggle API configured!"
echo ""

echo "Step 3: Downloading PlantVillage dataset..."
kaggle datasets download -d abdallahalidev/plantvillage-dataset
if [ $? -ne 0 ]; then
    echo "Error: Failed to download dataset"
    echo "Make sure kaggle.json is properly configured"
    exit 1
fi
echo "Done!"
echo ""

echo "Step 4: Extracting dataset..."
unzip -q plantvillage-dataset.zip -d plantvillage_data
echo "Done!"
echo ""

echo "Step 5: Organizing dataset into train/val/test splits..."
python organize_dataset.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to organize dataset"
    exit 1
fi
echo "Done!"
echo ""

echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "You can now:"
echo "  1. Run Jupyter Notebook: jupyter notebook plant_disease_finetuning.ipynb"
echo "  2. Or run Python script: python plant_disease_finetuning.py"
echo ""
