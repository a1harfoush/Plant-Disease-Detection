@echo off
echo ============================================================
echo Plant Disease Detection - Automated Setup
echo ============================================================
echo.

echo Step 1: Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)
echo Done!
echo.

echo Step 2: Checking Kaggle API setup...
if not exist "%USERPROFILE%\.kaggle\kaggle.json" (
    echo WARNING: kaggle.json not found!
    echo Please:
    echo   1. Go to kaggle.com -^> Account -^> API -^> Create New Token
    echo   2. Place kaggle.json in: %USERPROFILE%\.kaggle\
    echo   3. Run this script again
    pause
    exit /b 1
)
echo Kaggle API configured!
echo.

echo Step 3: Downloading PlantVillage dataset...
kaggle datasets download -d abdallahalidev/plantvillage-dataset
if %errorlevel% neq 0 (
    echo Error: Failed to download dataset
    echo Make sure kaggle.json is properly configured
    pause
    exit /b 1
)
echo Done!
echo.

echo Step 4: Extracting dataset...
powershell -command "Expand-Archive -Path plantvillage-dataset.zip -DestinationPath plantvillage_data -Force"
echo Done!
echo.

echo Step 5: Organizing dataset into train/val/test splits...
python organize_dataset.py
if %errorlevel% neq 0 (
    echo Error: Failed to organize dataset
    pause
    exit /b 1
)
echo Done!
echo.

echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo You can now:
echo   1. Run Jupyter Notebook: jupyter notebook plant_disease_finetuning.ipynb
echo   2. Or run Python script: python plant_disease_finetuning.py
echo.
pause
