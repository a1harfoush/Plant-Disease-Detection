@echo off
echo ============================================================
echo Kaggle API Setup Helper
echo ============================================================
echo.

set /p USERNAME="Enter your Kaggle username: "

echo Creating kaggle.json...
(
echo {
echo   "username": "%USERNAME%",
echo   "key": "KGAT_5915bcd8b367f31a3d4fd05b633b0b99"
echo }
) > kaggle.json

echo.
echo Moving to %USERPROFILE%\.kaggle\...
if not exist "%USERPROFILE%\.kaggle" mkdir "%USERPROFILE%\.kaggle"
move /Y kaggle.json "%USERPROFILE%\.kaggle\kaggle.json"

echo.
echo ============================================================
echo Kaggle API Setup Complete!
echo ============================================================
echo.
echo Testing connection...
kaggle --version
if %errorlevel% equ 0 (
    echo.
    echo Success! You can now download datasets.
    echo Try: kaggle datasets list
) else (
    echo.
    echo Please install kaggle: pip install kaggle
)
echo.
pause
