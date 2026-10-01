@echo off
setlocal

for %%I in ("%~dp0..\..") do set "REPO_ROOT=%%~fI"

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] conda was not found. Run this file from Miniconda Prompt.
    exit /b 1
)

call conda activate lerobot
if errorlevel 1 exit /b 1

cd /d "%REPO_ROOT%"
python -m pip install -e ".[feetech,hardware]"
if errorlevel 1 exit /b 1

lerobot-info
if errorlevel 1 exit /b 1

python -m unittest discover -v -s examples\eai_course\week4\task2 -p "test_*.py"
if errorlevel 1 exit /b 1

echo.
echo Course environment and offline tests are ready.
