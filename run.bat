@echo off
REM Thai Sign Language -> Text launcher
REM   run.bat          : setup (first time), train if no model, then start app
REM   run.bat train    : re-extract landmarks + retrain (after recording new samples)
REM   run.bat --camera 1
REM MediaPipe cannot load its models from a non-ASCII (Thai) path,
REM so the virtual environment lives in %USERPROFILE%\venvs\tsl
set VENV=%USERPROFILE%\venvs\tsl
set PY=%VENV%\Scripts\python.exe
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

if not exist "%PY%" (
    echo [setup] creating Python 3.10 virtual environment...
    py -3.10 -m venv "%VENV%" || goto :error
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install "mediapipe==0.10.21" "numpy<2" pillow pygrabber || goto :error
    "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :error
)

"%PY%" -c "import pygrabber" 2>nul || "%PY%" -m pip install pygrabber -q

if "%1"=="train" goto :train
if not exist "models\sign_model.pt" goto :train
goto :app

:train
echo [1/2] extracting landmarks from GIF dataset...
"%PY%" extract_landmarks.py || goto :error
echo [2/2] training model...
"%PY%" train.py || goto :error
if "%1"=="train" goto :eof

:app
"%PY%" app.py %*
goto :eof

:error
echo ERROR - see messages above
pause
