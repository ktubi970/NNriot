@echo off
REM ============================================================================
REM NNriot launcher — Web app or continuous trainer
REM Reads .env for RIOT_API_KEY, HOST, PORT, etc. (see .env.example)
REM ============================================================================

setlocal
cd /d "%~dp0"

REM --- Pick a Python interpreter that has the project deps ----------------
REM Prefer Anaconda (where flask + tensorflow are installed). Fall back to
REM system python if Anaconda isn't found.
set PYTHON=
if exist "%USERPROFILE%\anaconda3\python.exe" set PYTHON=%USERPROFILE%\anaconda3\python.exe
if not defined PYTHON if exist "%USERPROFILE%\miniconda3\python.exe" set PYTHON=%USERPROFILE%\miniconda3\python.exe
if not defined PYTHON set PYTHON=python

set PYTHONIOENCODING=utf-8

REM --- Optional env vars (override below or set in shell before launching) ---
REM   set NNRIOT_DB_PATH=D:\projet\NNriot\training_data.db
REM   set NNRIOT_BATCH_SIZE=100
REM   set NNRIOT_EPOCHS=1
REM   set NNRIOT_FETCH_TIMELINES=1
REM   set MOCK_OCR_ENABLED=1
REM   set CORS_ORIGINS=http://localhost:5000,http://127.0.0.1:5000
REM   set HOST=127.0.0.1
REM   set PORT=5000

REM --- Menu ---------------------------------------------------------------
if "%~1"=="" goto :menu
if /i "%~1"=="web" goto :run_web
if /i "%~1"=="train" goto :run_train
if /i "%~1"=="backfill" goto :run_backfill
echo Unknown action: %~1
goto :usage

:menu
echo.
echo ============================================================
echo   NNriot Launcher
echo ============================================================
echo   Python: %PYTHON%
echo.
echo   [1] Web app       (run.py — http://localhost:5000)
echo   [2] Trainer       (continuous_trainer.py — loops every 10 min)
echo   [3] Backfill labels (backfill_labels.py — one-shot)
echo   [Q] Quit
echo ============================================================
set /p CHOICE="Choose an action [1]: "
if "%CHOICE%"=="" set CHOICE=1
if /i "%CHOICE%"=="1" goto :run_web
if /i "%CHOICE%"=="2" goto :run_train
if /i "%CHOICE%"=="3" goto :run_backfill
if /i "%CHOICE%"=="q" goto :end
echo Invalid choice.
goto :menu

:run_web
echo.
echo Starting Flask web app...
"%PYTHON%" run.py
goto :after

:run_train
echo.
echo Starting continuous trainer (Ctrl-C to stop)...
"%PYTHON%" continuous_trainer.py
goto :after

:run_backfill
echo.
echo Running backfill_labels.py...
"%PYTHON%" backfill_labels.py
goto :after

:usage
echo.
echo Usage: start.bat [web^|train^|backfill]
echo   No arg = interactive menu
echo.
exit /b 1

:after
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
    echo.
    echo Process exited with code %EXITCODE%.
    pause
)
exit /b %EXITCODE%

:end
endlocal
