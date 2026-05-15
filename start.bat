@echo off
REM Launch the NNriot Flask web app.
REM Reads .env for RIOT_API_KEY, HOST, PORT, etc. — see .env.example.

setlocal
cd /d "%~dp0"

REM Optional env vars (see MULTI_OUTPUT_MODEL_PLAN.md):
REM   set NNRIOT_BATCH_SIZE=100
REM   set NNRIOT_EPOCHS=1
REM   set NNRIOT_DB_PATH=path\to\training_data.db
python final_web_app.py
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% neq 0 (
    echo.
    echo App exited with code %EXITCODE%.
    pause
)
exit /b %EXITCODE%
