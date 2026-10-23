@echo off
REM NNriot launcher. .env settings are read by the Python CLI.
setlocal
cd /d "%~dp0"

set "PYTHON="
if exist "%~dp0.venv312\Scripts\python.exe" set "PYTHON=%~dp0.venv312\Scripts\python.exe"
if not defined PYTHON if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON if exist "%USERPROFILE%\anaconda3\python.exe" set "PYTHON=%USERPROFILE%\anaconda3\python.exe"
if not defined PYTHON if exist "%USERPROFILE%\miniconda3\python.exe" set "PYTHON=%USERPROFILE%\miniconda3\python.exe"
if not defined PYTHON set "PYTHON=python"
set "PYTHONIOENCODING=utf-8"

if "%~1"=="" goto :menu
if /i "%~1"=="web" goto :run_cli
if /i "%~1"=="train" goto :run_cli
if /i "%~1"=="train-continuous" goto :run_cli
if /i "%~1"=="backfill" goto :run_cli
if /i "%~1"=="build" goto :run_build
if /i "%~1"=="dev" goto :run_dev
if /i "%~1"=="test" goto :run_test
echo Unknown action: %~1
goto :usage

:menu
echo.
echo ============================================================
echo   NNriot Launcher
echo   Python: %PYTHON%
echo ============================================================
echo   [1] Web App
echo   [2] Train pre-match candidate checkpoint
echo   [3] Continuous trainer
echo   [4] Backfill labels
echo   [5] Build React frontend
echo   [6] Frontend dev server
echo   [7] Run Playwright tests
echo   [Q] Quit
echo ============================================================
set /p CHOICE="Choose an action [1]: "
if "%CHOICE%"=="" set "CHOICE=1"
if /i "%CHOICE%"=="1" goto :run_web
if /i "%CHOICE%"=="2" goto :run_train
if /i "%CHOICE%"=="3" goto :run_train_continuous
if /i "%CHOICE%"=="4" goto :run_backfill
if /i "%CHOICE%"=="5" goto :run_build
if /i "%CHOICE%"=="6" goto :run_dev
if /i "%CHOICE%"=="7" goto :run_test
if /i "%CHOICE%"=="q" goto :end
echo Invalid choice.
goto :menu

:run_cli
REM %* preserves quoted arguments, including --db paths with spaces.
"%PYTHON%" cli.py %*
goto :after

:run_web
"%PYTHON%" cli.py web
goto :after

:run_train
"%PYTHON%" cli.py train
goto :after

:run_train_continuous
"%PYTHON%" cli.py train-continuous
goto :after

:run_backfill
"%PYTHON%" cli.py backfill
goto :after

:run_build
call npm.cmd run build
goto :after

:run_dev
call npm.cmd run dev
goto :after

:run_test
call npm.cmd run test:e2e
goto :after

:usage
echo.
echo Usage: start.bat [web^|train^|train-continuous^|backfill^|build^|dev^|test] [options]
echo   No arg = interactive menu
exit /b 1

:after
set "EXITCODE=%ERRORLEVEL%"
if %EXITCODE% neq 0 (
    echo.
    echo Process exited with code %EXITCODE%.
    pause
)
exit /b %EXITCODE%

:end
endlocal
