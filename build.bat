@echo off
setlocal
set PATH=%CD%\tools\w64devkit\bin;%PATH%

echo ----------------------------------------
echo [ Desert Strike ] Launching AI Build...
echo ----------------------------------------

:: Kill existing game if running
taskkill /IM desert_strike.exe /F >nul 2>&1

:: Build
make

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build Failed!
    pause
    exit /b %ERRORLEVEL%
)

:: Run
echo [SUCCESS] Build Complete. Launching...
bin\desert_strike.exe
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Game closed with exit code %ERRORLEVEL%
)

endlocal
