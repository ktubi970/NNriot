@echo off
REM Desert Strike: Return to the Gulf - Windows build wrapper
REM Prepends w64devkit's bin to PATH so make/gcc are found, then runs make.
REM Usage:
REM   build.bat              -> build (default target: all)
REM   build.bat run          -> build and launch the game
REM   build.bat clean        -> remove obj/ and bin/
REM
setlocal
set "SCRIPT_DIR=%~dp0"
set "PATH=%SCRIPT_DIR%tools\w64devkit\bin;%PATH%"

if "%~1"=="" (
    mingw32-make all
) else (
    mingw32-make %*
)

set "BUILD_EXIT=%ERRORLEVEL%"
endlocal & exit /b %BUILD_EXIT%
