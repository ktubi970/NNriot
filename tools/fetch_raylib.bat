@echo off
REM Desert Strike: fetch Raylib (MinGW-w64 prebuilt) into include\ and lib\.
REM Run from any directory; resolves paths relative to this script.
REM Re-running is safe: overwrites the headers and static library.
REM
setlocal
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "RAYLIB_VERSION=5.5"
set "RAYLIB_ZIP_NAME=raylib-%RAYLIB_VERSION%_win64_mingw-w64"
set "RAYLIB_URL=https://github.com/raysan5/raylib/releases/download/%RAYLIB_VERSION%/%RAYLIB_ZIP_NAME%.zip"
set "TMP_DIR=%ROOT_DIR%\tools\.raylib_tmp"
set "PATH=%ROOT_DIR%\tools\w64devkit\bin;%PATH%"

echo Fetching Raylib %RAYLIB_VERSION% for MinGW-w64...

if exist "%TMP_DIR%" rmdir /s /q "%TMP_DIR%"
mkdir "%TMP_DIR%"

wget -q -O "%TMP_DIR%\raylib.zip" "%RAYLIB_URL%"
if errorlevel 1 (
    echo Download FAILED: %RAYLIB_URL%
    exit /b 1
)

unzip -q "%TMP_DIR%\raylib.zip" -d "%TMP_DIR%"
if errorlevel 1 (
    echo Extraction FAILED
    exit /b 1
)

if not exist "%ROOT_DIR%\include" mkdir "%ROOT_DIR%\include"
if not exist "%ROOT_DIR%\lib"     mkdir "%ROOT_DIR%\lib"

copy /Y "%TMP_DIR%\%RAYLIB_ZIP_NAME%\include\raylib.h"  "%ROOT_DIR%\include\" >nul
copy /Y "%TMP_DIR%\%RAYLIB_ZIP_NAME%\include\raymath.h" "%ROOT_DIR%\include\" >nul
copy /Y "%TMP_DIR%\%RAYLIB_ZIP_NAME%\include\rlgl.h"    "%ROOT_DIR%\include\" >nul
copy /Y "%TMP_DIR%\%RAYLIB_ZIP_NAME%\lib\libraylib.a"   "%ROOT_DIR%\lib\"     >nul
copy /Y "%TMP_DIR%\%RAYLIB_ZIP_NAME%\LICENSE"           "%ROOT_DIR%\lib\RAYLIB_LICENSE.txt" >nul

rmdir /s /q "%TMP_DIR%"

echo Raylib %RAYLIB_VERSION% installed:
echo   include\raylib.h, raymath.h, rlgl.h
echo   lib\libraylib.a
endlocal
