@echo off
REM Desert Strike: unit-test runner.
REM Compiles every tests\test_*.c file as its own standalone executable
REM into bin\tests\, linking all src\*.c modules except src\main.c
REM (which owns the executable entry point). Stops at the first failure.
REM
setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "PATH=%ROOT_DIR%\tools\w64devkit\bin;%PATH%"
set "OUT_DIR=%ROOT_DIR%\bin\tests"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set "CFLAGS=-std=c99 -Wall -Wextra -Wpedantic -I%ROOT_DIR%\include"

REM Collect src\*.c minus main.c so tests can call into the game modules
REM without colliding with the production main() symbol.
set "SRC_FILES="
for %%S in ("%ROOT_DIR%\src\*.c") do (
    if /I not "%%~nxS"=="main.c" set "SRC_FILES=!SRC_FILES! "%%S""
)

set "FAIL_COUNT=0"
for %%F in ("%SCRIPT_DIR%test_*.c") do (
    set "TEST_NAME=%%~nF"
    set "TEST_EXE=%OUT_DIR%\!TEST_NAME!.exe"
    echo === Building !TEST_NAME! ===
    gcc %CFLAGS% "%%F" !SRC_FILES! -o "!TEST_EXE!"
    if errorlevel 1 (
        echo BUILD FAILED: !TEST_NAME!
        set /a FAIL_COUNT+=1
        goto :report
    )
    echo === Running !TEST_NAME! ===
    "!TEST_EXE!"
    if errorlevel 1 (
        echo TEST FAILED: !TEST_NAME!
        set /a FAIL_COUNT+=1
        goto :report
    )
)

:report
echo.
if "%FAIL_COUNT%"=="0" (
    echo All tests passed.
    endlocal & exit /b 0
) else (
    echo %FAIL_COUNT% test step^(s^) failed.
    endlocal & exit /b 1
)
