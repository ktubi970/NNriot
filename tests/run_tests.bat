@echo off
setlocal
set PATH=%CD%\tools\w64devkit\bin;%PATH%

echo [Building Tests...]
gcc tests/test_collision.c src/collision.c -Iinclude -Isrc -o bin/test_collision.exe -lm

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Test Build Failed!
    exit /b %ERRORLEVEL%
)

echo [Running Tests...]
bin\test_collision.exe

endlocal
