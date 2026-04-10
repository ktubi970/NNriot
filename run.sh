#!/bin/bash

# Desert Strike: Return to the Gulf - Build & Launch Script
# Robust version for w64devkit

# 1. Setup local environment
TOOLS_PATH="$PWD/tools/w64devkit/bin"
export PATH="$TOOLS_PATH:$PATH"

# Try to kill the game if it is running (to unlock the file)
# This handles the Windows "Permission Denied" issue.
echo "----------------------------------------"
echo "🛠️  PREPARING ENVIRONMENT..."
taskkill //IM desert_strike.exe //F > /dev/null 2>&1
echo "✅ Execution locks cleared."
echo "----------------------------------------"

# 2. Compile
echo "⚙️  COMPILING SOURCE..."
    # 4. Launch from root
    ./bin/desert_strike.exe
else
    echo "----------------------------------------"
    echo "❌ ERROR: Compilation Failed."
    echo "----------------------------------------"
    exit 1
fi
