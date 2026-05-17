# Desert Strike: Return to the Gulf - Build configuration
#
# Toolchain: w64devkit (GCC + GNU Make for Windows) - see tools/w64devkit
# Use build.bat from cmd/PowerShell to set PATH automatically, or run
# `make` directly if tools/w64devkit/bin is already on PATH.

CC      ?= gcc
CFLAGS  := -std=c99 -Wall -Wextra -Wpedantic -Iinclude -MMD -MP
LDFLAGS := -Llib
LDLIBS  := -lraylib -lopengl32 -lgdi32 -lwinmm

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
TARGET  := $(BIN_DIR)/desert_strike.exe

SOURCES := $(wildcard $(SRC_DIR)/*.c)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
DEPS    := $(OBJECTS:.o=.d)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CC) $(OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

-include $(DEPS)
