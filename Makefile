# Makefile for Desert Strike: Return to the Gulf
# Using w64devkit for Windows build
#
# Project layout (assets stay at the root, beside the sfc and screen references):
#   assets/       <-- sprites, loaded at runtime via relative path
#   src/          <-- C source files
#   bin/          <-- compiled executable only
#   tools/        <-- w64devkit compiler

CC      = tools/w64devkit/bin/gcc.exe
CFLAGS  = -Wall -std=c99 -Iinclude
LDFLAGS = -Llib -lraylib -lopengl32 -lgdi32 -lwinmm

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SOURCES = src/main.c src/player.c src/resources.c src/asset_metadata.c src/collision.c
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
TARGET  = $(BIN_DIR)/desert_strike.exe

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Run from the PROJECT ROOT so that assets/ resolves relative to it
run: all
	./$(TARGET)

clean:
	@rm -rf $(OBJ_DIR) $(BIN_DIR)
