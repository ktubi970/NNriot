# Desert Strike: Return to the Gulf 🚁

A modern, high-performance clone of the classic helicopter combat simulator, rebuilt from the ground up using the **C** programming language and **Raylib**.

## 🛠️ Technology Stack
- **Language**: C99
- **Graphics Framework**: [Raylib 5.0+](https://www.raylib.com/)
- **Build Tool**: GNU Make (via `w64devkit`)
- **Target OS**: Windows (native build)

## 📁 Project Structure
- `src/`: Core game logic (`main.c`, `player.c`, etc.)
- `tools/`: Development tools including the `w64devkit` compiler.
- `screen references/`: Reference screenshots from the original SNES game.
- `Desert Strike - Return to the Gulf (Europe).sfc`: Reference ROM.

## 🚀 Setup & Build
1.  **Download Raylib**: Download the Win64 version of Raylib.
2.  **Add to project**:
    - Place `raylib.h` in the `include/` folder (create it if missing).
    - Place `libraylib.a` in the `lib/` folder (create it if missing).
3.  **Build**:
    Open a `w64devkit` terminal in the root directory and run:
    ```bash
    make
    ```
	or run `build.bat`
4.  **Run**:
    ```bash
    ./bin/desert_strike.exe
    ```

## 🎮 Development Progress
- [x] Project Bootstrap
- [ ] Basic 8-way Helicopter Movement
- [ ] Asset Pipeline (Sprite loading)
- [ ] Isometric View Implementation
- [ ] Enemy AI & Combat System

---
![License](https://img.shields.io/badge/License-MIT-blue.svg)
