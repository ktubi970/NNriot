# Obsidian Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a structured Markdown knowledge base (Vault) within the project to track the development roadmap, optimized for Obsidian's Graph View.

**Architecture:** A central Roadmap index linking sequentially through project phases, categorized in `brain/`, `brain/Phases/`, `brain/Features/`, and `brain/Assets/` directories. Each phase will link to its predecessor/successor and individual feature notes.

**Tech Stack:** Markdown, Obsidian (Graph View)

---

### Task 1: Initialize Vault Directory Structure

**Files:**
- Create: `brain/`
- Create: `brain/Phases/`
- Create: `brain/Features/`
- Create: `brain/Assets/`

- [ ] **Step 1: Write the validation checks (TDD)**

Run: `powershell -Command "Test-Path brain/Phases/"`
Expected: `False`

- [ ] **Step 2: Create directory structure**

Run: `powershell -Command "New-Item -ItemType Directory -Force -Path brain/Phases/; New-Item -ItemType Directory -Force -Path brain/Features/; New-Item -ItemType Directory -Force -Path brain/Assets/"`

- [ ] **Step 3: Run validation to verify setup**

Run: `powershell -Command "Test-Path brain/Features/"`
Expected: `True`

- [ ] **Step 4: Commit**

Run:
```bash
git add brain/
git commit -m "chore: initialize obsidian brain directory structure"
```

---

### Task 2: Create Phase 1 - Engine Foundations

**Files:**
- Create: `brain/Phases/Phase 1 - Engine Foundations.md`

- [ ] **Step 1: Create Phase 1 file with content**

Run: 
```powershell
Set-Content -Path "brain/Phases/Phase 1 - Engine Foundations.md" -Value @"
# Phase 1 - Engine Foundations

**Previous:** None
**Next:** [[Phase 2 - Tactical Mechanics]]

status::in-progress

## Objectives
- [x] [[Bootstrapping]] (Raylib + Makefile)
- [ ] [[Isometric Transformation]] (Screen-to-World)
- [ ] [[Base Collision System]] (Circle-Object)
"@
```

- [ ] **Step 2: Verify file creation**

Run: `powershell -Command "Get-Content 'brain/Phases/Phase 1 - Engine Foundations.md' | Select-Object -First 1"`
Expected: `# Phase 1 - Engine Foundations`

- [ ] **Step 3: Commit**

Run:
```bash
git add "brain/Phases/Phase 1 - Engine Foundations.md"
git commit -m "docs: add Phase 1 to roadmap"
```

---

### Task 3: Create Phase 2 - Tactical Mechanics

**Files:**
- Create: `brain/Phases/Phase 2 - Tactical Mechanics.md`

- [ ] **Step 1: Create Phase 2 file with content**

Run: 
```powershell
Set-Content -Path "brain/Phases/Phase 2 - Tactical Mechanics.md" -Value @"
# Phase 2 - Tactical Mechanics

**Previous:** [[Phase 1 - Engine Foundations]]
**Next:** [[Phase 3 - Level and Mission Design]]

status::todo

## Objectives
- [ ] [[Helicopter Movement]] (8 directions)
- [ ] [[Weapons System]] (Hellfire/Hydra/M230)
- [ ] [[Resource Management]] (Munitions and fuel)
"@
```

- [ ] **Step 2: Verify file creation**

Run: `powershell -Command "Get-Content 'brain/Phases/Phase 2 - Tactical Mechanics.md' | Select-Object -First 1"`
Expected: `# Phase 2 - Tactical Mechanics`

- [ ] **Step 3: Commit**

Run:
```bash
git add "brain/Phases/Phase 2 - Tactical Mechanics.md"
git commit -m "docs: add Phase 2 to roadmap"
```

---

### Task 4: Create Phase 3 - Level and Mission Design

**Files:**
- Create: `brain/Phases/Phase 3 - Level and Mission Design.md`

- [ ] **Step 1: Create Phase 3 file with content**

Run: 
```powershell
Set-Content -Path "brain/Phases/Phase 3 - Level and Mission Design.md" -Value @"
# Phase 3 - Level and Mission Design

**Previous:** [[Phase 2 - Tactical Mechanics]]
**Next:** [[Phase 4 - Internal Polish]]

status::todo

## Objectives
- [ ] [[Mission Objectives]]
- [ ] [[HUD]]
- [ ] [[Level Layout]]
"@
```

- [ ] **Step 2: Verify file creation**

Run: `powershell -Command "Get-Content 'brain/Phases/Phase 3 - Level and Mission Design.md' | Select-Object -First 1"`
Expected: `# Phase 3 - Level and Mission Design`

- [ ] **Step 3: Commit**

Run:
```bash
git add "brain/Phases/Phase 3 - Level and Mission Design.md"
git commit -m "docs: add Phase 3 to roadmap"
```

---

### Task 5: Create Central Roadmap Index

**Files:**
- Create: `brain/Roadmap.md`

- [ ] **Step 1: Create Roadmap file with content**

Run: 
```powershell
Set-Content -Path "brain/Roadmap.md" -Value @"
# Desert Strike Roadmap

This is the main map of content for the project's development phases.

## Graph Flow
1. [[Phase 1 - Engine Foundations]]
2. [[Phase 2 - Tactical Mechanics]]
3. [[Phase 3 - Level and Mission Design]]
4. [[Phase 4 - Internal Polish]]

## Directories
- **Features:** Specific functional nodes like [[Isometric Transformation]].
- **Assets:** Tracking nodes like [[Sprite Registry]].
"@
```

- [ ] **Step 2: Verify file creation**

Run: `powershell -Command "Get-Content 'brain/Roadmap.md' | Select-Object -First 1"`
Expected: `# Desert Strike Roadmap`

- [ ] **Step 3: Commit**

Run:
```bash
git add "brain/Roadmap.md"
git commit -m "docs: create main roadmap index"
```
