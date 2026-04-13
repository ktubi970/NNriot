# Specification: Obsidian Knowledge Graph Roadmap

## 1. Overview
The goal is to create a structured Markdown knowledge base (Vault) within the project to track the development roadmap of "Desert Strike: Return to the Gulf". This vault will be optimized for Obsidian's Graph View to visualize progress and technical dependencies.

## 2. Directory Structure
```text
brain/
├── Roadmap.md                  # Main Index / Map of Content
├── Phases/                     # Sequential Milestone Notes
│   ├── Phase 1 - Foundations.md
│   ├── Phase 2 - Mechanics.md
│   └── ...
├── Features/                   # Technical Deep Dives
│   ├── Collision System.md
│   ├── Isometric Rendering.md
│   └── ...
└── Assets/                     # Asset Extraction Tracking
    └── Sprite Registry.md
```

## 3. Formatting Standards
- **Sequential Linking**: Each Phase note must link to its predecessor and successor to create a chronological path in the graph.
- **Status Tags**: Use visual indicators for progress:
  - `status::done` (🟢)
  - `status::in-progress` (🟡)
  - `status::todo` (⚪)
- **Feature Links**: Detailed tasks within Phases should be `[[wikilinks]]` to notes in the `Features/` or `Assets/` folders.

## 4. Implementation Plan
1. Create the `brain/` directory and subdirectories (`Phases/`, `Features/`, `Assets/`).
2. Generate `brain/Roadmap.md` with a high-level overview.
3. Generate the first three Phase notes based on the current state in `Claude.md`.
4. Ensure cross-linking between the Roadmap and Phases is correctly established.
