# VideoPipe — Vibe Coding App Spec (VC App Spec)

This folder contains **formal Markdown specifications** so that any developer (or AI assistant in a VC IDE like Cursor) can **reproduce or extend** the VideoPipe application from documentation alone.

## Purpose

- **Reproducibility**: Another vibe coder can rebuild or fork the program from these specs.
- **Onboarding**: New contributors (human or AI) get a single source of truth.
- **Spec-first**: The app is fully described before or alongside code.

## Document Index

| File | Description |
|------|-------------|
| [01-overview.md](01-overview.md) | What the app is, goals, tech stack, high-level flow |
| [02-requirements.md](02-requirements.md) | Functional and non-functional requirements |
| [03-architecture.md](03-architecture.md) | System architecture, pipeline DAG, context, data flow |
| [04-data-models.md](04-data-models.md) | Context, config, subtitle/segment structures, presets |
| [05-pipeline-spec.md](05-pipeline-spec.md) | Pipeline engine, node contract, execution order |
| [06-nodes-catalog.md](06-nodes-catalog.md) | All nodes: inputs, outputs, dependencies |
| [07-cli-and-config.md](07-cli-and-config.md) | CLI commands, YAML/JSON config format |
| [08-subtitles-and-effects.md](08-subtitles-and-effects.md) | STT, subtitle rendering, neon, transitions |
| [09-setup-and-run.md](09-setup-and-run.md) | Dependencies, install, run, examples |

## How to Use (Vibe Coder / AI)

1. **Start with** [01-overview.md](01-overview.md) and [03-architecture.md](03-architecture.md) to understand the product and structure.
2. **Implement or extend** using [05-pipeline-spec.md](05-pipeline-spec.md) and [06-nodes-catalog.md](06-nodes-catalog.md).
3. **Wire CLI and config** from [07-cli-and-config.md](07-cli-and-config.md).
4. **Match behavior** for subtitles and effects using [08-subtitles-and-effects.md](08-subtitles-and-effects.md).
5. **Run and verify** using [09-setup-and-run.md](09-setup-and-run.md).

## Conventions

- **Node names** are lowercase with underscores (e.g. `load_videos`, `render_subtitles`).
- **Dependencies** are expressed as a list of node names; execution order is resolved by topological sort.
- **Context** is the single shared state object passed through all nodes; nodes read/write it.
- **Config** can come from YAML/JSON file or CLI; file config is merged with defaults.

## Version

Spec matches **VideoPipe** as of the repo state when this spec was generated (see project `pyproject.toml` for package version).
