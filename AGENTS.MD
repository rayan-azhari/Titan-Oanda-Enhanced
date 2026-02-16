# Titan-Oanda-Algo: Agent Instructions

This file is the primary system prompt for the **Titan-Oanda-Algo** workspace. It governs the behaviour of all agents (Architect, Engineer, Researcher).

You operate within a **Modular Architecture** designed to separate *Core Logic* from *Research* and *Execution*.

---

## The Architecture

### Layer 1: Core Logic (The Library)

- **Located in:** `titan/`
- **Purpose:** Reusable, tested, production-grade code.
- **Components:** Adapters, data fetching, indicators, strategies, and models.
- **Rule:** No scripts here. Only classes and functions.

### Layer 2: Research (The Lab)

- **Located in:** `research/`
- **Purpose:** Experimental code, notebooks, and optimization loops.
- **Output:** New features or configuration parameters (to be moved to `config/` or `titan/`).

### Layer 3: Execution (The Interface)

- **Located in:** `scripts/`
- **Purpose:** Entry points for operators (CLI tools).
- **Examples:** `download_data.py`, `run_live_mtf.py`, `check_env.py`.

---

## Operating Principles

### 1. The "Titan" Standard

- **Strict Typing:** All `titan/` code must be fully type-hinted.
- **Absolute Imports:** Use `from titan.models.spread import ...`, never relative imports.
- **Configuration:** Use `titan.config` to load TOML files.

### 2. Scientific Rigour (The ML Mandate)

When training models in `research/ml/`:

- **Reproducibility:** Always set `random_state=42`.
- **Sanity Checks:** If a strategy looks too good to be true, it is likely data leakage. Check your lagging logic.

### 3. File Organisation

| Directory | Purpose |
|---|---|
| `directives/` | The instruction sets (SOPs) |
| `titan/` | **Core Package:** The brain (strategies, models, data) |
| `research/` | **Research:** VectorBT loops, ML experiments |
| `scripts/` | **Scripts:** CLI entry points (download, trade, test) |
| `config/` | **Config:** TOML parameters |
| `data/` | **Data:** Historical Parquet files |
| `models/` | **Artifacts:** Trained ML models (`.joblib`) |
| `.tmp/` | **Temp:** Logs and transient reports |

### 4. Self-Correction

If a script fails:
1. Read the stack trace.
2. Fix the code in `titan/` or `scripts/`.
3. Run it again.

> [!WARNING]
> **Constraint:** All scripts must use `uv` for dependencies. Do not use bare `pip`.
> **Constraint:** Use `decimal.Decimal` for all pricing logic.

---

## Summary

You are building a **Quantitative Trading System**, not a script collection.

**Prioritise:** Modularity, Type Safety, and Reproducibility.

Read the `directives/`. Write reusable code in `titan/`. Experiment in `research/`. Execute via `scripts/`.