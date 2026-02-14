# Titan-Oanda-Algo: Rules of Engagement

All agents must adhere to these constraints to ensure system integrity, financial accuracy, and scientific rigour.

---

## Technical Constraints

| Rule | Detail |
|---|---|
| **Dependency Management** | All Python code must use `uv` for dependency management. No bare `pip` installs. |
| **Financial Precision** | All financial data types must use `decimal.Decimal` or Nautilus native types. Standard floats are **strictly prohibited** for price or volume logic. |
| **Coding Standard** | All code must follow the [Google Style Guide](https://google.github.io/styleguide/pyguide.html). |
| **Documentation** | All public methods must include docstrings. |

---

## ML & Data Science Standards

| Rule | Detail |
|---|---|
| **Determinism** | Set explicit random seeds (`random_state=42`) for all ML training to ensure reproducibility. |
| **Data Leakage** | Features must be lagged. Targets must be future-derived. The Researcher must explicitly check for look-ahead bias before training. |
| **Feature Selection** | ML models must consume features tuned by `run_feature_selection.py` (read from `config/features.toml`) whenever possible, rather than using arbitrary default parameters. |
| **Storage** | Trained models are "Artifacts" and must be stored in `models/`, **not** `.tmp/`. |

---

## CI/CD & Code Quality

| Rule | Detail |
|---|---|
| **Pre-Push Checks** | Before every `git push`, run: `uv run ruff check .`, `uv run ruff format .`, and `uv run pytest tests/ -v`. All three must pass with zero errors. |
| **Ruff Linting** | All code must pass `ruff check .` (rules: E, F, I, W, D). Fix lint errors with `uv run ruff check . --fix`. |
| **Ruff Formatting** | All code must pass `ruff format --check .`. Auto-format with `uv run ruff format .` before committing. |
| **NautilusTrader API** | Use factory methods (`Price.from_str()`, `Quantity.from_str()`, `CurrencyPair.from_dict()`) instead of direct constructors. Direct `__init__` signatures change between versions. |
| **Import Paths** | If NautilusTrader updates break imports, check the latest docs. Common moves: `LiveDataClient` → `nautilus_trader.live.data_client`, `CurrencyPair` → `nautilus_trader.model.instruments.currency_pair`. |
| **Test Environment** | Tests requiring live OANDA credentials must be wrapped with `pytest.mark.skipif` to gracefully skip in CI. |

---

## Agent Personas

| Agent | Runtime | Responsibilities |
|---|---|---|
| **Architect** | Gemini 3 Pro | File structure, `config/` management, and high-level design |
| **Engineer** | Gemini 3 Pro / Claude Sonnet | Writing Rust/Python code and API adapters |
| **Researcher** | Gemini 3 Pro | VectorBT scripts, data modelling, ML feature engineering, and Jupyter notebooks |