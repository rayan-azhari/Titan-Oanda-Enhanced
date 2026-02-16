# Directive: Titan Package Architecture

> **Status: ✅ IMPLEMENTED**

## Goal

Provide a **robust, production-grade library** (`titan`) that houses all core trading logic, adapters, and models. This package separates deterministic execution code from experimental research scripts.

This architecture ensures:
1.  **Reusability:** Core logic (e.g., indicators, spread models) is shared between research and live trading.
2.  **Stability:** The `titan` package is tested and versioned; scripts depend on it.
3.  **Clarity:** Clear separation of concerns (Data vs. Logic vs. Execution).

## 1. Directory Structure & Module Breakdown

| Package | Purpose | Key Components |
|---|---|---|
| **`titan.adapters`** | Venue connectivity | `oanda/` (NautilusTrader adapter) |
| **`titan.config`** | Configuration loading | `tomllib` wrappers for `config/*.toml` |
| **`titan.data`** | Data fetching & validation | `oanda.py` (API logic), `validation.py` |
| **`titan.indicators`** | Technical Indicators | High-performance Numba/VectorBT factories |
| **`titan.models`** | Mathematical Models | `spread.py` (Cost estimation), `slippage.py` |
| **`titan.strategies`** | Production Strategies | NautilusTrader strategy classes |
| **`titan.utils`** | Shared Utilities | `logging`, `ops`, `notification` |

---

## 2. Detailed Module Documentation

### 2.1 `titan.adapters` (Venue Connectivity)
Custom adapters for linking the NautilusTrader engine to external exchanges.

-   **`titan.adapters.oanda`**: The OANDA V20 adapter.
    -   `data.py`: **`OandaDataClient`**
        -   Streams real-time price ticks (`QuoteTick`).
        -   Handles subscriptions and heartbeats.
    -   `execution.py`: **`OandaExecutionClient`**
        -   Submits orders (Market, Limit, Stop).
        -   Reconciles positions on startup.
    -   `parsing.py`: Utilities to convert OANDA JSON -> Nautilus Objects.

### 2.2 `titan.config` (Configuration)
Type-safe access to TOML configuration files.

-   **Usage:**
    ```python
    from titan.config import load_instruments_config
    config = load_instruments_config()
    pairs = config["instruments"]["pairs"]
    ```
-   **Files:**
    -   `instruments.toml`: Currency pairs and granularities.
    -   `risk.toml`: Position limits and drawdown caps.
    -   `features.toml`: Optimized indicator parameters (output of research).

### 2.3 `titan.data` (Data Engineering)
Logic for retrieving, cleaning, and storing market data.

-   **`titan.data.oanda`**:
    -   `fetch_candles(client, instrument, granularity)`: Rate-limited API wrapper.
    -   `candles_to_dataframe(candles)`: Converts JSON to Pandas DataFrame with correct types (`float`, `int`, `datetime`).
-   **`titan.data.validation`**:
    -   `validate_dataframe(df)`: Checks for:
        -   Missing timestamps (gaps).
        -   Zero volume.
        -   Price anomalies (High < Low).

### 2.4 `titan.indicators` (Signal Logic)
High-performance indicator implementations.

-   **Principles:**
    -   **Numba-compiled** for speed (`@njit`).
    -   **VectorBT-compatible** for backtesting.
-   **Key Indicators:**
    -   `gaussian_filter.py`: Ehlers Gaussian Channel (Filters noise, responsive logic).
    -   `mtf.py`: Multi-timeframe moving averages and RSI.

### 2.5 `titan.models` (Quant Models)
Mathematical models for simulation and risk.

-   **`titan.models.spread`**:
    -   `build_spread_series(df, pair)`: Estimates spread based on session time (Tokyo < London < NY).
    -   **Why?** Backtests using flat spreads are unrealistic. This model injects realistic transaction costs.

### 2.6 `titan.strategies` (Production Strategies)
Strategies ready for the **NautilusTrader** live engine.

-   **Structure:**
    -   `strategy.py`: The `Nautilus` strategy class (event-driven).
    -   `config.py`: Configuration dataclass (`StrategyConfig`).
    -   `signal.py`: Core signal logic (decoupled from the engine).
-   **Implemented:**
    -   `mtf/`: Multi-Timeframe Confluence (Trend + Momentum).
    -   `ml/`: Machine Learning (XGBoost signal execution).

### 2.7 `titan.utils` (Utilities)
Shared helper functions.

-   `ops.py`: Operational tools (e.g., `kill_switch` logic).
-   `notification.py`: Slack/Discord alerting via webhooks.
-   `logging.py`: Structured logging configuration (JSON/Console).

---

## 3. Development Guidelines

### 3.1 Coding Standards
All code in `titan/` must adhere to:
1.  **Strict Typing:** Use type hints for everything.
    -   `def calculate_sma(data: pd.Series, window: int) -> pd.Series:`
2.  **Docstrings:** Google Style docstrings for all public modules, classes, and functions.
3.  **Imports:** Absolute imports only.
    -   `from titan.models.spread import ...` (NOT `from ..models import ...`)

### 3.2 Testing
Unit tests mirror the package structure in `tests/`.
-   **Run:** `uv run pytest`
-   **Coverage:** Core logic (indicators, models) requires >90% coverage.

### 3.3 Adding a New Feature
1.  **Prototype** in `research/` (e.g., Jupyter notebook or script).
2.  **Refactor** core logic into `titan/` (e.g., `titan/indicators/new_indicator.py`).
3.  **Test** with `pytest`.
4.  **Integrate** into `scripts/` or `titan.strategies`.

---

## 4. Integration Patterns

### Research -> Production Pipeline
1.  **Idea:** "Let's try a Kalman Filter."
2.  **Research:** Implement `research/kalman_test.py`.
3.  **Promotion:** Move the filter logic to `titan/models/kalman.py`.
4.  **Execution:** Import `titan.models.kalman` in `titan/strategies/kalman_strat.py`.

### Script Usage
Scripts in `scripts/` should be thin wrappers around `titan` logic.
-   **Bad Script:** Contains complex math or class definitions.
-   **Good Script:** Imports `titan` classes, configures them, and runs.

Example `scripts/download_data.py`:
```python
from titan.data.oanda import fetch_candles
from titan.config import load_instruments_config

def main():
    config = load_instruments_config()
    # ... logic ...
```
