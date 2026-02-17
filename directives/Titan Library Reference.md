# Titan Library Reference

**Package Name:** `titan-oanda-algo`
**Import Name:** `titan`
**Version:** 0.1.0

The `titan` package is the core library powering the Titan OANDA Algo system. It contains reusable business logic, quantitative models, and infrastructure adapters, completely decoupled from execution scripts.

---

## 🏗️ Architecture Philosophy

The Titan library follows a **Strict Layered Architecture** to ensure stability and reproducibility.

### 1. The "No Scripts" Rule
*   `titan/` contains **only** functions, classes, and constants.
*   It **never** contains code that runs immediately on import (no `if __name__ == "__main__"`).
*   It **never** modifies `sys.path`.

### 2. Dependency Flow
Dependencies flow **inwards** towards the core models.
*   ❌ `titan.models` should NOT import from `titan.strategies` (Circles).
*   ✅ `titan.strategies` imports from `titan.models` and `titan.indicators`.
*   ✅ `scripts/` import from everything.

### 3. Configuration Injection
To keep the library testable:
*   Functions should accept configuration (dictionaries or objects) as arguments.
*   They should **avoid** loading files from `../../config` directly where possible.

---

## 📚 Modules Reference

### 1. `titan.adapters` (Nautilus-Oanda Adapter)

This module implements the custom **OANDA V20 Adapter** built for the Titan-Oanda-Algo project. It serves as the bridge between the event-driven **NautilusTrader** engine and **OANDA's REST/Streaming APIs**.

> **📘 Full Documentation:** See the [Titan-Oanda Adapter Guide](Titan-Oanda%20Adapter%20Guide.md) for detailed architecture, usage, and troubleshooting.

#### Key Components
*   **DataClient (`titan/adapters/oanda/data.py`):** Streams real-time price ticks.
*   **ExecutionClient (`titan/adapters/oanda/execution.py`):** Handles order submission and reconciliation.
*   **Instruments (`titan/adapters/oanda/instruments.py`):** Maps OANDA symbols to Nautilus instruments.

---

### 2. `titan.data`
Utilities for fetching, validating, and managing historical data.
*   **`titan.data.oanda`**: primitives for OANDA API data requests (candles, instruments).
    *   *Usage:* Used by `scripts/download_data.py` (which runs automatically in live strategies) to sync history.
    *   `fetch_candles(client, instrument, granularity, ...)`: Robust pagination for history.
*   **`titan.data.validation`**: Data integrity checks.
    *   `check_gaps(df)`: Detects missing candles.
    *   `check_outliers(df)`: flags suspicious price spikes.

### 3. `titan.indicators`
High-performance technical indicators optimized for both Numba (backtesting) and standard Python (live).
*   **`titan.indicators.gaussian_filter`**: Ehlers-based non-linear indicators (Gaussian Channel).
*   **`titan.indicators.common`**: Shared logic for standard indicators (SMA, EMA, RSI). [Planned/In-Progress]

### 4. `titan.models`
Quantitative models that model market physics and trading costs.
*   **`titan.models.spread`**: Time-varying spread and slippage estimation.
    *   `build_spread_series(df, pair)`: Estimates spread based on session (Tokyo/London/NY).
    *   `estimate_slippage(units, volume)`: Impact model based on square-root law.

### 5. `titan.strategies`
Production-grade strategy logic, separated from the execution harness.
*   **`titan.strategies.mtf`**: Multi-Timeframe Confluence logic.
*   **`titan.strategies.ml`**: Machine Learning signal generation and feature engineering.
    *   *Note:* Ensure `titan.strategies.ml.features` matches training code exactly to avoid drift.

### 6. `titan.utils`
Operational utilities for production handling.
*   **`titan.utils.ops`**: Emergency operations.
    *   `cancel_all_orders()`: Wipes pending orders.
    *   `close_all_positions()`: Flattens the account.
*   **`titan.utils.notification`**: Slack alerting integration.

---

## 📦 Installation

The package is designed to be installed in **editable mode** within your development environment.

```bash
# From the project root
pip install -e .
```

---

## 🚀 Usage Examples

### Fetching Data
```python
from titan.data.oanda import fetch_candles, candles_to_dataframe
import oandapyV20

client = oandapyV20.API(access_token="...")
candles = fetch_candles(client, "EUR_USD", "H1", count=500)
df = candles_to_dataframe(candles)
print(df.head())
```

### Checking Data Quality
```python
from titan.data.validation import check_gaps, check_outliers

# Validate a DataFrame
gap_count = check_gaps(df, "EUR_USD")
spike_count = check_outliers(df, "EUR_USD", z_threshold=5.0)

if gap_count == 0 and spike_count == 0:
    print("Data is clean!")
```

### Emergency Flatten
```python
from titan.utils.ops import close_all_positions, cancel_all_orders

# Emergency Switch
cancel_all_orders(client, account_id)
close_all_positions(client, account_id)
```

---

## 🛠️ Development Guidelines

1.  **Strict Separation**: Never import from `scripts/` or `research/` into `titan/`. The library must be self-contained.
2.  **No `sys.path`**: Do not use `sys.path.insert` in library code. Rely on proper package installation.
3.  **Type Hints**: All library functions must be fully type-hinted.
4.  **Config Injection**: Prefer passing configuration (dicts/objects) to functions rather than loading files from global paths inside the library.
