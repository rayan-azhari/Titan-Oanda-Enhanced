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

> **Version:** 1.1.0
> **Status:** Production-Ready (Practice Mode Verified)

#### 1.1 Architecture Overview

NautilusTrader requires specific interfaces to be implemented for a venue adapter. We implement two core clients:

##### DataClient (`titan/adapters/oanda/data.py`)
Responsible for streaming real-time price data.
-   **Class:** `OandaDataClient`
-   **Inherits:** `LiveDataClient`, `MarketDataClient` (Critical!)
-   **Source:** OANDA Pricing Stream (`/v3/accounts/{id}/pricing/stream`)
-   **Output:** `QuoteTick` objects (Bid/Ask) published to the Nautilus Message Bus.

##### ExecutionClient (`titan/adapters/oanda/execution.py`)
Responsible for order management and account state.
-   **Class:** `OandaExecutionClient`
-   **Inherits:** `LiveExecutionClient`
-   **Source:** OANDA REST API (`/v3/accounts/{id}/orders`) + Transactions Stream
-   **Output:** `OrderFilled`, `OrderCanceled`, `OrderFailed` events.

#### 1.2 Implementation Details & Fixes

This section details specific implementation choices and critical fixes applied during development.

##### OandaDataClient

###### ⚠️ Critical Inheritance Requirement
The client **MUST** inherit from both `LiveDataClient` and `MarketDataClient`.
```python
# CORRECT
class OandaDataClient(LiveDataClient, MarketDataClient):

# WRONG (Causes TypeError in DataEngine)
class OandaDataClient(LiveDataClient):
```
*Why?* The Nautilus `DataEngine` performs an `isinstance(client, MarketDataClient)` check before allowing subscription commands like `SubscribeBars`.

###### Subscription Methods (Command-Based API)
The Nautilus `DataEngine` dispatches **command objects** (not raw IDs) to subscribe methods. All methods must accept `(self, command)` and extract `.instrument_id` from the command:
```python
def subscribe_quote_ticks(self, command):   # command.instrument_id → InstrumentId
def subscribe_bars(self, command):           # command.instrument_id or command.bar_type.instrument_id
def subscribe_instrument(self, command):     # command.instrument_id
```
> [!CAUTION]
> These methods **must NOT be `async def`**. Nautilus calls them synchronously.
> If declared as `async`, the returned coroutine is silently discarded and subscriptions never execute.

Internally, all methods delegate to `_add_instrument()` which schedules a stream restart via `self._loop.create_task(self._restart_stream())`.

###### Data Flow Pipeline
```
OANDA PricingStream → _consume_stream() [executor thread]
  → _parse_quote() → QuoteTick
  → call_soon_threadsafe(_handle_data_py)
  → Nautilus DataEngine → Bar aggregation (INTERNAL)
  → Strategy.on_bar()
```

> [!IMPORTANT]
> Use `self._handle_data_py(tick)` to publish data — **not** `self._msgbus.publish_data()`.
> The `Price()` constructor requires an **integer** precision (e.g. `5` for `1.18523`), not `None`.

###### Connection Resilience
-   Implements **exponential backoff** (up to 60s) if the OANDA stream disconnects.
-   Automatically re-subscribes to all instruments upon reconnection.

##### OandaExecutionClient

###### ⚠️ Account ID Property
The base `LiveExecutionClient` defines `account_id` as a property. We cannot simply set `self.account_id = ...` in `__init__`.
**Fix:** We store the raw ID in `self._account_id` and override the property:
```python
@property
def account_id(self) -> AccountId:
    return AccountId(f"OANDA-{self._account_id}")
```

###### Reconciliation (Startup State)
When the engine starts (or reconnects), it must reconcile its internal state with OANDA to avoid duplicate trades or ghost positions. Two methods handle this:

**Order Reconciliation — `generate_order_status_reports`**
-   **Logic:**
    1.  Fetches all `PENDING` orders from OANDA REST API.
    2.  Filters for orders with `clientExtensions.id` (checking if they belong to Nautilus).
    3.  Maps OANDA string fields to Nautilus Enums (see below).
    4.  Returns a list of `OrderStatusReport` objects.

**Position Reconciliation — `generate_position_status_reports`**
-   **Endpoint:** `oandapyV20.endpoints.positions.OpenPositions`
-   **Logic:**
    1.  Fetches all open positions from OANDA.
    2.  Parses `long.units` (positive) and `short.units` (negative) for each instrument.
    3.  Computes net position: `net_units = long_units + short_units`.
    4.  If net is non-zero, creates a `PositionStatusReport` with:
        - `PositionSide.LONG` or `PositionSide.SHORT` based on net direction.
        - `Quantity(abs(net_units))` — Nautilus requires unsigned quantities.
        - `avg_px_open` from the dominant side's `averagePrice`.
    5.  Flat positions (net = 0) are skipped.
-   **Error Handling:** API failures are caught and logged; an empty list is returned.
-   **Tests:** 6 unit tests in `tests/test_oanda_reconciliation.py`.

###### Enum Mapping
OANDA returns strings; Nautilus expects Enums.
-   **Order Type:** `"MARKET"` → `OrderType.MARKET`, `"LIMIT"` → `OrderType.LIMIT`.
-   **Time In Force:** `"GTC"` → `TimeInForce.GTC`, else `TimeInForce.FOK`.

###### Transaction Event Handling
The `_handle_message` loop processes the Transactions Stream:
-   `ORDER_FILL` → Triggers `_handle_fill` → Publishes `OrderFilled`.
-   `ORDER_CANCEL` → Triggers `_handle_cancel` → Publishes `OrderCanceled`.
-   **Crucial:** We use `clientExtensions.id` to map the OANDA transaction back to the original Nautilus `ClientOrderId`.

#### 1.3 Usage Guide

##### Instantiation (in `run_live_mtf.py`)

```python
import asyncio
from execution.nautilus_oanda.config import OandaConfig
from execution.nautilus_oanda.data import OandaDataClient
from execution.nautilus_oanda.execution import OandaExecutionClient

# 1. Config
oanda_config = OandaConfig(
    account_id="001-001-...",
    access_token="...",
    environment="practice"
)

# 2. Clients
data_client = OandaDataClient(loop, client_id, venue, oanda_config, ...)
exec_client = OandaExecutionClient(loop, client_id, venue, oanda_config, ...)

# 3. Add to Trading Node
node.add_data_client(data_client)
node.add_exec_client(exec_client)
```

##### Required Environment Variables
The `setup_env.py` script usually handles this, but ensure these exist:
-   `OANDA_ACCOUNT_ID`: The 16-digit ID (e.g., `101-004-1234567-001`).
-   `OANDA_ACCESS_TOKEN`: The API key.
-   `OANDA_ENVIRONMENT`: `practice` or `live`.

#### 1.4 Troubleshooting

##### Common Errors & Solutions

###### 1. "TypeError: Cannot convert OandaDataClient to MarketDataClient"
-   **Context:** Startup, just after "Strategy Added".
-   **Cause:** `OandaDataClient` is missing `MarketDataClient` in its inheritance list.
-   **Fix:** Edit `titan/adapters/oanda/data.py` class definition.

###### 2. "AttributeError: 'NoneType' object has no attribute 'value'"
-   **Context:** Startup, during `ExecutionMassStatus` generation.
-   **Cause:** `self.account_id` is None.
-   **Fix:** Ensure `account_id` is implemented as a `@property` in `execution.py`, not a raw attribute.

###### 3. "NotImplementedError: method `generate_order_status_reports`..."
-   **Context:** Startup, immediately crashes.
-   **Cause:** Nautilus is trying to reconcile state but the method is missing.
-   **Fix:** Implement the method (copy from `execution.py`) to fetch open orders.

###### 4. "SubscribeBars" does nothing / Silent Failure
-   **Context:** Strategy says "Warming up" but never gets data.
-   **Cause:** `subscribe_bars` alias is missing in `OandaDataClient`. Nautilus calls this method by default for `SubscribeBars` commands.
-   **Fix:** Add `async def subscribe_bars(self, bar_type): await self.subscribe(...)`.

###### 5. "TypeError: Argument 'position_id' has incorrect type"
-   **Context:** Strategy `on_bar` logic.
-   **Cause:** Calling `cache.position(instrument_id)`. The cache expects a `PositionId` UUID, not an `InstrumentId`.
-   **Fix:** Use `cache.positions(instrument_id=...)` instead.

#### 1.5 Development Guidelines

If you need to extend this adapter:

1.  **Don't Break Inheritance:** Always verify generic types with `issubclass(Client, MarketDataClient)` via a script if unsure.
2.  **Thread Safety:** OANDA's `oandapyV20` is synchronous/blocking. Always run API calls in the executor:
    ```python
    await self._loop.run_in_executor(None, lambda: self._api.request(r))
    ```
3.  **Callback Safety:** When streaming data from a background thread (executor), use `call_soon_threadsafe` to push data back to the main method:
    ```python
    self._loop.call_soon_threadsafe(self.handle_data, tick)
    ```
4.  **Logging:** Use `self._log.info()` or `error()`, not `print()`.

#### 1.6 Supported Features Matrix

| Feature | Support | Notes |
|---|---|---|
| **Market Data** | | |
| Quote Ticks | ✅ Yes | Real-time Bid/Ask |
| Time Bars | ✅ Yes | Aggregated internally by Nautilus |
| Order Book | ❌ No | Not supported by OANDA V20 |
| Historical Data | ❌ No | Use Parquet files (`NautilusDataLoader`) |
| **Execution** | | |
| Market Orders | ✅ Yes | |
| Limit Orders | ✅ Yes | GTC by default |
| Stop/Trailing | ❌ No | Logic handled by Strategy (Soft Stops) |
| Reconciliation | ✅ Yes | Open orders + Open positions |

---

### 2. `titan.data`
Utilities for fetching, validating, and managing historical data.
*   **`titan.data.oanda`**: primitives for OANDA API data requests (candles, instruments).
    *   `fetch_candles(client, instrument, granularity, ...)`: Robust pagination for history.
*   **`titan.data.validation`**: Data integrity checks.
    *   `check_gaps(df)`: Detects missing candles.
    *   `check_outliers(df)`: flags suspicious price spikes.

### 3. `titan.indicators`
High-performance technical indicators optimized for both Numba (backtesting) and standard Python (live).
*   **`titan.indicators.common`**: Shared logic for SMA, EMA, RSI, MACD.
*   **`titan.indicators.chaos`**: Ehlers-based non-linear indicators (Gaussian Channel, Fractal Dimension).

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
