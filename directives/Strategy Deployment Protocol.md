# Strategy Deployment Protocol

This document outlines the systematic process for moving a strategy from the **Research Phase** (VectorBT/Backtesting) to **Live Deployment** (NautilusTrader/OANDA).

## 1. Prerequisites (Research Completion)
Before deployment, a strategy must pass the **Optimization Protocol**:
- **Protocol**: `directives/MTF Optimization Protocol.md`
- **Output**: A confirmed set of parameters (e.g., MA Type, Thresholds, Weights, Indicator Settings) that have been passed through In-Sample/Out-of-Sample validation.

---

## 2. Code Adaptation (Research → Production)
Research code (often vectorized, typically `research/`) needs to be ported to the Event-Driven Engine (`titan/strategies/`).

### A. Feature Engineering
- **Location**: `titan/strategies/ml/features.py`
- **Action**: Ensure all indicators used in research are implemented here as *reusable functions*.
- **Requirement**: Must support streaming/incremental updates or be efficient enough to recalculate on each bar (depending on engine architecture).
    - *Example*: During the MTF-5m deployment, we added `wma` logic here to match the research findings.

### B. Strategy Logic Class
- **Location**: `titan/strategies/mtf/strategy.py` (or similar)
- **Action**: Update the `NautilusTrader` strategy class to:
    1.  **Load Parameters**: Accept the optimized params via `Config` object or TOML file.
    2.  **Subscribe**: Subscribe effectively to the necessary BarTypes (Timeframes).
    3.  **Calculate**: Use the features from Step A to generate signals exactly as done in research.
    - *Critical*: Ensure `on_bar` logic handles multiple timeframes correctly (buffering, synchronization).

### C. Configuration
- **Location**: `config/<strategy_name>.toml`
- **Action**: Create a dedicated TOML file for the specific optimization (e.g., `mtf_5m.toml`).
- **Content**:
    - Global params (MA Type, Thresholds).
    - Weights.
    - Per-Timeframe indicator settings.
    - Risk parameters (Risk %, Max Leverage).

---

## 3. Data Hygiene & Warmup
**Critical**: Strategies require historical data to calculate initial indicators (MA, RSI, ATR) before the first live tick arrives. Without this, the strategy will either crash or wait hours/days to accumulate enough data.

### A. The Data Source
- **Script**: `scripts/download_data.py`
- **Function**: Downloads OHLCV data from OANDA and saves it as Parquet files in `data/`.
- **Naming Convention**: `data/{SYMBOL}_{GRANULARITY}.parquet` (e.g., `EUR_USD_M5.parquet`).
    - *Note*: The strategy looks for these **exact** filenames.

### B. Warmup Logic
- **Integration**: The runner script (`scripts/run_live_mtf_5m.py`) calls `download_data.py` *before* starting the trading node.
- **Verification**:
    - The runner prints: `📥 Checking for latest data...`
    - If successful: `✅ Data download finished.`
    - If failed: It logs a warning but proceeds with existing disk data.
- **Strategy Loading**:
    - Inside `MTFConfluenceStrategy._warmup_all()`, the code explicitly loads the tail of these parquet files into `self.history`.
    - **Debug Check**: If the dashboard shows "??", it means this loading step failed (file missing or empty).

---

## 4. Execution Infrastructure

### A. Runner Script
- **Location**: `scripts/run_live_<strategy>.py`
- **Action**: Create a dedicated runner script.
    - **Logging**: Configure specific file logging to capture strategy signals.
    - **Data Warmup**: Include a step to download/verify historical data (`scripts/download_data.py`) to ensure the strategy has context on startup.
    - **Instrument Loading**: Fetch tradable instruments from the live provider.
    - **Node Configuration**: Set up `TradingNode` with `OandaDataClient` and `OandaExecutionClient`.

### B. Environment
- **Credentials**: Ensure `.env` has valid `OANDA_ACCOUNT_ID` and `OANDA_ACCESS_TOKEN`.
- **Mode**: Set `OANDA_ENVIRONMENT=practice` for initial testing.

---

## 4. Verification & Launch

### A. Dry Run / Paper Trading
1.  **Start Script**: `python scripts/run_live_<strategy>.py`
2.  **Verify Warmup**: Check logs (`DEBUG` level) to ensure historical data is loaded from disk.
3.  **Verify Dashboard**: Ensure the terminal output shows populated indicators (not "??") and active signal weights.
4.  **Confirm Execution**: Wait for a signal (or force one in logic temporarily) to verify `_open_position` accesses the account and calculates size correctly.

### B. Live Monitoring
- **Logs**: Monitor `.tmp/logs/` for errors or unhandled exceptions.
- **Process**: Run via a process manager (e.g., `systemd`, `supervisord`, or `nohup`) for long-term execution on a server.

---

## Checklist
- [ ] Optimization complete & params locked?
- [ ] Features implemented in shared logic?
- [ ] Strategy class updated to read params?
- [ ] Config TOML created?
- [ ] Runner script created with Logging & Warmup?
- [ ] Dry run successful (Data loaded, Dashboard active)?
