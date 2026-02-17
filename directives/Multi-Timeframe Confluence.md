# Directive: Multi-Timeframe Confluence Strategy

> **Status: ✅ COMPLETE & OPTIMIZED**

## Goal

Execute high-probability trades by requiring alignment across **Daily (Trend)**, **H4 (Swing)**, and **H1 (Entry)** timeframes. This strategy filters out lower-timeframe noise by ensuring the broader market context is supportive.

## Strategy Logic

**Formula:**
$$ Confluence = w_D \cdot Signal_D + w_{H4} \cdot Signal_{H4} + w_{H1} \cdot Signal_{H1} + w_W \cdot Signal_W $$

Where each $Signal_{TF} \in [-1, +1]$ is derived from:
1.  **Trend:** SMA Crossover (Fast > Slow = +0.5, else -0.5)
2.  **Momentum:** RSI (> 50 = +0.5, else -0.5)

**Entry Condition:**
-   **Long:** Confluence $\ge +0.10$
-   **Short:** Confluence $\le -0.10$

### Exit Logic (Signal Only)
-   **No Trailing Stops:** Backtests proved that tight stops destroy performance (-90% return). We rely purely on the signal.
-   **Long Exit:** Confluence drops below +0.10 (Neutral) or flips Short.
-   **Short Exit:** Confluence rises above -0.10 (Neutral) or flips Long.

### Risk Management
-   **Size:** Volatility-Adjusted (1% Risk Equivalent).
-   **Formula:** $\text{Units} = \frac{\text{Equity} \times 0.01}{2 \times \text{ATR}}$
-   **Logic:** Size the trade *as if* it had a 2 ATR stop, but **do not place the stop**. This keeps exposure constant per unit of volatility without getting whipsawed.
-   **Cap:** Max 5x Leverage.

## Optimized Configuration (Stage 3 Results - M5 Deployment)

Through extensive parameter sweeping (Stages 1-3), the following configuration yielded the best stability for the **5-Minute** deployment:

### 1. Timeframe Weights
| Timeframe | Weight | Role |
|---|---|---|
| **H4** | **0.40** | **Dominant Trend Bias** (Primary Driver) |
| **H1** | **0.40** | Swing Confirmation |
| **M5** | **0.20** | Entry Trigger |
| *D/W* | *0.00* | *Excluded (Too slow for M5 scalping)* |

### 2. Indicator Parameters
| TF | MA Type | Fast | Slow | RSI Period |
|---|---|---|---|---|
| **All** | **WMA** | 20 | 50 | 14 |

*Note: Determining specific periods per timeframe yielded no benefit over a robust global default.*

## Performance (EUR/USD, Jan-Feb 2026)

| Metric | Practice Live (2 Weeks) |
|---|---|
| **Return** | **+0.77%** |
| **Drawdown** | **< 1%** |
| **Stability** | **High** |

## Execution

### Run Backtest
To verify performance with current configuration:
```bash
uv run python scripts/run_backtest_mtf.py
```

### Run Optimization
To re-optimize (e.g., for a different pair):
```bash
# Stage 1: Threshold & MA Type
uv run python research/mtf/run_optimisation.py

# Stage 2: Timeframe Weights
uv run python research/mtf/run_stage2.py

# Stage 3: Period Tuning
uv run python research/mtf/run_stage3.py
```

## Configuration File
Settings are stored in `config/mtf.toml`. This file is automatically updated by `run_mtf_stage3.py`.

## Live Execution (Practice Mode)

To deploy this strategy to the **OANDA Practice Environment**:

### 1. Prerequisites
- `OANDA_ACCOUNT_ID` and `OANDA_ACCESS_TOKEN` set in `.env`.
- `OANDA_ENVIRONMENT=practice` in `.env`.

### 2. Run Command
```bash
uv run python scripts/run_live_mtf.py
```

### 3. Implementation Details
- **Runner:** `scripts/run_live_mtf.py`
- **Strategy Class:** `titan.strategies.mtf.strategy` (`MTFConfluenceStrategy`).
- **Bar Types:** Requires explicit OANDA-specific BarType strings (e.g., `EUR/USD.OANDA-1-HOUR-MID-INTERNAL`) to ensure correct subscription.
- **Warmup:** The strategy automatically loads historical data from `data/` (parquet) to warm up the indicators instantly. No waiting for live bars required.

## Troubleshooting & Implementation Notes

### Position Retrieval
- **Error:** `TypeError: Argument 'position_id' has incorrect type`
- **Cause:** Calling `cache.position(instrument_id)` instead of `cache.positions(instrument_id=...)`.
- **Fix:** `cache.position()` expects a specific `PositionId` UUID. To find positions by instrument, use:
    ```python
    positions = self.cache.positions(instrument_id=self.instrument_id)
    position = positions[-1] if positions else None
    ```
