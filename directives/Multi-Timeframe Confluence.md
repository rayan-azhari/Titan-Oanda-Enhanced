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

## Optimized Configuration (Stage 3 Results)

Through extensive parameter sweeping (Stages 1-3), the following configuration yielded a **Combined Sharpe of 1.75** on EUR/USD:

### 1. Timeframe Weights
| Timeframe | Weight | Role |
|---|---|---|
| **Daily (D)** | **0.60** | **Dominant Trend Bias** (Primary Driver) |
| **H4** | **0.25** | Swing Confirmation |
| **H1** | **0.10** | Entry Timing (Fine-tuning) |
| **Weekly (W)** | **0.05** | Minimal Regime Filter |

### 2. Indicator Parameters
| TF | Fast SMA | Slow SMA | RSI Period |
|---|---|---|---|
| **D** | 13 | 20 | 14 |
| **H4** | 10 | 50 | 21 |
| **H1** | 10 | 30 | 21 |
| **W** | 13 | 21 | 10 |

## Performance (EUR/USD, 2005-2026)

| Metric | In-Sample (2005-2019) | Out-of-Sample (2019-2026) |
|---|---|---|
| **Sharpe Ratio** | 1.66 | **1.83** |
| **Total Return** | +293% (Avg L/S) | +69% (Avg L/S) |
| **Max Drawdown** | -8.6% | -4.2% |
| **Parity Score** | — | **1.10 (Pass)** |

## Execution

### Run Backtest
To verify performance with current configuration:
```bash
uv run python execution/run_mtf_backtest.py
```

### Run Optimization
To re-optimize (e.g., for a different pair):
```bash
# Stage 1: Threshold & MA Type
uv run python execution/run_mtf_optimisation.py

# Stage 2: Timeframe Weights
uv run python execution/run_mtf_stage2.py

# Stage 3: Period Tuning
uv run python execution/run_mtf_stage3.py
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
uv run python execution/run_live_mtf.py
```

### 3. Implementation Details
- **Runner:** `execution/run_live_mtf.py` (Custom runner, distinct from ML runner).
- **Strategy Class:** `strategies/mtf_strategy.py` (`MTFConfluenceStrategy`).
- **Bar Types:** Requires explicit OANDA-specific BarType strings (e.g., `EUR/USD.OANDA-1-HOUR-MID-INTERNAL`) to ensure correct subscription.
- **Warmup:** The strategy automatically loads historical data from `data/raw/` (parquet) to warm up the indicators instantly. No waiting for live bars required.
