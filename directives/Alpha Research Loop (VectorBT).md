# Directive: Alpha Research Loop (VectorBT)

## Goal

Identify and optimise a viable **daily swing trading** strategy for **EUR/USD**, **GBP/USD**, and **AUD/USD** using vectorised backtesting on higher timeframes.

## Timeframes

| Granularity | Use Case |
|---|---|
| **H1** | Lowest timeframe — entry/exit timing |
| **H4** | Primary analysis frame |
| **D** | Trend confirmation |
| **W** | Regime filter (trending vs ranging) |

## Inputs

- Instrument list from `config/instruments.toml`
- Parameter ranges (e.g., RSI 10–25, MA 20–100)

## Execution Steps

### 1. Data Ingestion

Run `scripts/download_data.py` to pull **2+ years of H1/H4/D/W OHLC data**.
Store in `data/` as Parquet (automatically handles pagination and resume).

### 2. Data Validation

Run `titan/data/validation.py` to check for gaps, duplicates, and outlier spikes.

### 3. Strategy Optimisation (In-Sample)

- **Researcher Agent** runs `research/alpha_loop/run_vbt_optimisation.py`.
- Uses open-source `vectorbt` (free) — no Pro license needed.
- Data is split **70% in-sample / 30% out-of-sample**.
- Optimisation runs on IS data only.

### 4. Out-of-Sample Validation

- Best candidates from IS are tested on the held-out OOS data.
- **Reject** any candidate whose OOS Sharpe drops below 50% of IS Sharpe (overfitting signal).

### 5. Candidate Selection

- Generate **Sharpe Ratio Heatmap** (plotly).
- **Architect Agent** identifies the "Plateau of Stability".

### 6. Multi-Timeframe Confluence (MTF)

- Run `scripts/run_backtest_mtf.py`.
- Tests strategies that require alignment across H1, H4, D, and W timeframes.
- Generates `mtf_confluence_{IS/OOS}.html` reports.

### 6.5. Gaussian Channel Optimisation

- Run `research/gaussian/run_optimisation.py`.
- Sweeps Ehlers Gaussian Channel parameters (Period × Poles × Sigma) on EUR/USD H1.
- Saves optimised parameters to `config/gaussian_channel_config.toml`.
- Generates a Sharpe heatmap in `.tmp/reports/`.

### 7. Parity Transfer

- Convert optimal parameters into `config/strategy_config.toml`.

### 8. VBT → ML Bridge (Feature Selection)

- Run `research/alpha_loop/run_feature_selection.py`.
- Sweeps 7 indicator families + MTF confluence filters across parameter ranges.
- Scores candidates by **Stability** = `min(IS, OOS) / max(IS, OOS)`.
- Writes winning parameters to `config/features.toml` for the ML pipeline.

> [!NOTE]
> **Upgrade path:** If parameter space grows too large for the free VectorBT, upgrade to VectorBT Pro for `@vbt.chunked` memory management. The API is compatible.

## Outputs

- Optimised strategy configuration
- Performance heatmap *(Artifact)*
- OOS validation report