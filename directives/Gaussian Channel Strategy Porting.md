# Directive: Gaussian Channel Strategy Porting

> **Status: ✅ COMPLETE**

## Goal

Translate the "Gaussian Channel" logic from Pine Script to a high-performance VectorBT indicator and validate its profitability on OANDA data.

## Inputs

- **Source Logic:** "Gaussian Channel" by DonovanWall (TradingView).
- **Data:** EUR/USD H1 (from `data/`).

## Implementation

### Math Translation

- **File:** `titan/indicators/gaussian_filter.py`
- **Engine:** Numba `@njit` for all recursive loops; numpy for array allocation.
- **Formula:** Ehlers Gaussian Filter — cascaded 1-pole EMAs applied N times (N = poles).
- **Alpha:** Solved from the -3 dB cutoff equation for N cascaded EMAs.
- **Bands:** True Range is filtered by the same Gaussian logic, then scaled by `sigma` to form Upper/Lower bands around the Middle Line.

### VectorBT Factory

- **Wrapper:** `GaussianChannel` built via `vbt.IndicatorFactory.from_custom_func`.
- **Inputs:** `high`, `low`, `close`.
- **Params:** `period`, `poles`, `sigma`.
- **Outputs:** `upper`, `lower`, `middle`.

### Optimisation Script

- **File:** `research/gaussian/run_optimisation.py`
- **Ranges:**
  - `period`: 50 to 300 (step 10)
  - `poles`: 1, 2, 3, 4
  - `sigma`: 1.5, 2.0, 2.5, 3.0

### Signal Logic

- **Long:** Price crosses above Upper Band (Momentum Breakout) OR Price bounces off Middle Line (Trend Following).
- **Short:** Price crosses below Lower Band.

## Outputs

- `config/gaussian_channel_config.toml` — Optimised parameters.
- `.tmp/reports/gaussian_channel_heatmap.html` — Heatmap (Poles vs Period vs Sharpe).
- `.tmp/reports/gaussian_channel_scoreboard.csv` — Full results table.

## Tests

- `tests/test_gaussian.py` — 4 unit tests (alpha, kernel bounds, VBT single-param, VBT multi-param).

## Usage

```bash
uv run python research/gaussian/run_optimisation.py
```