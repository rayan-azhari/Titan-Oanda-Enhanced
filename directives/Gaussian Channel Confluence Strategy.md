# Directive: Gaussian Channel Confluence Strategy

> **Status: ❌ SUPERSEDED**
>
> **Note:** This specific hypothesis was tested but superseded by the general **Multi-Timeframe Confluence** framework (`directives/Multi-Timeframe Confluence.md`), which optimized towards an **SMA + RSI** approach (Sharpe 1.75) rather than Gaussian Channels for the MTF components.

## Goal

Original Hypothesis: Implement a Multi-Timeframe (MTF) Gaussian Channel strategy where entries are taken on H1 only when aligned with H4, D1, and W1 trends.

**Outcome:** The general MTF optimization sweep (Stage 1) found that simple SMAs outperformed Gaussian Channels for the trend filter component. We proceed with the SMA-based MTF strategy.

## Execution Steps

### 1. Indicator Construction (Numba)

- ✅ Already implemented in `titan/indicators/gaussian_filter.py`.
- Accepts dynamic `poles` (smoothness) and `period` inputs.
- Wrapped in `vbt.IndicatorFactory` as `GaussianChannel`.

### 2. The MTF Factory (VectorBT)

Create a script `research/gaussian/run_confluence.py`:

- **Step A: Resample** — Take H1 Close prices and resample to H4, D1, W1.
- **Step B: Calculate** — Run the Gaussian Channel logic on all four series independently.
- **Step C: Broadcast** — Use `vbt.base.reshaping.broadcast_to()` to expand the H4, D1, and W1 indicator values back to the shape of the H1 index.
- **Result:** Every H1 bar now knows what the Weekly Trend was at that specific moment (forward-filled).

### 3. Confluence Logic

Define "Trend Up": Price > Gaussian Middle Line.

**Signal Generation:**
- `is_weekly_up` = W1_Price > W1_Midline
- `is_daily_up` = D1_Price > D1_Midline
- `is_h4_up` = H4_Price > H4_Midline
- `entry_signal` = (`is_weekly_up` AND `is_daily_up` AND `is_h4_up`) AND (H1_Price crosses above H1_Upper_Band)

> **Note:** Validating "Full Confluence" (4 timeframes) vs "Partial Confluence" (3 timeframes) is part of the optimisation.

### 4. Optimisation Loop

- Optimise the lag (Poles) and period specifically for the higher timeframes.
- **Hypothesis:** The Weekly channel likely needs different settings (e.g., slower) than the H1 channel.

## Outputs

- `config/gaussian_confluence_config.toml`
- A plot showing the H1 price with the "shadow" of the Weekly Channel overlaid.