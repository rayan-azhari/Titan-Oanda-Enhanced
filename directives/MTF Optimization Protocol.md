# MTF Strategy Optimization Protocol (VectorBT)

This document outlines the standard 3-Stage Optimization Process for Multi-Timeframe (MTF) Confluence strategies using VectorBT.

**Goal:** efficiently tune high-dimensional strategy parameters without overfitting, using a greedy approach and rigorous In-Sample (IS) / Out-of-Sample (OOS) validation.

---

## 🏗️ Prerequisites
- **Data**: Parquet files for all target timeframes (e.g., M5, H1, H4, D) loaded.
- **Base Strategy**: MTF Confluence logic implemented in VectorBT (Signal Broadcasting, Weighted Scoring).
- **Split**: Data split into 70% In-Sample (IS) and 30% Out-of-Sample (OOS).

---

## 🔄 Stage 1: Global Parameters (Regime & Sensitivity)
**Objective:** Determine the best Moving Average type and the required "Conviction" (Threshold).

### Execution
Run the Stage 1 sweeper:
```bash
uv run python research/mtf/run_optimisation.py
```
> **Auto-Save:** The best `MA_Type` and `Threshold` are automatically saved to `.tmp/mtf_state.json`.

### Parameters Swept
- **MA Type**: `SMA`, `EMA`, `WMA` (Applied to *all* timeframes).
- **Confirmation Threshold**: `0.10` to `0.85` (Step 0.05).

### Output
- **Console**: prints the Best Parameter Set (e.g., `WMA`, `0.55`).
- **Report**: `.tmp/reports/mtf_stage1_scoreboard.csv`
- **Action**: Update `mtf_strategy_5m_stage2.py` with these winning values before proceeding.

---

## ⚖️ Stage 2: Timeframe Weights (Structure)
**Objective:** Determine the optimal balance of influence between timeframes (e.g., "Is this a D1 trend strategy or an H1 tactical strategy?").

### Execution
Run the Stage 2 sweeper:
```bash
uv run python research/mtf/mtf_strategy_5m_stage2.py
```
> **Auto-Load:** Automatically loads `MA_Type` and `Threshold` from Stage 1 (if available).
> **Auto-Save:** Saves the winning `Weights` to `.tmp/mtf_state.json`.

### Parameters Swept
- **Weights**: Grid search of heuristic weight distributions (e.g., Balanced `[0.25, 0.25...]` vs Trend `[0.05, 0.15, 0.30, 0.50]`).
- **Fixed**: `MA_Type` and `Threshold` (from Stage 1).

### Output
- **Console**: prints the Best Weight Config (e.g., `Balanced Higher`).
- **Report**: `.tmp/reports/mtf_stage2_weights.csv`
- **Action**: Update `mtf_strategy_5m_stage3.py` with these winning weights.

---

## 🎯 Stage 3: Indicator Tuning (Fine-Grained)
**Objective:** Tune specific indicator periods (`fast_ma`, `slow_ma`, `rsi_period`) for each timeframe individually.

### Execution
Run the Stage 3 sweeper:
```bash
uv run python research/mtf/mtf_strategy_5m_stage3.py
```
> **Auto-Load:** Loads global params (Stage 1) and weights (Stage 2).
> **Auto-Save:** Saves the detailed indicator configuration.

### Strategy: Greedy Optimization
To avoid combinatorial explosion, we optimize timeframes one by one in order of importance:
**Order:** `H4` → `H1` → `M5` → `D`

1.  **Optimize H4**: Sweep params, keeping others default. Lock best H4 params.
2.  **Optimize H1**: Sweep params, keeping H4 fixed (optimized). Lock best H1 params.
3.  **Repeat**: For M5 and D1.

### Output
- **Console**: prints final optimized parameters for all timeframes.
- **Report**: `.tmp/reports/mtf_stage3_params.csv`

---

## ✅ Next Step: Deployment
Once you have the final configuration from Stage 3:
1.  Update `config/mtf.toml` with the optimized values.
2.  Proceed to **[Strategy Deployment Protocol](Strategy%20Deployment%20Protocol.md)** to go live.

