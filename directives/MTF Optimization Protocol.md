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
**Objective:** Determine the best Moving Average type and the required "Conviction" (Threshold) for the strategy. These are the most impactful global variables.

### Parameters to Sweep
- **MA Type**: `SMA`, `EMA`, `WMA` (Applied to *all* timeframes).
- **Confirmation Threshold**: `0.10` to `0.85` (Step 0.05).
    - Determines how much consensus is needed (e.g., 0.55 means strong agreement).

### Validation
- **Metric**: Sharpe Ratio.
- **Stability Check**: `Parity = OOS Sharpe / IS Sharpe`.
- **Selection Criteria**:
    1.  Filter for `Parity > 0.5` (OOS performance retains at least 50% of IS).
    2.  Select configuration with the highest **IS Sharpe** from the stable set.

**Outcome:** Fixed `MA_Type` (e.g., WMA) and `Threshold` (e.g., 0.55) for subsequent stages.

---

## ⚖️ Stage 2: Timeframe Weights (Structure)
**Objective:** Determine the optimal balance of influence between timeframes (e.g., "Is this a D1 trend strategy or an H1 tactical strategy?").

### Parameters to Sweep
- **Weights**: Combinations of M5, H1, H4, D1 (or relevant TFs).
    - *Examples*:
        - **Balanced**: `[0.25, 0.25, 0.25, 0.25]`
        - **Trend Dominant**: `[0.05, 0.15, 0.30, 0.50]`
        - **Tactical**: `[0.10, 0.40, 0.30, 0.20]`
        - **Fast Scalp**: `[0.40, 0.30, 0.20, 0.10]`

### Constraints
- **Fixed**: `MA_Type` and `Threshold` from Stage 1.

### Validation
- **Metric**: Sharpe Ratio & Parity.
- **Selection Criteria**:
    - High Stability (Parity).
    - Robustness across multiple weight profiles is a good sign.
    - Pick the profile that aligns with the desired trade frequency and stability.

**Outcome:** Fixed `Weights` (e.g., `[0.1, 0.3, 0.3, 0.3]`).

---

## 🎯 Stage 3: Indicator Tuning (Fine-Grained)
**Objective:** Tune specific indicator periods (`fast_ma`, `slow_ma`, `rsi_period`) for each timeframe individually.

### Strategy: Greedy Optimization
To avoid combinatorial explosion (tuning 3 params × 4 TFs = 12 dimensions simultaneously), we use a **Greedy** approach based on weight importance.

**Order:** `Dominant TF` → `Secondary TF` → `...` → `Least Impactful TF`.
*(Example for H4/D1 heavy strategy: H4 → H1 → M5 → D)*

### Execution Loop
1.  **Optimize TF #1 (e.g., H4)**:
    - Sweep `fast_ma` (e.g., 10-30), `slow_ma` (e.g., 40-60), `rsi_period` (10-20).
    - **Keep all other TFs at default.**
    - Select best params (IS/OOS validated).
    - **Lock** these parameters for H4.
2.  **Optimize TF #2 (e.g., H1)**:
    - Sweep params for H1.
    - **H4 is fixed to optimized values.** Others are default.
    - Select and **Lock** H1 params.
3.  **Repeat** for remaining TFs.

### Validation
- **Check**: Does tuning actually improve over the baseline? If the improvement is marginal (< 5%), prefer **Defaults** (Robustness > Fitting).

---

## ✅ Final Output
A fully specified `config.toml` containing:
1.  Global `MA_Type` & `Threshold`.
2.  Specific `Weights`.
3.  Per-Timeframe `Fast`, `Slow`, `RSI` settings.

This configuration is then ready for live deployment.
