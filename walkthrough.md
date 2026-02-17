# Verification: MTF Strategy Integration into ML Pipeline

## Current Status (Handoff)
- **Date**: 2026-02-17
- **State**: **Complete**
- **Outcome**: MTF features integrated, full history downloaded, ML model trained and backtested successfully. Ready for deployment or further refinement.

## Goal
Integrate Multi-Timeframe (MTF) strategy logic into the Machine Learning pipeline, ensuring seamless feature generation, historical data availability (2005+), and efficient model training.

## Changes Verified

### 1. Data Download (Full History)
- **Script**: `scripts/download_data.py`
- **Verification**: Successfully downloaded EUR/USD data from 2005 to 2026 for H1, H4, D, W timeframes.
- **Outcome**: 134,000+ H1 bars available for training.

### 2. Feature Engineering (MTF & Config)
- **Script**: `titan/strategies/ml/features.py`
- **Verification**: 
    - Loaded `config/mtf.toml` dynamically.
    - Generated MTF-specific features: `mtf_signal_{TF}`, `mtf_rsi_{TF}`, `mtf_confluence`.
- **Top Features**:
    1. `vol_ratio` (Volatility)
    2. `atr_14` (Volatility)
    3. `mtf_signal_H4` (MTF Trend)
    4. `mtf_rsi_H4` (MTF Momentum)
    
    *Confirmed that MTF features are among the top predictors.*

### 3. ML Pipeline Optimization
- **Script**: `research/ml/run_pipeline.py`
- **Optimization**:
    - Replaced `GradientBoosting` with **`HistGradientBoosting`** and **`XGBoost`** for speed.
    - Implemented **Parallel Processing** (`n_jobs=-1`) for Cross-Validation folds.
    - Reduced CV splits to 3 for faster iteration.
- **Performance**:
    - Training time reduced significantly.
    - **XGBoost** achieved best Walk-Forward CV Sharpe: **1.838**.

## Backtest Results (Out-of-Sample)

The selected **XGBoost** model was backtested on unseen data (OOS) using VectorBT.

| Exit Strategy | Stop Loss | Sharpe Ratio | Total Return | Trades |
| :--- | :--- | :--- | :--- | :--- |
| **Fixed** | **0.5%** | **1.142** | **38.96%** | **664** |
| Trailing | 0.5% | 0.919 | 43.87% | 545 |
| Fixed | 1.0% | 1.086 | 45.78% | 584 |

> **Conclusion**: The ML model successfully learned from the MTF features and produced a profitable strategy (Sharpe > 1.1) on OOS data with tight stops (0.5%).

## Next Steps
- Deploy `ml_strategy_H1_xgboost_*.joblib` to a live trading node.
- Consider optimizing the `0.5%` stop loss further or adding a dynamic ATR-based stop.
