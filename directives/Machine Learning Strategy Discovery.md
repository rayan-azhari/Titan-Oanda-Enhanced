# Directive: Machine Learning Strategy Discovery

## Goal

Train and validate a Machine Learning model to predict **price direction** (Classification) or **future returns** (Regression), utilising VectorBT Pro for efficient pipeline management.

## Inputs

- Raw Parquet data from `data/`
- Raw Parquet data from `data/`
- Feature definitions from `config/features.toml` (auto-tuned by VBT)

## Execution Steps

### 1. Unified Pipeline Execution

- Run `research/ml/run_pipeline.py`.
- This script handles:
    1.  **Feature Engineering**: Builds 30+ features (Volume, Momentum, MTF Bias).
    2.  **Target Engineering**: 3-class target (LONG/SHORT/FLAT) with 0.5 ATR threshold.
    3.  **Model Training**: Walk-forward cross-validation (5 folds, GradientBoosting/XGBoost).
    4.  **VBT Backtest**: Tests base signals and optimizes exits (Fixed vs Trailing stops) on OOS data.

> [!CAUTION]
> **Look-ahead Bias:** Ensure targets are shifted backwards by 1 period. The script handles this automatically.

### 2. Performance Evaluation

- Review the console output and JSON report in `.tmp/reports/`.
- Key metrics: **Signal Sharpe Ratio**, **Win Rate**, and **OOS Equity Curve**.

### 3. Model Serialisation

- The best performing model (highest Validation Sharpe) is automatically saved to `models/` as a `.joblib` file.

### 4. Deployment

- The saved model is automatically picked up by the live engine.
- See: `directives/Live Deployment and Monitoring.md`
- Run: `scripts/run_live_ml.py`

## 5. Common Errors

### `TypeError: Argument 'position_id' has incorrect type`
- **Context:** Occurs during `on_bar` when checking current positions using `cache.position()`.
- **Cause:** Using `cache.position(instrument_id)` which expects a specific position UUID.
- **Solution:** Use `cache.positions(instrument_id=...)` to list all positions for that instrument, then select the last one.
    ```python
    positions = self.cache.positions(instrument_id=self.instrument_id)
    position = positions[-1] if positions else None
    ```