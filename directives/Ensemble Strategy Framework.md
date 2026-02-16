# Directive: Ensemble Strategy Framework

## Goal

Run **multiple uncorrelated strategies** simultaneously, combining their signals into a weighted ensemble to reduce single-strategy risk and improve capital efficiency.

## Inputs

- Trained `.joblib` models in `models/` (one per strategy)
- Strategy configs in `config/` (one per strategy)
- Ensemble registry in `config/ensemble.toml`

## Architecture

```
┌────────────────────────────────────────────────────┐
│               Ensemble Engine                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ RSI Mean │  │ MA Cross │  │ Breakout Momentum│ │
│  │ Reversion│  │          │  │                  │ │
│  └────┬─────┘  └────┬─────┘  └───────┬──────────┘ │
│       ↓              ↓                ↓            │
│  ┌─────────────────────────────────────────────┐   │
│  │     Weighted Signal Aggregation (≥0.3)      │   │
│  └─────────────────────────────────────────────┘   │
│       ↓                                            │
│  ┌─────────────────────────────────────────────┐   │
│  │   Correlation Filter + Weight Rebalancer    │   │
│  └─────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘
```

## Execution Steps

### 1. Strategy Discovery

- For each strategy type, run the Alpha Research Loop and ML Discovery.
- Save each trained model with a descriptive name.

### 2. Register Strategies

- Add each strategy to `config/ensemble.toml` with initial weights.
- Set `active = true` only after the model passes OOS validation.

### 3. Correlation Check

- **Researcher Agent** runs `research/ml/run_ensemble.py`.
- Strategies with pairwise correlation > `correlation_threshold` (default 0.70) have their weights automatically reduced.

### 4. Signal Generation

- Each active strategy independently predicts on the latest features.
- Signals are combined using performance-weighted voting.
- Ensemble only trades when weighted consensus exceeds ±0.3 threshold.

### 5. Rebalancing

- Weights are recalculated at `rebalance_frequency` (default: weekly).
- Strategies that underperform are downweighted; strong performers are upweighted.

## Safety Constraints

> [!IMPORTANT]
> - Minimum 2 active strategies required to trade
> - No single strategy may hold > 60% of capital
> - All strategies must independently pass OOS validation

## Outputs

- Ensemble signal (BUY / SELL / HOLD)
- Per-strategy signal breakdown
- Correlation matrix report
- Rebalanced weight allocations
