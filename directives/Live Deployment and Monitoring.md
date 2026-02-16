# Directive: Live Deployment and Monitoring

## Goal

Containerise the Nautilus system and deploy it to a low-latency **Google Compute Engine (GCE)** instance.

## Pre-Flight Checklist

> [!CAUTION]
> **All items must pass before deployment.**

- [ ] `config/risk.toml` reviewed — max drawdown, position limits, daily loss cap
- [ ] `scripts/kill_switch.py` tested on practice account
- [ ] OOS Sharpe ≥ 50% of IS Sharpe (from VBT optimisation)
- [ ] Model Sharpe ≥ 1.5 (from `train_ml_model.py`)
- [ ] `titan/data/validation.py` passed on latest data

## Local Practice Deployment

Before deploying to the cloud, verify the strategy in the local **Practice Environment**.

### 1. Configuration
Ensure `.env` is set to practice mode:
```ini
OANDA_ENVIRONMENT=practice
```

### 2. Execution
Run the specific runner for your strategy:

**A. Multi-Timeframe Confluence (Signal-Based)**
```bash
uv run python scripts/run_live_mtf.py
```

**B. ML Strategy (XGBoost)**
```bash
uv run python scripts/run_live_ml.py
```

### 3. Validation
- Check console for `[INFO] Strategy Started`.
- Verify subscriptions (e.g., `EUR/USD.OANDA-1-HOUR...`).
- Wait for "Warmup complete".
- Monitor via OANDA mobile app or web dashboard.

## Cloud Execution Steps (GCE)

### 1. Containerisation

- **DevOps Agent** runs `scripts/build_docker.py`.
- Base: `python:3.11-slim` with `nautilus_trader` wheel.

> [!IMPORTANT]
> **Critical:** Copy the `models/` directory into the container if an ML strategy is active.

### 2. Infrastructure

- Deploy to `europe-west2` (London) for OANDA proximity.
- Use `e2-standard-2` instance type.

### 3. Headless Monitoring (The Guardian)

- Initialise "Guardian" agent in headless mode.
- **Task:** SSH log monitoring for `"ERROR"` strings.
- **Notification:** Trigger Slack alert on failure via `titan/utils/notification.py`.

## Success Criteria

- System running in container on GCE.
- Guardian agent active and reporting heartbeat.