# Titan-Oanda-Algo User Guide

Welcome to the **Titan-Oanda-Algo** system. This guide covers the complete workflow for operating the algorithmic trading platform, from initial setup to live trading.

## 1. Prerequisites & Installation

### Requirements
- **Python 3.11+**
- **OANDA Account** (Practice or Live)
- **Git**

### Installation
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/rayan-azhari/Titan-Oanda-Algo.git
    cd Titan-Oanda-Algo
    ```

2.  **Install Dependencies:**
    We recommend using `uv` for fast dependency management, but standard `pip` works too.
    ```bash
    # Option A: Using uv (Recommended)
    pip install uv
    uv sync

    # Option B: Using pip
    pip install -r pyproject.toml
    ```

### Environment Setup
1.  **Run the Setup Script:**
    This interactive script will create your `.env` file.
    ```bash
    uv run python execution/setup_env.py
    ```
    *You will need your OANDA `Account ID` and `Access Token`.*

2.  **Verify Connection:**
    Ensure your credentials are correct and you can connect to OANDA.
    ```bash
    uv run python execution/verify_connection.py
    ```

---

## 2. Workflow: Data Acquisition

Before testing or training, you need historical data.

### Download Historical Data
Use the downloader script to fetch OHLCV data for all instruments defined in `config/instruments.toml`.

```bash
uv run python execution/download_oanda_data.py
```
- **Output:** Parquet files stored in `data/raw/` (e.g., `EUR_USD_M15.parquet`).
- **Features:** Automatically resumes interrupted downloads and handles rate limits.

---

## 3. Workflow: Strategy Development & Backtesting

The system uses **VectorBT** for high-performance backtesting and parameter optimization.

### Run Optimization
To find the best parameters (e.g., RSI period, entry thresholds) for a strategy:

```bash
uv run python execution/run_vbt_optimisation.py
```
- **What it does:** Runs thousands of backtests across different parameter combinations.
- **Output:**
    - Heatmaps in `reports/` (e.g., `sharpe_heatmap_EUR_USD_RSI.html`).
    - Best parameter sets printed to console.

---

## 4. Workflow: Machine Learning Pipeline

For more advanced strategies, the system includes an ML pipeline using **XGBoost** and **RandomForest**.

### Step 1: Feature Engineering
Generate technical indicators and target variables from raw data.

```bash
uv run python execution/build_ml_features.py
```
- **Output:** Processed feature datasets in `data/processed/`.

### Step 2: Train Models
Train models to predict future price direction.

```bash
uv run python execution/train_ml_model.py
```
- **Output:**
    - Trained models saved to `models/` (e.g., `xgb_EUR_USD.joblib`).
    - Performance reports (Classification Report, Confusion Matrix).

---

## 5. Workflow: Live Trading

The core of the system is the **NautilusTrader** node, which executes strategies in real-time.

### Running in Live/Practice Mode
Ensure your `.env` is set to `OANDA_ENVIRONMENT=practice` (or `live` for real money).

```bash
uv run python execution/run_nautilus_live.py
```

### What Happens:
1.  **Authentication:** Connects to OANDA using your credentials.
2.  **Instrument Loading:** Fetches available instruments.
3.  **Strategy Loading:** Instantiates the strategy (currently `SimplePrinter` for testing).
4.  **Event Loop:** Starts streaming data and processing events.

### Monitoring
- **Console:** Watch for logs like `[INFO] QUOTE EUR_USD: 1.0500 / 1.0501`.
- **Logs:** Detailed logs are saved to `.tmp/logs/`.
- **Slack:** If configured, critical alerts (e.g., disconnects) are sent to your Slack channel.

---

## 6. Emergency Operations

If the system behaves unexpectedly or you need to exit the market immediately.

### Kill Switch
**WARNING:** This script is destructive.
It will **cancel ALL open orders** and **close ALL open positions** for the account.

```bash
uv run python execution/kill_switch.py
```

---

## 7. Deployment (Docker)

For 24/7 operation, deploy the system as a Docker container.

### Build Image
```bash
uv run python execution/build_docker_image.py
```
This creates a production-ready image `titan-oanda-algo`.

### Run Container
```bash
docker run --env-file .env titan-oanda-algo
```
