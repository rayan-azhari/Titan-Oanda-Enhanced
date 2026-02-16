# Titan-Oanda Workspace Structure

This document outlines the file organization of the Titan-Oanda project.

## 📦 Root Directory

| Directory/File | Description |
|---|---|
| **`titan/`** | **Core Package**. Contains all reusable logic, models, and adapters. |
| **`research/`** | **Research Lab**. Experimental code, backtesting, and ML pipelines. |
| **`scripts/`** | **Entry Points**. User-facing scripts for running the system. |
| **`config/`** | **Configuration**. TOML files for strategy parameters and risk. |
| **`data/`** | **Data Store**. Historical market data in Parquet format. |
| **`tests/`** | **Test Suite**. Unit and integration tests. |
| `README.md` | Project overview and quick start guide. |
| `USER_GUIDE.md` | Detailed manual for operators. |

---

## 🏗️ Detailed Structure

### 1. `titan/` (The Engine)
*Library code only. No executable scripts.*

```text
titan/
├── adapters/
│   └── oanda/          # NautilusTrader OANDA Adapter
├── config/             # Config loading utilities
├── data/
│   ├── oanda.py        # OANDA API fetching logic
│   └── validation.py   # Data integrity checks
├── indicators/         # High-performance indicators (Numba)
├── models/             # Quant models (Spread, Slippage)
├── strategies/         # Production-ready strategies
│   ├── mtf/            # Multi-Timeframe Confluence
│   └── ml/             # Machine Learning execution
└── utils/              # Logging and notifications
```

### 2. `research/` (The Lab)
*Experimental code. Output feeds into config/ or titan/ models.*

```text
research/
├── alpha_loop/         # VectorBT optimization loop
├── gaussian/           # Gaussian Channel research
├── ml/                 # ML training pipeline & Feature selection
└── mtf/                # MTF strategy optimization
```

### 3. `scripts/` ( The Control Panel)
*Executable scripts to run the system.*

```text
scripts/
├── download_data.py    # Fetch history
├── check_env.py        # Verify environment
├── run_backtest_mtf.py # Run MTF backtest
├── run_live_mtf.py     # Deploy MTF strategy Live
└── run_live_ml.py      # Deploy ML strategy Live
```

### 4. `config/` (The Controls)
*Parameterizing the system.*

| File | Purpose |
|---|---|
| `instruments.toml` | Pairs to trade and download. |
| `risk.toml` | Position sizing and drawdown limits. |
| `mtf.toml` | Parameters for the MTF strategy. |
| `features.toml` | Selected features for the ML model. |
