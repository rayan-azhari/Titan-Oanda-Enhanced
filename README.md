# Titan-Oanda-Algo-Enhanced

> A quantitative **swing trading** system for OANDA — ML-driven strategy discovery, VectorBT optimisation, NautilusTrader execution, and GCE deployment.

📘 **[Read the User Guide](USER_GUIDE.md)** for complete setup and usage instructions.

---

## Architecture

This project follows a **3-layer architecture** that separates *Probabilistic Intent* (AI) from *Deterministic Execution* (Code).

| Layer | Location | Purpose |
|---|---|---|
| **Directive** | `directives/` | Standard Operating Procedures — step-by-step instructions |
| **Orchestration** | Agent context | Intelligent routing — read directives, choose tools, handle errors |
| **Scripts** | `scripts/` | Entry points for trading, backtesting, and utilities |
| **Titan** | `titan/` | Core package (strategies, adapters, utils) |
| **Research** | `research/` | Experimental code (VectorBT, ML training) |

## Trading Style

**Daily swing trading** on higher timeframes:

| Timeframe | Role |
|---|---|
| H1 | Entry/exit timing |
| H4 | Primary analysis |
| D | Trend confirmation |
| W | Regime filter |

## Directory Structure

```
├── AGENTS.MD                      ← Agent system prompt
├── Titan Workspace Rules.md       ← Technical & ML constraints
├── directives/                    ← SOPs
│   ├── Titan Library Reference.md      ← [API DOCS] Detailed Package Guide
│   ├── Titan-Oanda Adapter Guide.md    ← [NEW] Architecture & Usage Guide
│   ├── Workspace Structure.md          ← File Layout Docs
│   ├── Alpha Research Loop (VectorBT).md
│   ├── ... (and other directives)
├── titan/                         ← [CORE] Package (Library Code)
│   ├── adapters/                  ← NautilusTrader Adapters (OANDA)
│   ├── config/                    ← Config Loading
│   ├── data/                      ← Data Fetching & ValidationLogic
│   ├── indicators/                ← Shared Indicators (Numba/VBT)
│   ├── models/                    ← Quant Models (Spread, Slippage)
│   ├── strategies/                ← Production Strategies (MTF, ML)
│   └── utils/                     ← Logging, Ops, Notifications
├── research/                      ← [RESEARCH] Experimental Code
│   ├── alpha_loop/                ← VectorBT Optimization
│   ├── gaussian/                  ← Gaussian Channel Research
│   ├── ml/                        ← ML Pipeline & Feature Selection
│   └── mtf/                       ← MTF Strategy Optimization
├── scripts/                       ← [ENTRY POINTS] Executable Scripts
│   ├── download_data.py           ← Unified Data Downloader
│   ├── check_env.py               ← Environment Verifier
│   ├── run_backtest_mtf.py        ← MTF Switch Backtest
│   ├── run_live_mtf.py            ← Live MTF Strategy
│   ├── run_live_ml.py             ← Live ML Strategy
│   ├── build_docker.py            ← Docker Builder
│   └── ...
├── config/                        ← [CONFIG] TOML Configuration
│   ├── instruments.toml           ← Currency pairs
│   ├── risk.toml                  ← Risk limits
│   └── ...
├── data/                          ← [DATA] Historical Parquet Files
├── models/                        ← [MODELS] Trained .joblib models
├── tests/                         ← [TESTS] Unit Tests
├── .tmp/                          ← [TEMP] Logs, Reports, Intermediate Data
├── pyproject.toml                 ← Dependencies (uv)
└── .env.example                   ← Credential Template
```

## Quick Start

### 1. Install dependencies
```bash
uv sync
```

### 2. Configure credentials
```bash
uv run python scripts/setup_env.py
```
Or manually: `cp .env.example .env` and edit.

### 3. Verify connection
```bash
uv run python scripts/verify_connection.py
```

### 4. Alpha Research Loop
```bash
uv run python scripts/download_data.py                 # Download raw OHLCV
uv run python research/alpha_loop/run_vbt_optimisation.py        # Run VBT parameter sweep
uv run python research/gaussian/run_optimisation.py    # Gaussian Channel sweep
uv run python research/alpha_loop/run_feature_selection.py       # Run Feature Selection Bridge
uv run python scripts/run_backtest_mtf.py              # Test MTF Confluence Strategy
```

### 5. ML Strategy Discovery
```bash
# Runs full pipeline: Feature Engineering -> Target Eng -> Training -> OOS Backtest
uv run python research/ml/run_pipeline.py
```

### 6. Ensemble Signal Aggregation
```bash
uv run python research/ml/run_ensemble.py
```

### 7. Deployment (Docker)
```bash
uv run python scripts/build_docker.py
docker run --env-file .env titan-oanda-algo
```

### 8. NautilusTrader Live

#### Start the MTF Confluence Strategy
```bash
# Multi-Timeframe Confluence Strategy (H1 + H4 + D + W)
uv run python scripts/run_live_mtf.py

# OR for the ML Strategy:
uv run python scripts/run_live_ml.py
```
The engine will:
1. **Checks for and automatically downloads latest data** (runs `scripts/download_data.py`).
2. Load instruments from OANDA.
3. Warm up indicators from local Parquet data (`data/EUR_USD_H1.parquet`, etc.).
4. Reconcile open positions with OANDA (if any exist).
5. Subscribe to the live price stream and start processing bars.

A **status dashboard** prints to the Nautilus log on every bar close showing per-timeframe SMA direction, RSI, confluence score, and position state.

#### Stop the Strategy
```powershell
# Option 1: Press Ctrl+C in the terminal running the strategy

# Option 2: Kill from another terminal
Get-Process -Name "python" | Stop-Process -Force
```

#### Monitor
```powershell
# Tail the log file in real time
Get-Content ".tmp/logs/mtf_live_*.log" -Tail 50 -Wait

# Check which python processes are running
Get-Process -Name "python" -ErrorAction SilentlyContinue
```
Logs are stored in `.tmp/logs/` with timestamps (e.g. `mtf_live_20260216_161315.log`).

## Research Tools

| Tool | Role | Cost |
|---|---|---|
| **VectorBT** (free) | Broad parameter sweeps, heatmaps | Free |
| **Backtesting.py** | Visual trade inspection | Free |
| **NautilusTrader** | Final validation with real spread/slippage | Free |
| **VectorBT Pro** | Optional upgrade for large-scale optimisation | ~$25/mo |

## Testing & CI/CD

This project uses **GitHub Actions** for Continuous Integration (`.github/workflows/ci.yml`). Three checks run on every push to `main`:

| Step | Command | Purpose |
|---|---|---|
| **Lint** | `uv run ruff check .` | Style, imports, unused vars |
| **Format** | `uv run ruff format --check .` | Consistent code formatting |
| **Test** | `uv run pytest tests/ -v --tb=short -x` | Unit tests |

### Pre-Push Checklist
Run all three locally before pushing:
```bash
# 1. Install dev tools (once)
uv sync --extra dev

# 2. Lint + auto-fix
uv run ruff check . --fix

# 3. Auto-format
uv run ruff format .

# 4. Run tests
uv run pytest tests/ -v
```
If all pass locally with zero errors, CI will also pass.

> **📖 Full CI/CD troubleshooting guide:** See [USER_GUIDE.md § CI/CD Pipeline & Code Quality](USER_GUIDE.md#-cicd-pipeline--code-quality).

## Roadmap

- [x] Ensemble / multi-strategy framework
- [x] Time-varying spread model
- [x] Multi-timeframe confluence signals (H1 + H4 + D + W)
- [x] ML Strategy Discovery (XGBoost + Walk-Forward Validation)
- [x] Dockerization for cloud deployment
- [x] VBT → ML Feature Selection Bridge (auto-tune indicators, feed into ML)
- [x] Model → Live Engine Bridge (deploy .joblib models to NautilusTrader)
- [x] Gaussian Channel Strategy (Ehlers filter + Numba + VBT optimisation)
- [x] Adapter Reconciliation (position sync on engine restart)
- [x] Live Trading Execution (Verified Entry/Exit/Reconciliation)
- [x] Data Client Streaming Fix (4 bugs in subscribe/parse/publish pipeline)
- [ ] Configure Slack Alerts for live trading monitoring
- [ ] VectorBT Pro upgrade for production-scale mining
- [ ] **Strategy Tests:** Add integration tests for `mtf_confluence.py` with fixed data inputs
- [x] **Refactor:** Move `run_*.py` scripts to `scripts/` directory
- [ ] **CI/CD:** Add end-to-end "dry run" test for key scripts

## Rules of Engagement

See [Titan Workspace Rules.md](Titan%20Workspace%20Rules.md) for the full constraints. Key rules:

- **`uv` only** — no bare `pip` installs
- **`decimal.Decimal`** for all financial types
- **`random_state=42`** — always
- **No look-ahead bias** — features lagged, targets future-derived
- **Google Style Guide** for all code
