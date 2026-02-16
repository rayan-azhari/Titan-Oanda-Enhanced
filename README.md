# Titan-Oanda-Algo

> A quantitative **swing trading** system for OANDA — ML-driven strategy discovery, VectorBT optimisation, NautilusTrader execution, and GCE deployment.

📘 **[Read the User Guide](USER_GUIDE.md)** for complete setup and usage instructions.

---

## Architecture

This project follows a **3-layer architecture** that separates *Probabilistic Intent* (AI) from *Deterministic Execution* (Code).

| Layer | Location | Purpose |
|---|---|---|
| **Directive** | `directives/` | Standard Operating Procedures — step-by-step instructions |
| **Orchestration** | Agent context | Intelligent routing — read directives, choose tools, handle errors |
| **Execution** | `execution/` | Deterministic Python scripts — API calls, data processing, ML training |

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
│   ├── Alpha Research Loop (VectorBT).md
│   ├── Machine Learning Strategy Discovery.md
│   ├── Nautilus-Oanda Adapter Construction.md
│   ├── Strategy Validation (Backtesting.py).md
│   ├── Ensemble Strategy Framework.md
│   ├── Multi-Timeframe Confluence.md      ← (SMA + RSI Optimized)
│   ├── Gaussian Channel Strategy Porting.md
│   ├── Gaussian Channel Confluence Strategy.md ← (Superseded)
│   ├── Live Deployment and Monitoring.md
│   └── Workspace Initialisation.md
├── execution/                     ← Python scripts
│   ├── setup_env.py               ← Interactive .env setup
│   ├── verify_connection.py       ← OANDA connection test
│   ├── download_oanda_data.py     ← Historical H1/H4/D/W OHLC data
│   ├── validate_data.py           ← Data quality checks
│   ├── nautilus_oanda/            ← Custom NautilusTrader Adapter
│   │   ├── config.py              ← Configuration
│   │   ├── data.py                ← Streaming DataClient
│   │   ├── execution.py           ← Order ExecutionClient
│   │   ├── instruments.py         ← InstrumentProvider
│   │   └── parsing.py             ← OANDA <-> Nautilus mapper
│   ├── indicators/                ← Custom VectorBT indicators
│   │   └── gaussian_filter.py     ← Ehlers Gaussian Channel (Numba + VBT)
│   ├── spread_model.py            ← Time-varying spread estimation
│   ├── run_vbt_optimisation.py    ← VectorBT parameter sweep + OOS validation
│   ├── run_gaussian_optimisation.py ← Gaussian Channel parameter sweep
│   ├── mtf_confluence.py          ← Multi-timeframe signal alignment
│   ├── run_feature_selection.py   ← VBT → ML Feature Selection Bridge
│   ├── build_ml_features.py       ← Feature matrix (X) + target (y) + MTF
│   ├── train_ml_model.py          ← Walk-forward ML training
│   ├── run_backtesting_validation.py ← Backtesting.py visual audit
│   ├── run_ensemble.py            ← Multi-strategy signal aggregation
│   ├── rate_limiter.py            ← Token bucket for OANDA API
│   ├── parse_oanda_instruments.py ← Legacy instrument parser
│   ├── run_live.py                ← Legacy Python-only engine (placeholder)
│   ├── run_nautilus_live.py       ← NautilusTrader Live Engine
│   ├── fetch_eur_usd.py           ← OANDA API Data Downloader (Raw Parquet)
│   ├── run_mtf_backtest.py        ← Multi-Timeframe Confluence Strategy (VBT)
│   ├── run_ml_strategy.py         ← End-to-End ML Pipeline (Feature Eng + Train + OOS)
│   ├── kill_switch.py             ← Emergency: flatten all positions
│   ├── build_docker_image.py      ← Docker container for GCE
│   └── send_notification.py       ← Slack alert integration
├── config/                        ← TOML configuration
│   ├── instruments.toml           ← Currency pairs & granularities
│   ├── features.toml              ← Technical indicator definitions
│   ├── strategy_config.toml       ← Optimised strategy parameters
│   ├── training.toml              ← ML model & hyperparameters
│   ├── risk.toml                  ← Position & risk limits
│   ├── spread.toml                ← Session-based spread estimates
│   ├── ensemble.toml              ← Multi-strategy registry & weights
│   ├── mtf.toml                   ← Multi-timeframe weights & params
│   └── gaussian_channel_config.toml ← Gaussian Channel optimised params
├── models/                        ← Deliverable: trained .joblib models
├── tests/                         ← Unit tests
├── .tmp/                          ← Intermediate: raw data, reports, logs
├── pyproject.toml                 ← Dependencies (managed by uv)
└── .env.example                   ← Credential template
```

## Quick Start

### 1. Install dependencies
```bash
uv sync
```

### 2. Configure credentials
```bash
uv run python execution/setup_env.py
```
Or manually: `cp .env.example .env` and edit.

### 3. Verify connection
```bash
uv run python execution/verify_connection.py
```

### 4. Alpha Research Loop
```bash
uv run python execution/fetch_eur_usd.py              # Download raw OHLCV
uv run python execution/run_vbt_optimisation.py        # Run VBT parameter sweep
uv run python execution/run_gaussian_optimisation.py   # Gaussian Channel sweep
uv run python execution/run_feature_selection.py       # Run Feature Selection Bridge
uv run python execution/run_mtf_backtest.py            # Test MTF Confluence Strategy
```

### 5. ML Strategy Discovery
```bash
# Runs full pipeline: Feature Engineering -> Target Eng -> Training -> OOS Backtest
uv run python execution/run_ml_strategy.py
```

### 6. Ensemble Signal Aggregation
```bash
uv run python execution/run_ensemble.py
```

### 7. Deployment (Docker)
```bash
uv run python execution/build_docker_image.py
docker run --env-file .env titan-oanda-algo
```

### 8. NautilusTrader Live
```bash
# Deploys the latest trained model from models/ to OANDA live trading
# - Auto-loads latest .joblib model
# - Auto-warms up strategy with local Parquet data for instant readiness
# - Ensures OANDA_ACCOUNT_ID and OANDA_ACCESS_TOKEN are set in .env
uv run python execution/run_nautilus_live.py

# OR for the Multi-Timeframe Confluence Strategy:
uv run python execution/run_live_mtf.py
```

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
- [ ] Configure Slack Alerts for live trading monitoring
- [ ] VectorBT Pro upgrade for production-scale mining

## Rules of Engagement

See [Titan Workspace Rules.md](Titan%20Workspace%20Rules.md) for the full constraints. Key rules:

- **`uv` only** — no bare `pip` installs
- **`decimal.Decimal`** for all financial types
- **`random_state=42`** — always
- **No look-ahead bias** — features lagged, targets future-derived
- **Google Style Guide** for all code
