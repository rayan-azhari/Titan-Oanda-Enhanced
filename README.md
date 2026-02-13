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
│   ├── Multi-Timeframe Confluence.md
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
│   ├── spread_model.py            ← Time-varying spread estimation
│   ├── run_vbt_optimisation.py    ← VectorBT parameter sweep + OOS validation
│   ├── mtf_confluence.py          ← Multi-timeframe signal alignment
│   ├── build_ml_features.py       ← Feature matrix (X) + target (y) + MTF
│   ├── train_ml_model.py          ← Walk-forward ML training
│   ├── run_backtesting_validation.py ← Backtesting.py visual audit
│   ├── train_ml_model.py          ← Walk-forward ML training
│   ├── run_backtesting_validation.py ← Backtesting.py visual audit
│   ├── run_ensemble.py            ← Multi-strategy signal aggregation
│   ├── rate_limiter.py            ← Token bucket for OANDA API
│   ├── parse_oanda_instruments.py ← Legacy instrument parser
│   ├── run_live.py                ← Legacy Python-only engine (placeholder)
│   ├── run_nautilus_live.py       ← NautilusTrader Live Engine
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
│   └── mtf.toml                   ← Multi-timeframe weights & params
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
uv run python execution/download_oanda_data.py
uv run python execution/validate_data.py
uv run python execution/run_vbt_optimisation.py
uv run python execution/mtf_confluence.py
```

### 5. ML Strategy Discovery
```bash
uv run python execution/build_ml_features.py
uv run python execution/train_ml_model.py
```

### 7. Deployment (Docker)
```bash
uv run python execution/build_docker_image.py
docker run --env-file .env titan-oanda-algo
```

### 7. NautilusTrader Live (Recommended)
```bash
# Ensure OANDA_ACCOUNT_ID and OANDA_ACCESS_TOKEN are set in .env
uv run python execution/run_nautilus_live.py
```

## Research Tools

| Tool | Role | Cost |
|---|---|---|
| **VectorBT** (free) | Broad parameter sweeps, heatmaps | Free |
| **Backtesting.py** | Visual trade inspection | Free |
| **NautilusTrader** | Final validation with real spread/slippage | Free |
| **VectorBT Pro** | Optional upgrade for large-scale optimisation | ~$25/mo |

## Roadmap

- [x] Ensemble / multi-strategy framework
- [x] Time-varying spread model
- [x] Multi-timeframe confluence signals (H1 + H4 + D)
- [x] Dockerization for cloud deployment
- [ ] VectorBT Pro upgrade for production-scale mining

## Rules of Engagement

See [Titan Workspace Rules.md](Titan%20Workspace%20Rules.md) for the full constraints. Key rules:

- **`uv` only** — no bare `pip` installs
- **`decimal.Decimal`** for all financial types
- **`random_state=42`** — always
- **No look-ahead bias** — features lagged, targets future-derived
- **Google Style Guide** for all code
