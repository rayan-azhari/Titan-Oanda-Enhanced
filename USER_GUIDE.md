# Titan-Oanda-Algo: The Complete Beginner's Guide

Welcome! This guide is designed to take you from "I have a computer" to "I am running an algorithmic trading system on OANDA."

---

## 🏗️ The Big Picture: How This Works

This system is a **Quant Trading Engine**. It automates the process of finding trading strategies and executing them. Here is the workflow:

1.  **Data Acquisition:** We download historical price data (candles) from OANDA. *You can't test a strategy without history.*
2.  **Strategy Discovery:** We use **VectorBT** to simulate thousands of trading rules (e.g., "Buy when RSI is low") on that history to find what actually makes money.
3.  **Machine Learning:** We train AI models to recognize complex patterns that simple rules miss.
4.  **Live Trading:** We turn on **NautilusTrader**, which connects to OANDA, watches the market in real-time, and executes the winning strategies.

---

## 🚀 Phase 1: Getting Started (Installation)

### Prerequisites (What you need)
1.  **A Computer:** Windows, Mac, or Linux.
2.  **Python 3.11+:** The programming language we use. [Download Here](https://www.python.org/downloads/).
3.  **VS Code:** A good code editor. [Download Here](https://code.visualstudio.com/).
4.  **OANDA Account:** You need an account to trade. Start with a **Practice Account** (Play Money).
    - Go to [OANDA](https://www.oanda.com/).
    - Create an account.
    - Go to "Manage API Access" and generate a **Personal Access Token**.
    - Write down your **Account ID** (format: `000-000-0000000-000`).

### Step 1: Download the Code
Open your terminal (Command Prompt or PowerShell) and run:

```bash
git clone https://github.com/rayan-azhari/Titan-Oanda-Algo.git
cd Titan-Oanda-Algo
```

### Step 2: Install the Brains (Dependencies)
We need to install the libraries that do the math.

```bash
# First, install 'uv' (it makes things faster)
pip install uv

# Now install everything else
uv sync
```
*Tip: If `uv` doesn't work, just use `pip install -r pyproject.toml`.*

### Step 3: Connect Your Account
We need to tell the system your OANDA secret password.

1.  Run the setup script:
    ```bash
    uv run python execution/setup_env.py
    ```
2.  It will ask for your **Account ID** and **Token**. Paste them in.
3.  It creates a hidden file called `.env`.

**Sanity Check:**
Run this verification script. If it works, you are ready to go!
```bash
uv run python execution/verify_connection.py
```
✅ **Success:** You see "Connected to OANDA", your account balance, and open trades.
❌ **Failure:** "Unauthorized" or "Connection Error". Check your Token and Account ID in `.env`.

---

## 📊 Phase 2: Get the Data

We cannot trade without knowing what the market did in the past.

**The Command:**
```bash
uv run python execution/download_oanda_data.py
```

**What it does:**
- Downloads price candles (Open, High, Low, Close) for major currency pairs (EUR/USD, GBP/USD, etc.).
- Saves them as **Parquet** files (highly efficient data files) in the `data/` folder.

**Sanity Check:**
- Look in the `data/raw/` folder. You should see files like `EUR_USD_M15.parquet`.
- If the folder is empty, the download failed. Check your internet or API limits.

---

## 🧪 Phase 3: Find a Winning Strategy (Backtesting)

Now we play "What If?". What if we bought every time RSI was below 30 last year? Would we be rich?

**The Command:**
```bash
uv run python execution/run_vbt_optimisation.py
```

**What it does:**
- Uses **VectorBT** to run thousands of simulations.
- It tests different combinations (e.g., RSI Period 14 vs 21, Threshold 30 vs 25).
- It generates a **Heatmap** showing which settings were profitable.

**Sanity Check:**
- Look in `reports/`. Open `sharpe_heatmap_EUR_USD_RSI.html` in your web browser.
- Green areas = Profitable settings. Red areas = Loss-making settings.

---

## 🔗 Phase 3.5: Feature Selection Bridge (VBT → ML)

This is the **Titan Sequence** — the bridge between simple backtesting and intelligent ML. Instead of feeding the ML model hardcoded indicators, we first use VBT to discover *which* indicator parameters actually work, then feed those into the ML pipeline automatically.

**The Command:**
```bash
uv run python execution/run_feature_selection.py
```

**What it does:**
1. **Single-TF Indicator Sweep** — Tests 7 indicator families across hundreds of parameter combos:

   | Indicator | What it sweeps |
   |---|---|
   | RSI | Window 5–30, entry threshold 15–40 |
   | SMA Cross | Fast 5–30, slow 20–100 |
   | EMA Cross | Fast 5–25, slow 15–60 |
   | MACD | Fast/slow/signal combinations |
   | Bollinger | Window 10–30, std dev 1.5–3.0 |
   | Stochastic | %K 5–21, %D 2–5 |
   | ADX Filter | Period 10–25, threshold 20–30 |

2. **MTF Confluence Sweep** — Tests higher-timeframe bias filters:
   - Which TFs to use as context (D, W, or both)
   - SMA fast/slow speeds on each higher TF
   - RSI period and bullish/bearish threshold per TF

3. **Scoring** — Each combo is backtested on both In-Sample (70%) and Out-of-Sample (30%) data. Scored by:
   - `Stability = min(IS, OOS) / max(IS, OOS)` — penalises overfitting
   - `Score = OOS_Sharpe × Stability` — rewards robust profitability

4. **Auto-Config** — Writes the winning parameters to `config/features.toml`

**Sanity Check:**
- Open `config/features.toml` — you should see tuned values like `rsi = { window = 9, entry = 28 }`
- Open `reports/feature_scoreboard_EUR_USD.json` — all indicators ranked by score

---

## 📈 Phase 3.6: Gaussian Channel Strategy

The **Gaussian Channel** is a volatility-based indicator from the Ehlers Gaussian Filter. Instead of simple moving averages, it uses a cascade of EMAs (controlled by the "poles" parameter) to create a smoother, lower-lag channel. This is useful for catching momentum breakouts and trend-following bounces.

**The Command:**
```bash
uv run python execution/run_gaussian_optimisation.py
```

**What it does:**
1. Loads EUR/USD H1 data from `data/`.
2. Runs the `GaussianChannel` indicator across a parameter grid:

   | Parameter | Range |
   |---|---|
   | Period | 50 – 300 (step 10) |
   | Poles | 1, 2, 3, 4 |
   | Sigma | 1.5, 2.0, 2.5, 3.0 |

3. **Signal Logic:**
   - **Long:** Price crosses above Upper Band (momentum breakout) OR bounces off Middle Line (trend following).
   - **Short:** Price crosses below Lower Band.
4. Calculates Sharpe Ratio for each combo.
5. Saves the best parameters to `config/gaussian_channel_config.toml`.
6. Generates an interactive **Heatmap** (Poles vs Period, coloured by Sharpe).

**Sanity Check:**
- Open `config/gaussian_channel_config.toml` — you should see tuned values like `period = 140`, `poles = 3`, `sigma = 2.0`.
- Open `.tmp/reports/gaussian_channel_heatmap.html` in your browser to see the heatmap.
- Open `.tmp/reports/gaussian_channel_scoreboard.csv` for the full results table.

---

## 🦅 Phase 3.7: Multi-Timeframe Confluence Strategy

This is our **best performing strategy** (Sharpe 1.75 on EUR/USD). It filters out noise by requiring alignment across Daily, H4, and H1 timeframes.

**The Command:**
```bash
uv run python execution/run_mtf_backtest.py
```

**How it works:**
It calculates a weighted "Confluence Score" based on **SMA Crossovers** and **RSI Momentum** across multiple timeframes:

| Timeframe | Weight | Role | Logic |
|---|---|---|---|
| **Daily (D)** | **60%** | Trend Bias | SMA(13, 20) + RSI(14) |
| **H4** | **25%** | Swing Confirm | SMA(10, 50) + RSI(21) |
| **H1** | **10%** | Entry Timing | SMA(10, 30) + RSI(21) |
| **Weekly** | **5%** | Regime Filter | SMA(13, 21) + RSI(10) |

**Signal Logic:**
- **Long:** Confluence Score ≥ +0.10
- **Short:** Confluence Score ≤ -0.10

**Optimization Process:**
We found these exact settings through a rigorous 3-stage sweep:
1.  **Stage 1:** Determined that **SMA** beats EMA, and a low threshold (**0.10**) is better than 0.30.
2.  **Stage 2:** Found that **Daily Trend** is the most important factor (60% weight).
3.  **Stage 3:** Tuned the exact lookback periods for each timeframe independently.

To re-run this optimization yourself (e.g., for a different pair):
```bash
uv run python execution/run_mtf_optimisation.py  # Stage 1
uv run python execution/run_mtf_stage2.py        # Stage 2
uv run python execution/run_mtf_stage3.py        # Stage 3
```

**Sanity Check:**
- Open `config/mtf.toml` to see the optimized parameters.
- Check `.tmp/reports/mtf_confluence_EUR_USD_oos.html` to see the strategy performance.

---

## 🧠 Phase 4: Train the AI (Machine Learning)

Simple rules (RSI < 30) are good, but AI is better. The ML pipeline now **automatically reads** the tuned parameters from `config/features.toml` (written by Phase 3.5). If the config doesn't exist, it falls back to sensible defaults.

**The Command (full pipeline):**
```bash
uv run python execution/run_ml_strategy.py
```

**What it does:**
1. Loads tuned indicator parameters from `config/features.toml`
2. Builds a feature matrix (RSI, SMA, MACD, Bollinger, Stochastic, ADX, MTF bias) using those tuned params
3. Engineers a 3-class target: LONG, SHORT, FLAT
4. Trains XGBoost via walk-forward cross-validation
5. Backtests ML predictions via VectorBT
6. Saves the model if profitable

**Sanity Check:**
- Look in `models/`. You should see `.joblib` files (e.g., `xgb_EUR_USD.joblib`).
- Creating these files means your AI is ready to make decisions.

---

## 💸 Phase 5: Live Trading

This is it. The system connects to OANDA and trades for real.

**⚠️ DANGER ZONE:**
- By default, we run in **PRACTICE** mode.
- To trade real money, you must change `OANDA_ENVIRONMENT` to `live` in your `.env` file. **DO NOT DO THIS UNTIL YOU ARE SURE.**

**The Command:**
```bash
uv run python execution/run_nautilus_live.py
```

**What Happens:**
1.  **Model Loading:** The system finds the latest `.joblib` model in `models/` and loads it.
2.  **Smart Warmup:** It loads recent data history from `data/` so the strategy is ready to trade **immediately** (no waiting weeks for buffers to fill).
3.  **Logs:** You will see text scrolling in the console.
    - `[INFO] NautilusTrader started...`
    - `[INFO] Loaded latest model: xgb_EUR_USD.joblib`
    - `[INFO] Warmed up with 500 bars.`
4.  **Streaming:** You should see live quotes appearing:
    - `QUOTE EUR_USD: 1.0850 / 1.0851`
5.  **Trading:** If the model predicts a profitable move, it will automatically place a trade.

**How to Stop:**
Press `Ctrl + C` in the terminal.

---

## 🚨 Emergency: The Kill Switch

If the bot goes crazy or you just want out **NOW**.

**The Command:**
```bash
uv run python execution/kill_switch.py
```

**What it does:**
1.  **Cancels** all pending orders.
2.  **Closes** all open trades immediately (at market price).
3.  **Stops** everything.

---

## � Optional: Set Up Slack Alerts

Want the bot to message you when it trades?

1.  **Create a Slack App:** Go to [api.slack.com/apps](https://api.slack.com/apps) and click **Create New App** -> **From scratch**. Name it "Titan Bot" and pick your workspace.
2.  **Activate Webhooks:** Click **Incoming Webhooks** in the sidebar and toggle it **On**.
3.  **Add Webhook:** Click **Add New Webhook to Workspace**, pick a channel (e.g., `#trading-logs`), and click **Allow**.
4.  **Copy the URL:** It looks like `https://hooks.slack.com/services/T000.../B000.../XXXX...`.
5.  **Save it:** Add it to your `.env` file manually:
    ```bash
    SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
    ```

---

## �🐳 Advanced: Running in the Cloud (Docker)

If you want the bot to run 24/7 without your laptop being on, you use **Docker**.

1.  **Build the Container:**
    ```bash
    uv run python execution/build_docker_image.py
    ```
2.  **Run It:**
    ```bash
    docker run --env-file .env titan-oanda-algo
    ```

---

## 🔄 CI/CD Pipeline & Code Quality

Every time you push code to GitHub, an automated **CI/CD pipeline** runs to check your code quality and test suite. If it fails, your changes are flagged as broken. **Always run these checks locally before pushing.**

### What the Pipeline Checks

The CI pipeline (`.github/workflows/ci.yml`) runs three checks:

| Step | Command | What it Checks |
|---|---|---|
| **Ruff Linter** | `uv run ruff check .` | Import errors, unused variables, naming rules |
| **Ruff Formatter** | `uv run ruff format --check .` | Code formatting (spacing, quotes, trailing commas) |
| **Pytest** | `uv run pytest tests/ -v --tb=short -x` | Unit tests pass |

### How to Run Checks Locally (Pre-Push Checklist)

**Always run these 3 commands before pushing:**

```bash
# Step 1: Install test dependencies (only needed once)
uv sync --extra dev

# Step 2: Fix lint errors
uv run ruff check . --fix

# Step 3: Auto-format code
uv run ruff format .

# Step 4: Run tests
uv run pytest tests/ -v
```

If all 3 pass locally with **zero errors**, the CI pipeline will also pass.

### Common CI Failures & Fixes

**1. `E402 Module level import not at top of file`**
- **Cause:** Imports appear after `sys.path.insert()` or `load_dotenv()` calls.
- **Fix:** Already suppressed for `execution/*.py` and `tests/*` in `pyproject.toml`. If adding new files, ensure they follow the same pattern or add them to the `per-file-ignores` section.

**2. `F841 Local variable is assigned to but never used`**
- **Cause:** You assigned a variable but never read it.
- **Fix:** Remove it, or prefix with `_` (e.g., `_unused_var = ...`).

**3. `E501 Line too long`**
- **Cause:** A line exceeds 100 characters.
- **Fix:** Break the line. Use parentheses for multi-line strings.

**4. `ImportError` in tests**
- **Cause:** NautilusTrader API changed (classes moved to new modules).
- **Fix:** Check the [NautilusTrader docs](https://nautilustrader.io/docs) for updated import paths. Common moves:
  - `LiveDataClient` → `nautilus_trader.live.data_client`
  - `LiveExecutionClient` → `nautilus_trader.live.execution_client`
  - `CurrencyPair` → `nautilus_trader.model.instruments.currency_pair`

**5. `TypeError` in NautilusTrader objects**
- **Cause:** Constructor signatures changed between versions.
- **Fix:** Use factory methods instead of direct constructors:
  - `Price.from_str("1.05")` instead of `Price(Decimal("1.05"))`
  - `Quantity.from_str("100")` instead of `Quantity(100, 0)`
  - `CurrencyPair.from_dict({...})` instead of `CurrencyPair(...)`

**6. `Would reformat: file.py` (Formatter)**
- **Cause:** Code isn't formatted to Ruff's standard (quotes, spacing, etc.).
- **Fix:** Run `uv run ruff format .` — it auto-formats everything.

### Ruff Configuration

All linting and formatting rules are configured in `pyproject.toml` under `[tool.ruff]`:

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "W", "D"]  # Enabled rule categories
ignore = ["D100", "D102", "D103", "D104", "D107", "D205", "D415", "D417"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["E402", "D"]           # Tests: allow late imports
"execution/*.py" = ["E402"]         # Scripts: allow sys.path before imports
"strategies/*" = ["D"]              # Strategies: skip docstring rules
```

If you add a new directory with Python files, you may need to add it here.

---

## ❓ Troubleshooting / FAQ

**Q: "Command not found: uv"**
A: You didn't install `uv`. Try `pip install uv` again. Or just use `python` instead of `uv run python`.

**Q: "ConnectionResetError"**
A: OANDA disconnected you. The script usually reconnects automatically. If it happens constantly, check your internet.

**Q: "401 Unauthorized"**
A: Your Token is wrong or expired. Generate a new one on the OANDA website and update your `.env` file.

**Q: I don't see any trades!**
A: The market might be closed (Weekends). Or the strategy just hasn't found a good setup yet. Be patient.

**Q: CI pipeline fails with "Would reformat"**
A: Run `uv run ruff format .` locally, commit, and push again. The formatter auto-fixes all formatting.

**Q: CI pipeline fails with "ruff check" errors**
A: Run `uv run ruff check . --fix` locally. It auto-fixes most issues. For remaining errors, read the error message — it tells you the exact file, line, and what's wrong.

**Q: Tests pass locally but fail in CI**
A: Check if you have `.env` variables that tests depend on. CI doesn't have your `.env` file, so tests requiring live OANDA credentials are auto-skipped. If tests fail for a different reason, check the CI logs for the exact error.

**Q: Feature Selection sweep is slow**
A: The sweep tests ~1,200+ parameter combos across 7 indicators + MTF confluence. On H4 data with ~2 years of history, expect 5–15 minutes per pair. To speed up, reduce the parameter ranges in `run_feature_selection.py` (e.g., widen the step sizes).

**Q: `config/features.toml` shows all defaults, nothing tuned**
A: No indicator passed the minimum OOS Sharpe threshold (0.3). This means the asset/timeframe may not have strong single-indicator signals. Try a different pair or timeframe, or lower `min_sharpe_oos`.

---

**Happy Trading!** 🚀
