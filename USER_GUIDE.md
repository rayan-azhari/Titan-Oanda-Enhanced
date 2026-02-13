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

## 🧠 Phase 4: Train the AI (Machine Learning)

Simple rules (RSI < 30) are good, but AI is better. Let's teach a computer to predict the market.

**Step 1: Create Features (The Inputs)**
The AI needs to "see" the market. We calculate technical indicators for it.
```bash
uv run python execution/build_ml_features.py
```

**Step 2: Train the Model (The Brain)**
We teach the model using the past data.
```bash
uv run python execution/train_ml_model.py
```

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
1.  **Logs:** You will see text scrolling in the console.
    - `[INFO] NautilusTrader started...`
    - `[INFO] Connected to OANDA...`
    - `[INFO] SimplePrinter started...`
2.  **Streaming:** You should see live quotes appearing:
    - `QUOTE EUR_USD: 1.0850 / 1.0851`
3.  **Trading:** If the strategy sees a signal, it will automatically place a trade.

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

## ❓ Troubleshooting / FAQ

**Q: "Command not found: uv"**
A: You didn't install `uv`. Try `pip install uv` again. Or just use `python` instead of `uv run python`.

**Q: "ConnectionResetError"**
A: OANDA disconnected you. The script usually reconnects automatically. If it happens constantly, check your internet.

**Q: "401 Unauthorized"**
A: Your Token is wrong or expired. Generate a new one on the OANDA website and update your `.env` file.

**Q: I don't see any trades!**
A: The market might be closed (Weekends). Or the strategy just hasn't found a good setup yet. Be patient.

---

**Happy Trading!** 🚀
