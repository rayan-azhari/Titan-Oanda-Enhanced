"""run_mtf_portfolio.py — Realistic Portfolio Simulation.

Simulates the MTF Confluence strategy with proper risk management:
- Unified Long/Short Portfolio (reversals handled automatically)
- Volatility-Adjusted Sizing (1% Equity Risk per trade)
- Trailing Stop Loss (2.0 * ATR)
- Leverage Cap (5.0x)

Usage:
    uv run python execution/run_mtf_portfolio.py
"""

import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = PROJECT_ROOT / "config" / "mtf.toml"

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.")
    sys.exit(1)

from execution.spread_model import build_spread_series

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

INIT_CASH = 100_000.0  # $100k account
RISK_PER_TRADE = 0.01  # 1% equity risk per trade
ATR_PERIOD = 14  # For volatility sizing
STOP_ATR_MULT = 2.0  # Stop loss distance multiplier
MAX_LEVERAGE = 5.0  # Hard cap on position size


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def load_data(pair: str, gran: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{pair}_{gran}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def load_mtf_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_tf_signal(close: pd.Series, fast_ma: int, slow_ma: int, rsi_period: int) -> pd.Series:
    fast = close.rolling(fast_ma).mean()
    slow = close.rolling(slow_ma).mean()
    rsi = compute_rsi(close, rsi_period)
    ma_sig = pd.Series(np.where(fast > slow, 0.5, -0.5), index=close.index)
    rsi_sig = pd.Series(np.where(rsi > 50, 0.5, -0.5), index=close.index)
    return ma_sig + rsi_sig


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr0 = abs(high - low)
    tr1 = abs(high - close.shift())
    tr2 = abs(low - close.shift())
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_confluence(pair: str, mtf_config: dict) -> tuple[pd.Series, pd.DataFrame]:
    """Calculate weighted confluence score."""
    weights = mtf_config.get("weights", {})
    primary_df = load_data(pair, "H4")
    if primary_df is None:
        raise ValueError("H4 data missing")

    primary_index = primary_df.index
    signals_sum = pd.Series(0.0, index=primary_index)
    total_weight = 0.0

    print("  Timeframes:")
    for tf in ["H1", "H4", "D", "W"]:
        w = weights.get(tf, 0.0)
        if w == 0:
            continue

        cfg = mtf_config.get(tf, {})
        df = load_data(pair, tf)
        if df is None:
            continue

        sig = compute_tf_signal(
            df["close"],
            cfg.get("fast_ma", 20),
            cfg.get("slow_ma", 50),
            cfg.get("rsi_period", 14),
        )
        resampled = sig.reindex(primary_index, method="ffill")
        signals_sum += resampled * w
        total_weight += w
        print(f"    {tf}: w={w:.2f} (loaded {len(df)} bars)")

    if total_weight > 0 and total_weight < 1.0:
        signals_sum /= total_weight

    return signals_sum, primary_df


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    pair = "EUR_USD"
    print("=" * 60)
    print("  MTF Portfolio Simulation (Risk-Managed)")
    print("=" * 60)
    print(f"  Initial Equity: ${INIT_CASH:,.0f}")
    print(f"  Risk Per Trade: {RISK_PER_TRADE:.1%}")
    print(f"  Using params from: {CONFIG_PATH}")

    # 1. Load Config & Confluence
    cfg = load_mtf_config()
    threshold = cfg.get("confirmation_threshold", 0.10)
    confluence, primary_df = compute_confluence(pair, cfg)

    close = primary_df["close"]
    high = primary_df["high"]
    low = primary_df["low"]

    # 2. Compute ATR for sizing
    atr = compute_atr(primary_df, ATR_PERIOD)

    # 3. Generate Signals
    # Long: >= threshold
    # Short: <= -threshold
    entries_long = confluence >= threshold
    entries_short = confluence <= -threshold

    # Exits: Neutrality or Reversal will be handled by Portfolio logic
    # But explicitly, we exit if confluence crosses zero against us
    # VBT handles reversals if we provide both entries and short_entries

    # 4. Position Sizing Logic (Fixed Risk Amount based on Initial Equity)
    # Stop Distance = 2.0 * ATR
    # Risk Amount = INIT_CASH * RISK_PER_TRADE ($1,000)
    # Units = Risk Amount / Stop Distance

    risk_amt = INIT_CASH * RISK_PER_TRADE
    stop_dist = STOP_ATR_MULT * atr

    # Calculate Stop Loss % (needed for sl_stop argument)
    stop_loss_pct = stop_dist / close
    stop_loss_pct = stop_loss_pct.replace(0, np.nan).fillna(0.0)

    # Raw unit size
    raw_units = risk_amt / stop_dist

    # Cap leverage (Max Units = (Equity * Leverage) / Price)
    # Approximation using Init Cash since vectorized
    max_units = (INIT_CASH * MAX_LEVERAGE) / close

    target_units = np.minimum(raw_units, max_units)
    target_units = target_units.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 5. Build Portfolio

    spread_series = build_spread_series(primary_df, pair)
    avg_spread = float(spread_series.mean())
    print(f"  Avg Spread Cost: {avg_spread:.5f}")

    # Run 1: With Trailing Stop
    print("\nRunning Scenario 1: Trailing Stop (2.0 ATR)...")
    pf_stop = vbt.Portfolio.from_signals(
        close=close,
        high=high,
        low=low,
        entries=entries_long,
        short_entries=entries_short,
        exits=(confluence < 0),
        short_exits=(confluence > 0),
        size=target_units,
        size_type="amount",
        init_cash=INIT_CASH,
        fees=avg_spread,
        freq="4h",
        sl_stop=stop_loss_pct,
        sl_trail=True,
    )

    # Run 2: Signal Only (No Stop)
    print("Running Scenario 2: Signal Only (No Stop)...")
    pf_signal = vbt.Portfolio.from_signals(
        close=close,
        high=high,
        low=low,
        entries=entries_long,
        short_entries=entries_short,
        exits=(confluence < 0),
        short_exits=(confluence > 0),
        size=target_units,
        size_type="amount",
        init_cash=INIT_CASH,
        fees=avg_spread,
        freq="4h",
        # No sl_stop
    )

    # 6. Analyze & Compare
    print("\n" + "=" * 80)
    print(f"  COMPARISON: Trailing Stop ({STOP_ATR_MULT}x ATR) vs Signal Only")
    print("=" * 80)

    headers = ["Metric", "With Stop", "Signal Only", "Diff"]
    row_fmt = "{:<15} {:<15} {:<15} {:<15}"
    print(row_fmt.format(*headers))
    print("-" * 60)

    def get_metrics(pf):
        return {
            "Return": pf.total_return(),
            "Sharpe": pf.sharpe_ratio(),
            "Max DD": pf.max_drawdown(),
            "Win Rate": pf.trades.win_rate(),
            "Trades": pf.trades.count(),
            "Final Eq": pf.value().iloc[-1],
        }

    m1 = get_metrics(pf_stop)
    m2 = get_metrics(pf_signal)

    # helper for formatting
    def fmt(val, is_pct=True):
        return f"{val:.2%}" if is_pct else f"{val:.2f}"

    print(
        row_fmt.format(
            "Total Return",
            fmt(m1["Return"]),
            fmt(m2["Return"]),
            f"{m2['Return'] - m1['Return']:+.2%}",
        )
    )
    print(
        row_fmt.format(
            "Sharpe",
            fmt(m1["Sharpe"], False),
            fmt(m2["Sharpe"], False),
            f"{m2['Sharpe'] - m1['Sharpe']:+.2f}",
        )
    )
    print(
        row_fmt.format(
            "Max Drawdown",
            fmt(m1["Max DD"]),
            fmt(m2["Max DD"]),
            f"{m2['Max DD'] - m1['Max DD']:+.2%}",
        )
    )
    print(
        row_fmt.format(
            "Win Rate",
            fmt(m1["Win Rate"]),
            fmt(m2["Win Rate"]),
            f"{m2['Win Rate'] - m1['Win Rate']:+.2%}",
        )
    )
    print(
        row_fmt.format(
            "Trades", str(m1["Trades"]), str(m2["Trades"]), f"{m2['Trades'] - m1['Trades']}"
        )
    )
    print(
        row_fmt.format(
            "Final Equity",
            f"${m1['Final Eq']:,.0f}",
            f"${m2['Final Eq']:,.0f}",
            f"${m2['Final Eq'] - m1['Final Eq']:,.0f}",
        )
    )

    # 7. Save Reports
    pf_stop.trades.records_readable.to_csv(REPORTS_DIR / "trades_with_stop.csv", index=False)
    pf_signal.trades.records_readable.to_csv(REPORTS_DIR / "trades_signal_only.csv", index=False)

    # Plot both equity curves
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=pf_stop.value().index, y=pf_stop.value(), name="With Stop"))
    fig.add_trace(go.Scatter(x=pf_signal.value().index, y=pf_signal.value(), name="Signal Only"))
    fig.update_layout(title="Equity Curve Comparison", yaxis_title="Equity ($)")

    html_path = REPORTS_DIR / "mtf_comparison.html"
    fig.write_html(str(html_path))
    print(f"\nSaved comparison report to {html_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
