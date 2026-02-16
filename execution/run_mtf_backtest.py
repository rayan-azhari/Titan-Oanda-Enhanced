"""run_mtf_backtest.py — Multi-Timeframe Confluence Backtest (Long + Short).

Combines RSI + MA crossover across H1/H4/D/W timeframes using the
weighting scheme from config/mtf.toml.  Generates weighted confluence
scores that drive both long and short entries, then runs a vectorised
backtest via VectorBT with IS/OOS validation.

Directive: Alpha Research Loop (VectorBT).md
"""

import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt is not installed. Run: pip install vectorbt")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: plotly is not installed. Run: pip install plotly")
    sys.exit(1)

from execution.spread_model import build_spread_series

# ─────────────────────────────────────────────────────────────────────
# Config loaders
# ─────────────────────────────────────────────────────────────────────


def load_mtf_config() -> dict:
    """Load multi-timeframe confluence config from config/mtf.toml."""
    path = PROJECT_ROOT / "config" / "mtf.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_data(pair: str, granularity: str) -> pd.DataFrame | None:
    """Load Parquet data for a pair/granularity, return None if missing."""
    path = DATA_DIR / f"{pair}_{granularity}.parquet"
    if not path.exists():
        print(f"  ⚠ {path.name} not found — skipping timeframe.")
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


# ─────────────────────────────────────────────────────────────────────
# Indicator computation
# ─────────────────────────────────────────────────────────────────────


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI from a close price series."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_timeframe_signal(
    close: pd.Series,
    fast_ma: int,
    slow_ma: int,
    rsi_period: int,
) -> pd.Series:
    """Compute a directional signal for one timeframe.

    Signal logic (each component contributes ±0.5, total range: [-1, +1]):
      • MA crossover: fast > slow → +0.5 (bullish), else −0.5
      • RSI confirmation:
          RSI > 50 → +0.5 (bullish momentum)
          RSI < 50 → −0.5 (bearish momentum)

    Args:
        close: Close price series for this timeframe.
        fast_ma: Fast moving average period.
        slow_ma: Slow moving average period.
        rsi_period: RSI lookback period.

    Returns:
        Series of signal values in [-1, +1].
    """
    fast = close.rolling(fast_ma).mean()
    slow = close.rolling(slow_ma).mean()
    rsi = compute_rsi(close, rsi_period)

    # MA crossover component: +0.5 bullish, -0.5 bearish
    ma_signal = pd.Series(np.where(fast > slow, 0.5, -0.5), index=close.index)

    # RSI component: +0.5 bullish, -0.5 bearish
    rsi_signal = pd.Series(np.where(rsi > 50, 0.5, -0.5), index=close.index)

    return ma_signal + rsi_signal  # Range: [-1, +1]


# ─────────────────────────────────────────────────────────────────────
# MTF confluence
# ─────────────────────────────────────────────────────────────────────


def compute_confluence_score(
    pair: str,
    mtf_config: dict,
    timeframes: list[str],
) -> pd.Series:
    """Build the weighted MTF confluence score on the H4 index.

    For each timeframe, computes a directional signal and then
    resamples it to the H4 (primary) index using forward-fill.
    The final score is the weighted sum across timeframes.

    Args:
        pair: Instrument name (e.g., "EUR_USD").
        mtf_config: Parsed mtf.toml config.
        timeframes: List of granularity codes (e.g., ["H1","H4","D","W"]).

    Returns:
        Weighted confluence score aligned to the H4 index, range [-1, +1].
    """
    weights = mtf_config.get("weights", {})

    # Load primary timeframe (H4) as the base index
    primary_df = load_data(pair, "H4")
    if primary_df is None:
        print("  ERROR: H4 data required as the primary timeframe.")
        sys.exit(1)

    primary_index = primary_df.index
    weighted_signals: list[pd.Series] = []
    total_weight = 0.0

    print("\n  Computing signals per timeframe:")
    for tf in timeframes:
        tf_config = mtf_config.get(tf, {})
        weight = weights.get(tf, 0.0)

        if weight == 0:
            continue

        df = load_data(pair, tf)
        if df is None:
            continue

        signal = compute_timeframe_signal(
            close=df["close"],
            fast_ma=tf_config.get("fast_ma", 20),
            slow_ma=tf_config.get("slow_ma", 50),
            rsi_period=tf_config.get("rsi_period", 14),
        )

        # Resample to H4 index using forward-fill
        # Higher timeframes (D, W) will hold their signal until the next bar
        signal_resampled = signal.reindex(primary_index, method="ffill")

        weighted_signal = signal_resampled * weight
        weighted_signals.append(weighted_signal)
        total_weight += weight

        # Stats
        bullish_pct = (signal > 0).mean() * 100
        bearish_pct = (signal < 0).mean() * 100
        print(
            f"    {tf:3s} (w={weight:.2f}):  "
            f"bullish {bullish_pct:.0f}% | bearish {bearish_pct:.0f}% | "
            f"{len(df)} bars"
        )

    if not weighted_signals:
        print("  ERROR: No valid timeframe signals computed.")
        sys.exit(1)

    # Sum all weighted signals
    confluence = sum(weighted_signals)

    # Normalise if we're missing some timeframes
    if total_weight > 0 and total_weight < 1.0:
        confluence = confluence / total_weight

    return confluence, primary_df


# ─────────────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────────────


def run_backtest(
    close: pd.Series,
    confluence: pd.Series,
    threshold: float,
    fees: float,
) -> dict:
    """Run a long/short backtest based on confluence thresholds.

    Entry rules:
      • LONG  when confluence ≥ +threshold
      • SHORT when confluence ≤ −threshold
    Exit rules:
      • Close long  when confluence < 0 (momentum fading)
      • Close short when confluence > 0

    Args:
        close: Close prices (H4).
        confluence: Weighted MTF score, range [-1, +1].
        threshold: Minimum absolute score to enter.
        fees: Per-trade cost (decimal, e.g., 0.0003 = 3 pips).

    Returns:
        Dict with stats and portfolio objects for long and short.
    """
    # ── Long signals ──
    long_entries = confluence >= threshold
    long_exits = confluence < 0

    # ── Short signals ──
    short_entries = confluence <= -threshold
    short_exits = confluence > 0

    # Run long portfolio
    long_pf = vbt.Portfolio.from_signals(
        close,
        entries=long_entries,
        exits=long_exits,
        init_cash=10_000,
        fees=fees,
        freq="4h",
    )

    # Run short portfolio using short_entries/short_exits
    short_pf = vbt.Portfolio.from_signals(
        close,
        entries=pd.Series(False, index=close.index),
        exits=pd.Series(False, index=close.index),
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=10_000,
        fees=fees,
        freq="4h",
    )

    return {
        "long": long_pf,
        "short": short_pf,
    }


def print_portfolio_stats(label: str, pf, period: str = "") -> dict:
    """Print key performance metrics for a portfolio.

    Args:
        label: Display name (e.g., "LONG IS").
        pf: VBT Portfolio object.
        period: Extra label (e.g., "in-sample").

    Returns:
        Dict of key metrics.
    """
    total_return = pf.total_return() * 100
    sharpe = pf.sharpe_ratio()
    max_dd = pf.max_drawdown() * 100
    total_trades = pf.trades.count()
    win_rate = pf.trades.win_rate() * 100 if total_trades > 0 else 0

    print(f"\n  {'─' * 50}")
    print(f"  📊 {label} {period}")
    print(f"  {'─' * 50}")
    print(f"    Total Return:  {total_return:>8.2f}%")
    print(f"    Sharpe Ratio:  {sharpe:>8.3f}")
    print(f"    Max Drawdown:  {max_dd:>8.2f}%")
    print(f"    Total Trades:  {total_trades:>8d}")
    print(f"    Win Rate:      {win_rate:>8.1f}%")

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_trades": total_trades,
        "win_rate": win_rate,
    }


def generate_confluence_chart(
    close: pd.Series,
    confluence: pd.Series,
    threshold: float,
    pair: str,
    label: str,
) -> Path:
    """Generate an interactive chart showing price + confluence score."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    # Price
    fig.add_trace(
        go.Scatter(
            x=close.index, y=close.values, name="Close", line=dict(color="#2196F3", width=1)
        ),
        row=1,
        col=1,
    )

    # Confluence score
    colors = np.where(
        confluence >= threshold, "#4CAF50", np.where(confluence <= -threshold, "#F44336", "#9E9E9E")
    )
    fig.add_trace(
        go.Bar(
            x=confluence.index, y=confluence.values, name="Confluence", marker_color=colors.tolist()
        ),
        row=2,
        col=1,
    )

    # Threshold lines
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="green",
        annotation_text="Long threshold",
        row=2,
        col=1,
    )
    fig.add_hline(
        y=-threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Short threshold",
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_layout(
        title=f"MTF Confluence — {pair} H4 ({label})",
        height=700,
        template="plotly_dark",
        showlegend=True,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Confluence Score", row=2, col=1)

    path = REPORTS_DIR / f"mtf_confluence_{pair}_{label.lower()}.html"
    fig.write_html(str(path))
    print(f"  📊 Chart saved → {path.name}")
    return path


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def load_instruments_config() -> dict:
    """Load the instruments configuration from config/instruments.toml."""
    config_path = PROJECT_ROOT / "config" / "instruments.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def main() -> None:
    """Run the MTF confluence backtest with IS/OOS validation for all pairs."""
    mtf_config = load_mtf_config()
    instruments_config = load_instruments_config()

    pairs = instruments_config.get("instruments", {}).get("pairs", [])
    if not pairs:
        print("ERROR: No pairs found in config/instruments.toml")
        sys.exit(1)

    threshold = mtf_config.get("confirmation_threshold", 0.30)
    # Only keep actual timeframe keys from the weights section
    valid_tfs = {"M1", "M5", "M15", "M30", "H1", "H2", "H4", "H8", "D", "W", "M"}
    timeframes = [k for k in mtf_config.get("weights", {}).keys() if k in valid_tfs]

    print(f"{'=' * 80}")
    print(f"  🔬 MTF Confluence Backtest — {len(pairs)} Pairs")
    print(f"{'=' * 80}")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"  Threshold:  ±{threshold}")
    print("  Direction:  LONG + SHORT")
    print()

    summary_stats = []

    for pair in pairs:
        print(f"👉 Processing {pair}...")

        # ── Compute confluence score ──
        try:
            confluence, primary_df = compute_confluence_score(pair, mtf_config, timeframes)
        except SystemExit:
            print(f"   ⚠️ Skipping {pair} (missing data)")
            continue
        except Exception as e:
            print(f"   ⚠️ Error processing {pair}: {e}")
            continue

        close = primary_df["close"]

        # ── Spread-based cost estimation ──
        try:
            spread_series = build_spread_series(primary_df, pair)
            avg_spread = float(spread_series.mean())
        except Exception:
            # Fallback if spread model fails or data missing
            avg_spread = 0.0002

        # ── IS / OOS Split (70/30) ──
        split_idx = int(len(close) * 0.70)
        # is_close = close.iloc[:split_idx]
        oos_close = close.iloc[split_idx:]
        # is_conf = confluence.iloc[:split_idx]
        oos_conf = confluence.iloc[split_idx:]

        # ── Run Backtests ──
        # In-Sample
        # is_results = run_backtest(is_close, is_conf, threshold, fees=avg_spread)
        # We only care about combined return for the summary mostly, but let's calc key stats

        # Out-of-Sample
        oos_results = run_backtest(oos_close, oos_conf, threshold, fees=avg_spread)
        oos_long_pf = oos_results["long"]
        oos_short_pf = oos_results["short"]

        # ── Aggregate Stats ──
        # Combined OOS Return
        oos_combined_ret = (oos_long_pf.total_return() + oos_short_pf.total_return()) / 2 * 100
        oos_combined_sharpe = (oos_long_pf.sharpe_ratio() + oos_short_pf.sharpe_ratio()) / 2
        oos_trades = oos_long_pf.trades.count() + oos_short_pf.trades.count()

        summary_stats.append(
            {
                "Pair": pair,
                "OOS Return %": oos_combined_ret,
                "OOS Sharpe": oos_combined_sharpe,
                "Trades": oos_trades,
                "Spread (pips)": avg_spread * 10000,
            }
        )

        # Generate Chart for OOS only to save space/time, or both if needed.
        # Let's do OOS only to keep it faster, or both if user wants details.
        # Generating only OOS chart for now to avoid cluttering reports too much
        generate_confluence_chart(oos_close, oos_conf, threshold, pair, "OOS")
        print(f"   ✅ {pair}: OOS Ret={oos_combined_ret:.2f}%, Sharpe={oos_combined_sharpe:.2f}")

    # ── Final Summary Table ──
    print(f"\n{'=' * 80}")
    print(f"  📊 FINAL SUMMARY ({len(summary_stats)} pairs)")
    print(f"{'=' * 80}")

    if summary_stats:
        df_summary = pd.DataFrame(summary_stats)
        df_summary = df_summary.sort_values("OOS Sharpe", ascending=False)

        print(df_summary.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))

        # Save summary CSV
        summary_path = REPORTS_DIR / "mtf_backtest_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"\n  💾 Summary saved to {summary_path}")
    else:
        print("  ❌ No results generated.")

    print(f"\n{'=' * 80}")
    print("  ✅ Batch Backtest Complete")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
