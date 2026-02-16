"""run_mtf_optimisation.py — Stage 1: Threshold & MA-Type Sweep.

Sweeps confirmation_threshold and ma_type (SMA / EMA / WMA) across all
timeframes in the MTF Confluence strategy.  Uses IS/OOS validation to
avoid overfitting.

Directive: Alpha Research Loop (VectorBT).md  —  Step 6 (MTF Confluence)

Usage:
    uv run python research/mtf/run_optimisation.py
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
CONFIG_PATH = PROJECT_ROOT / "config" / "mtf.toml"

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed. Run: pip install vectorbt")
    sys.exit(1)

from titan.models.spread import build_spread_series  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Parameter grid — Stage 1
# ─────────────────────────────────────────────────────────────────────

THRESHOLDS = [round(x, 2) for x in np.arange(0.10, 0.85, 0.05)]
MA_TYPES = ["SMA", "EMA", "WMA"]

# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────


def load_data(pair: str, granularity: str) -> pd.DataFrame | None:
    """Load Parquet data for a pair/granularity, return None if missing."""
    path = DATA_DIR / f"{pair}_{granularity}.parquet"
    if not path.exists():
        print(f"  WARNING: {path.name} not found — skipping timeframe.")
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def load_mtf_config() -> dict:
    """Load multi-timeframe confluence config from config/mtf.toml."""
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


# ─────────────────────────────────────────────────────────────────────
# Moving-average helpers
# ─────────────────────────────────────────────────────────────────────


def compute_ma(close: pd.Series, period: int, ma_type: str) -> pd.Series:
    """Compute a moving average using the specified type.

    Args:
        close: Close price series.
        period: Lookback period.
        ma_type: One of 'SMA', 'EMA', 'WMA'.

    Returns:
        Moving average series.
    """
    if ma_type == "EMA":
        return close.ewm(span=period, adjust=False).mean()
    elif ma_type == "WMA":
        weights = np.arange(1, period + 1, dtype=float)
        return close.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    else:  # SMA (default)
        return close.rolling(period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI from a close price series."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ─────────────────────────────────────────────────────────────────────
# Signal computation
# ─────────────────────────────────────────────────────────────────────


def compute_timeframe_signal(
    close: pd.Series,
    fast_ma: int,
    slow_ma: int,
    rsi_period: int,
    ma_type: str = "SMA",
) -> pd.Series:
    """Compute a directional signal for one timeframe.

    Signal components (each +/-0.5, total range [-1, +1]):
      - MA crossover: fast > slow -> +0.5 else -0.5
      - RSI: > 50 -> +0.5 else -0.5

    Args:
        close: Close price series.
        fast_ma: Fast MA period.
        slow_ma: Slow MA period.
        rsi_period: RSI lookback.
        ma_type: 'SMA', 'EMA', or 'WMA'.

    Returns:
        Series of signal values in [-1, +1].
    """
    fast = compute_ma(close, fast_ma, ma_type)
    slow = compute_ma(close, slow_ma, ma_type)
    rsi = compute_rsi(close, rsi_period)

    ma_signal = pd.Series(np.where(fast > slow, 0.5, -0.5), index=close.index)
    rsi_signal = pd.Series(np.where(rsi > 50, 0.5, -0.5), index=close.index)

    return ma_signal + rsi_signal


def compute_confluence(
    pair: str,
    mtf_config: dict,
    timeframes: list[str],
    ma_type: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build weighted MTF confluence score on H4 index.

    Args:
        pair: Instrument name.
        mtf_config: Parsed mtf.toml.
        timeframes: List of granularity codes.
        ma_type: MA type to use across all timeframes.

    Returns:
        Tuple of (confluence Series, primary DataFrame).
    """
    weights = mtf_config.get("weights", {})

    primary_df = load_data(pair, "H4")
    if primary_df is None:
        print("  ERROR: H4 data required.")
        sys.exit(1)

    primary_index = primary_df.index
    weighted_signals: list[pd.Series] = []
    total_weight = 0.0

    for tf in timeframes:
        tf_config = mtf_config.get(tf, {})
        weight = weights.get(tf, 0.0)
        if weight == 0:
            continue

        df = load_data(tf_pair := pair, tf)  # noqa: F841
        if df is None:
            continue

        signal = compute_timeframe_signal(
            close=df["close"],
            fast_ma=tf_config.get("fast_ma", 20),
            slow_ma=tf_config.get("slow_ma", 50),
            rsi_period=tf_config.get("rsi_period", 14),
            ma_type=ma_type,
        )

        signal_resampled = signal.reindex(primary_index, method="ffill")
        weighted_signals.append(signal_resampled * weight)
        total_weight += weight

    if not weighted_signals:
        print("  ERROR: No valid signals.")
        sys.exit(1)

    confluence = sum(weighted_signals)
    if 0 < total_weight < 1.0:
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
    """Run long/short backtest with given threshold.

    Returns:
        Dict with 'long' and 'short' VBT Portfolio objects.
    """
    long_pf = vbt.Portfolio.from_signals(
        close,
        entries=confluence >= threshold,
        exits=confluence < 0,
        init_cash=10_000,
        fees=fees,
        freq="4h",
    )

    short_pf = vbt.Portfolio.from_signals(
        close,
        entries=pd.Series(False, index=close.index),
        exits=pd.Series(False, index=close.index),
        short_entries=confluence <= -threshold,
        short_exits=confluence > 0,
        init_cash=10_000,
        fees=fees,
        freq="4h",
    )

    return {"long": long_pf, "short": short_pf}


def extract_stats(pf) -> dict:
    """Pull key metrics from a VBT Portfolio."""
    n = pf.trades.count()
    return {
        "total_return": pf.total_return(),
        "sharpe": pf.sharpe_ratio(),
        "max_drawdown": pf.max_drawdown(),
        "n_trades": n,
        "win_rate": float(pf.trades.win_rate()) if n > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: sweep threshold x ma_type, report best combos."""
    pair = "EUR_USD"
    mtf_config = load_mtf_config()
    valid_tfs = {"M1", "M5", "M15", "M30", "H1", "H2", "H4", "H8", "D", "W", "M"}
    timeframes = [k for k in mtf_config.get("weights", {}) if k in valid_tfs]

    print("=" * 60)
    print("  Stage 1: Threshold x MA-Type Sweep")
    print("=" * 60)
    print(f"  Pair:       {pair}")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"  Thresholds: {len(THRESHOLDS)} values ({THRESHOLDS[0]} .. {THRESHOLDS[-1]})")
    print(f"  MA types:   {', '.join(MA_TYPES)}")
    total = len(MA_TYPES) * len(THRESHOLDS)
    print(f"  Total combos: {total}")

    results: list[dict] = []

    for ma_idx, ma_type in enumerate(MA_TYPES):
        print(f"\n--- {ma_type} ({ma_idx + 1}/{len(MA_TYPES)}) ---")

        # Compute confluence once per MA type (threshold changes don't
        # affect the indicator, only the entry/exit signals)
        confluence, primary_df = compute_confluence(pair, mtf_config, timeframes, ma_type)
        close = primary_df["close"]

        # Spread cost
        spread_series = build_spread_series(primary_df, pair)
        avg_spread = float(spread_series.mean())

        # IS / OOS split (70/30)
        split = int(len(close) * 0.70)
        is_close, oos_close = close.iloc[:split], close.iloc[split:]
        is_conf, oos_conf = confluence.iloc[:split], confluence.iloc[split:]

        for threshold in THRESHOLDS:
            # IS backtest
            is_res = run_backtest(is_close, is_conf, threshold, fees=avg_spread)
            is_long = extract_stats(is_res["long"])
            is_short = extract_stats(is_res["short"])

            # OOS backtest
            oos_res = run_backtest(oos_close, oos_conf, threshold, fees=avg_spread)
            oos_long = extract_stats(oos_res["long"])
            oos_short = extract_stats(oos_res["short"])

            # Parity ratios
            long_parity = oos_long["sharpe"] / is_long["sharpe"] if is_long["sharpe"] != 0 else 0
            short_parity = (
                oos_short["sharpe"] / is_short["sharpe"] if is_short["sharpe"] != 0 else 0
            )

            # Combined Sharpe (average of long + short IS+OOS)
            combined_sharpe = (
                is_long["sharpe"] + is_short["sharpe"] + oos_long["sharpe"] + oos_short["sharpe"]
            ) / 4

            results.append(
                {
                    "ma_type": ma_type,
                    "threshold": threshold,
                    # IS
                    "is_long_ret": is_long["total_return"],
                    "is_long_sharpe": is_long["sharpe"],
                    "is_long_dd": is_long["max_drawdown"],
                    "is_long_trades": is_long["n_trades"],
                    "is_long_wr": is_long["win_rate"],
                    "is_short_ret": is_short["total_return"],
                    "is_short_sharpe": is_short["sharpe"],
                    "is_short_dd": is_short["max_drawdown"],
                    "is_short_trades": is_short["n_trades"],
                    "is_short_wr": is_short["win_rate"],
                    # OOS
                    "oos_long_ret": oos_long["total_return"],
                    "oos_long_sharpe": oos_long["sharpe"],
                    "oos_long_dd": oos_long["max_drawdown"],
                    "oos_long_trades": oos_long["n_trades"],
                    "oos_long_wr": oos_long["win_rate"],
                    "oos_short_ret": oos_short["total_return"],
                    "oos_short_sharpe": oos_short["sharpe"],
                    "oos_short_dd": oos_short["max_drawdown"],
                    "oos_short_trades": oos_short["n_trades"],
                    "oos_short_wr": oos_short["win_rate"],
                    # Parity
                    "long_parity": long_parity,
                    "short_parity": short_parity,
                    "combined_sharpe": combined_sharpe,
                }
            )

        done = (ma_idx + 1) * len(THRESHOLDS)
        print(f"  [{done}/{total}] combos done")

    # ── Results ──
    df = pd.DataFrame(results)
    csv_path = REPORTS_DIR / "mtf_stage1_scoreboard.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull scoreboard: {csv_path}")

    # ── Best overall (by combined Sharpe, filter: both parities > 0) ──
    valid = df[(df["long_parity"] > 0) & (df["short_parity"] > 0)]
    if valid.empty:
        print("\nWARNING: No combos pass parity. Showing best by combined_sharpe.")
        valid = df

    best = valid.loc[valid["combined_sharpe"].idxmax()]

    print("\n" + "=" * 60)
    print("  BEST PARAMETERS (Stage 1)")
    print("=" * 60)
    print(f"  MA Type:    {best['ma_type']}")
    print(f"  Threshold:  {best['threshold']:.2f}")
    print(f"  Combined Sharpe: {best['combined_sharpe']:.4f}")
    print()
    print(
        f"  IS  LONG:   ret={best['is_long_ret']:.2%}  "
        f"sharpe={best['is_long_sharpe']:.3f}  "
        f"dd={best['is_long_dd']:.2%}  "
        f"trades={int(best['is_long_trades'])}  "
        f"wr={best['is_long_wr']:.1%}"
    )
    print(
        f"  IS  SHORT:  ret={best['is_short_ret']:.2%}  "
        f"sharpe={best['is_short_sharpe']:.3f}  "
        f"dd={best['is_short_dd']:.2%}  "
        f"trades={int(best['is_short_trades'])}  "
        f"wr={best['is_short_wr']:.1%}"
    )
    print(
        f"  OOS LONG:   ret={best['oos_long_ret']:.2%}  "
        f"sharpe={best['oos_long_sharpe']:.3f}  "
        f"dd={best['oos_long_dd']:.2%}  "
        f"trades={int(best['oos_long_trades'])}  "
        f"wr={best['oos_long_wr']:.1%}"
    )
    print(
        f"  OOS SHORT:  ret={best['oos_short_ret']:.2%}  "
        f"sharpe={best['oos_short_sharpe']:.3f}  "
        f"dd={best['oos_short_dd']:.2%}  "
        f"trades={int(best['oos_short_trades'])}  "
        f"wr={best['oos_short_wr']:.1%}"
    )
    print(f"  Long Parity:  {best['long_parity']:.2f}")
    print(f"  Short Parity: {best['short_parity']:.2f}")

    # ── Best per MA type ──
    print("\n--- Best per MA Type ---")
    for ma in MA_TYPES:
        sub = valid[valid["ma_type"] == ma]
        if sub.empty:
            print(f"  {ma}: no valid combos")
            continue
        b = sub.loc[sub["combined_sharpe"].idxmax()]
        print(
            f"  {ma:4s}  thresh={b['threshold']:.2f}  "
            f"comb_sharpe={b['combined_sharpe']:.4f}  "
            f"IS_long={b['is_long_ret']:.2%}  "
            f"OOS_long={b['oos_long_ret']:.2%}  "
            f"parity_L={b['long_parity']:.2f}  "
            f"parity_S={b['short_parity']:.2f}"
        )

    # ── Generate heatmap ──
    try:
        import plotly.express as px

        fig = px.imshow(
            df.pivot(index="ma_type", columns="threshold", values="combined_sharpe"),
            labels=dict(x="Threshold", y="MA Type", color="Combined Sharpe"),
            title="Stage 1: Combined Sharpe — MA Type vs Threshold",
            color_continuous_scale="RdYlGn",
            aspect="auto",
        )
        heatmap_path = REPORTS_DIR / "mtf_stage1_heatmap.html"
        fig.write_html(str(heatmap_path))
        print(f"\nHeatmap: {heatmap_path}")
    except ImportError:
        print("\nPlotly not installed — skipping heatmap.")

    print("\nDone.")


if __name__ == "__main__":
    main()
