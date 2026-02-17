"""Multi-Timeframe Confluence Strategy Research (M5) - Stage 2 Optimization.

This script implements Stage 2 of the VectorBT optimization for the MTF Strategy-5m.

Stage 2 Goal:
    - Fix `ma_type` and `threshold` to best values from Stage 1 (WMA, 0.55).
    - Sweep Timeframe Weights (M5, H1, H4, D) to find the optimal balance.
    - Use IS/OOS validation with Parity check.

Parameters:
    - MA Type: WMA
    - Threshold: 0.55
    - Weight Combos: Grid search of heuristic weight distributions.
"""

import sys
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import pandas as pd
import vectorbt as vbt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(pair: str, granularity: str) -> pd.DataFrame:
    """Load Parquet data for a specific pair and granularity."""
    path = DATA_DIR / f"{pair}_{granularity}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    return df


def compute_ma(close: pd.Series, period: int, ma_type: Literal["SMA", "EMA", "WMA"]) -> pd.Series:
    """Compute Moving Average based on type."""
    if ma_type == "EMA":
        return close.ewm(span=period, adjust=False).mean()
    elif ma_type == "WMA":
        weights = np.arange(1, period + 1, dtype=float)
        return close.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    else:  # SMA
        return close.rolling(period).mean()


def compute_rsa_ma_signal(
    close: pd.Series,
    fast_ma: int,
    slow_ma: int,
    rsi_period: int,
    ma_type: Literal["SMA", "EMA", "WMA"] = "SMA",
) -> pd.Series:
    """Compute directional signal based on MA Crossover and RSI."""
    fast = compute_ma(close, fast_ma, ma_type)
    slow = compute_ma(close, slow_ma, ma_type)
    rsi = vbt.RSI.run(close, rsi_period).rsi

    # +0.5 if Fast > Slow, -0.5 if Fast < Slow
    ma_sig = np.where(fast > slow, 0.5, -0.5)

    # +0.5 if RSI > 50, -0.5 if RSI < 50
    rsi_sig = np.where(rsi > 50, 0.5, -0.5)

    return pd.Series(ma_sig + rsi_sig, index=close.index)


def get_portfolio_stats(pf: vbt.Portfolio) -> Dict[str, float]:
    """Extract key metrics from a portfolio."""
    return {
        "total_return": pf.total_return() * 100,
        "sharpe": pf.sharpe_ratio(),
        "max_drawdown": pf.max_drawdown() * 100,
        "trades": pf.trades.count(),
        "win_rate": pf.trades.win_rate() * 100 if pf.trades.count() > 0 else 0.0,
    }


def run_stage2_optimization(pair: str = "EUR_USD") -> None:
    """Run Stage 2 Optimization (Weights)."""
    print(f"Running MTF Strategy-5m Stage 2 Optimization for {pair}...")

    # 1. Load Data
    try:
        m5 = load_data(pair, "M5")
        h1 = load_data(pair, "H1")
        h4 = load_data(pair, "H4")
        d1 = load_data(pair, "D")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Align ranges
    start_dt = max(m5.index.min(), h1.index.min(), h4.index.min(), d1.index.min())
    end_dt = min(m5.index.max(), h1.index.max(), h4.index.max(), d1.index.max())
    m5 = m5.loc[start_dt:end_dt]

    print(f"Data Range: {start_dt} to {end_dt}")
    print(f"M5 Bars: {len(m5)}")

    # Split IS/OOS
    split_idx = int(len(m5) * 0.70)

    is_close = m5["close"].iloc[:split_idx]
    oos_close = m5["close"].iloc[split_idx:]

    # Fixed Parameters (Load from State or Default)
    MA_TYPE = "WMA"
    THRESHOLD = 0.55

    try:
        import research.mtf.state_manager as state_manager

        s1 = state_manager.get_stage1()
        if s1:
            MA_TYPE, THRESHOLD = s1
            print(f"Loaded Stage 1 State: MA={MA_TYPE}, Threshold={THRESHOLD}")
        else:
            print(f"No Stage 1 State found. Using defaults: MA={MA_TYPE}, Threshold={THRESHOLD}")
    except ImportError:
        print("Could not import state_manager. Using defaults.")

    print(f"Active Params: MA={MA_TYPE}, Threshold={THRESHOLD}")

    # Default Indicator Params (Fixed for Stage 2)
    FAST_MA, SLOW_MA, RSI_PERIOD = 20, 50, 14

    # Pre-calculate signals (expensive part done once)
    print("Calculating signals...")
    sig_m5 = compute_rsa_ma_signal(m5["close"], FAST_MA, SLOW_MA, RSI_PERIOD, MA_TYPE)
    sig_h1 = compute_rsa_ma_signal(h1["close"], FAST_MA, SLOW_MA, RSI_PERIOD, MA_TYPE)
    sig_h4 = compute_rsa_ma_signal(h4["close"], FAST_MA, SLOW_MA, RSI_PERIOD, MA_TYPE)
    sig_d1 = compute_rsa_ma_signal(d1["close"], FAST_MA, SLOW_MA, RSI_PERIOD, MA_TYPE)

    # Resample to M5
    sig_h1_re = sig_h1.reindex(m5.index, method="ffill").fillna(0)
    sig_h4_re = sig_h4.reindex(m5.index, method="ffill").fillna(0)
    sig_d1_re = sig_d1.reindex(m5.index, method="ffill").fillna(0)

    # Weight Configurations to Sweep
    # Logic: Test various dominance profiles (Balanced, D1-heavy, H1-heavy, etc.)
    # Weights don't technically need to sum to 1, but we usually normalize.
    # We will test arbitrary combinations and let the code normalize if needed,
    # but for consistent threshold application, sum=1 is best.

    weight_combos = [
        # (M5, H1, H4, D1, "Label")
        (0.1, 0.3, 0.3, 0.3, "Balanced Higher"),
        (0.25, 0.25, 0.25, 0.25, "Equal Weight"),
        (0.05, 0.15, 0.30, 0.50, "Trend Dominant (D1)"),
        (0.05, 0.10, 0.25, 0.60, "Heavy Trend (D1)"),
        (0.10, 0.40, 0.30, 0.20, "Tactical (H1)"),
        (0.40, 0.30, 0.20, 0.10, "Fast Scalp (M5)"),
        (0.0, 0.2, 0.3, 0.5, "No M5 Noise"),
        (0.1, 0.2, 0.2, 0.5, "Balanced Trend"),
    ]

    results = []

    for w_m5, w_h1, w_h4, w_d1, label in weight_combos:
        # Normalize sum to 1.0 ensures Threshold=0.55 means same "conviction"
        total_w = w_m5 + w_h1 + w_h4 + w_d1
        nw_m5 = w_m5 / total_w
        nw_h1 = w_h1 / total_w
        nw_h4 = w_h4 / total_w
        nw_d1 = w_d1 / total_w

        confluence_score = (
            sig_m5 * nw_m5 + sig_h1_re * nw_h1 + sig_h4_re * nw_h4 + sig_d1_re * nw_d1
        )

        is_score = confluence_score.iloc[:split_idx]
        oos_score = confluence_score.iloc[split_idx:]

        # --- IS Backtest ---
        pf_is = vbt.Portfolio.from_signals(
            is_close,
            entries=is_score >= THRESHOLD,
            exits=is_score < 0,
            short_entries=is_score <= -THRESHOLD,
            short_exits=is_score > 0,
            freq="5min",
            init_cash=10000,
            fees=0.0003,
            slippage=0.0,
        )
        stats_is = get_portfolio_stats(pf_is)

        # --- OOS Backtest ---
        pf_oos = vbt.Portfolio.from_signals(
            oos_close,
            entries=oos_score >= THRESHOLD,
            exits=oos_score < 0,
            short_entries=oos_score <= -THRESHOLD,
            short_exits=oos_score > 0,
            freq="5min",
            init_cash=10000,
            fees=0.0003,
            slippage=0.0,
        )
        stats_oos = get_portfolio_stats(pf_oos)

        parity = stats_oos["sharpe"] / stats_is["sharpe"] if stats_is["sharpe"] != 0 else 0

        results.append(
            {
                "Label": label,
                "Weights": f"[{w_m5}, {w_h1}, {w_h4}, {w_d1}]",
                "is_sharpe": stats_is["sharpe"],
                "oos_sharpe": stats_oos["sharpe"],
                "parity": parity,
                "is_ret": stats_is["total_return"],
                "oos_ret": stats_oos["total_return"],
            }
        )

    df = pd.DataFrame(results).sort_values("is_sharpe", ascending=False)

    print("\n--- Stage 2 Results (Weights) ---")
    print(df.to_string(index=False))

    csv_path = REPORTS_DIR / "mtf_stage2_weights.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    # Best Candidate
    candidates = df[df["parity"] > 0.5]
    if not candidates.empty:
        best = candidates.iloc[0]
        print(f"\n🏆 BEST WEIGHT CONFIG: {best['Label']}")
        print(f"   Weights: {best['Weights']}")
        print(f"   IS Sharpe: {best['is_sharpe']:.2f}")
        print(f"   OOS Sharpe: {best['oos_sharpe']:.2f}")

        # Save to State
        try:
            # Parse weights string back to dict or list?
            # The weights string is like "[0.1, 0.3, ...]"
            # We need to map it back to TFs: M5, H1, H4, D1
            # For now, let's just save the raw list or a dict if we can infer order
            # The combo list was: w_m5, w_h1, w_h4, w_d1
            import ast

            w_list = ast.literal_eval(best["Weights"])
            best_weights = {
                "M5": w_list[0],
                "H1": w_list[1],
                "H4": w_list[2],
                "D": w_list[3],
            }
            state_manager.save_stage2(best_weights)
        except Exception as e:
            print(f"Error saving state: {e}")

    else:
        print("\n⚠️ No candidates passed parity check > 0.5")


if __name__ == "__main__":
    run_stage2_optimization()
