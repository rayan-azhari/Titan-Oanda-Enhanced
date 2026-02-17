"""Multi-Timeframe Confluence Strategy Research (M5) - Stage 3 Optimization.

This script implements Stage 3 of the VectorBT optimization for the MTF Strategy-5m.

Stage 3 Goal:
    - Sweep `fast_ma`, `slow_ma`, and `rsi_period` for each timeframe.
    - Fixed: `ma_type=WMA`, `threshold=0.55`, `weights=[0.1, 0.3, 0.3, 0.3]`.
    - Strategy: Greedy optimization in order of importance (H4 -> H1 -> M5 -> D).
    - Validation: IS/OOS Parity check.

Execution:
    1. Optimize H4 parameters (while others are default).
    2. Fix H4 to best found parameters.
    3. Optimize H1 parameters (while H4 is fixed best, others default).
    4. ... and so on for M5 and D.
"""

import itertools
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
        return close.rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    else:  # SMA
        return close.rolling(period).mean()


def compute_rsa_ma_signal(
    close: pd.Series,
    fast_ma: int,
    slow_ma: int,
    rsi_period: int,
    ma_type: Literal["SMA", "EMA", "WMA"] = "WMA",
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
        # "win_rate": pf.trades.win_rate() * 100 if pf.trades.count() > 0 else 0.0,
    }


def run_stage3_optimization(pair: str = "EUR_USD") -> None:
    """Run Stage 3 Optimization (Greedy Parameter Sweep)."""
    print(f"Running MTF Strategy-5m Stage 3 Optimization for {pair}...")

    # --- 1. Load Data ---
    m5 = load_data(pair, "M5")
    h1 = load_data(pair, "H1")
    h4 = load_data(pair, "H4")
    d1 = load_data(pair, "D")

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

    # --- 2. Fixed Constants (Stage 1 & 2) ---
    MA_TYPE = "WMA"
    THRESHOLD = 0.55
    # Weights: M5, H1, H4, D1 (Balanced Higher)
    WEIGHTS = {"M5": 0.1, "H1": 0.3, "H4": 0.3, "D": 0.3}

    # Defaults
    current_params = {
        "M5": {"fast": 20, "slow": 50, "rsi": 14},
        "H1": {"fast": 20, "slow": 50, "rsi": 14},
        "H4": {"fast": 20, "slow": 50, "rsi": 14},
        "D": {"fast": 20, "slow": 50, "rsi": 14},
    }

    # --- 3. Optimization Schedule (Greedy) ---
    order = ["H4", "H1", "M5", "D"]

    # Sweep Ranges
    fast_range = [10, 20, 30]
    slow_range = [40, 50, 60]
    rsi_range = [14, 21]

    final_results = []

    for target_tf in order:
        print(f"\n--- Optimizing {target_tf} ---")
        best_sharpe = -999.0
        best_p = None

        combos = list(itertools.product(fast_range, slow_range, rsi_range))

        for fast, slow, rsi in combos:
            # Update specific TF params tentatively
            test_params = current_params.copy()
            test_params[target_tf] = {"fast": fast, "slow": slow, "rsi": rsi}

            # Compute signals for ALL TFs using current strongest known params
            signals = {}
            for tf, data in [("M5", m5), ("H1", h1), ("H4", h4), ("D", d1)]:
                p = test_params[tf]
                sig = compute_rsa_ma_signal(
                    data["close"], p["fast"], p["slow"], p["rsi"], MA_TYPE
                )
                if tf != "M5":
                    sig = sig.reindex(m5.index, method="ffill").fillna(0)
                signals[tf] = sig

            # Confluence
            confluence_score = (
                signals["M5"] * WEIGHTS["M5"]
                + signals["H1"] * WEIGHTS["H1"]
                + signals["H4"] * WEIGHTS["H4"]
                + signals["D"] * WEIGHTS["D"]
            )

            # IS Backtest
            is_score = confluence_score.iloc[:split_idx]
            pf_is = vbt.Portfolio.from_signals(
                is_close,
                entries=is_score >= THRESHOLD,
                exits=is_score < 0,
                short_entries=is_score <= -THRESHOLD,
                short_exits=is_score > 0,
                freq="5min",
                init_cash=10000,
                fees=0.0003,
            )
            stats_is = get_portfolio_stats(pf_is)

            # OOS Backtest
            oos_score = confluence_score.iloc[split_idx:]
            pf_oos = vbt.Portfolio.from_signals(
                oos_close,
                entries=oos_score >= THRESHOLD,
                exits=oos_score < 0,
                short_entries=oos_score <= -THRESHOLD,
                short_exits=oos_score > 0,
                freq="5min",
                init_cash=10000,
                fees=0.0003,
            )
            stats_oos = get_portfolio_stats(pf_oos)

            parity = stats_oos["sharpe"] / stats_is["sharpe"] if stats_is["sharpe"] != 0 else 0

            # Valid Candidate? (Parity > 0.5)
            # Relaxed parity check for greedy step to avoid getting stuck,
            # but prefer > 0.5.
            if stats_is["sharpe"] > best_sharpe and parity > 0.4:
                best_sharpe = stats_is["sharpe"]
                best_p = {"fast": fast, "slow": slow, "rsi": rsi}
                print(
                    f"  New Best {target_tf}: {best_p} | "
                    f"IS Sharpe: {best_sharpe:.2f} | Parity: {parity:.2f}"
                )

        if best_p:
            print(f"✅ Locked {target_tf}: {best_p}")
            current_params[target_tf] = best_p
            final_results.append({"TF": target_tf, "Params": str(best_p), "IS_Sharpe": best_sharpe})
        else:
            print(f"⚠️ No improvement for {target_tf}, keeping defaults.")

    print("\n" + "=" * 40)
    print("      FINAL OPTIMIZED PARAMETERS      ")
    print("=" * 40)
    for tf, p in current_params.items():
        print(f"  {tf}: {p}")

    # Save results
    df = pd.DataFrame(final_results)
    csv_path = REPORTS_DIR / "mtf_stage3_params.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    run_stage3_optimization()
