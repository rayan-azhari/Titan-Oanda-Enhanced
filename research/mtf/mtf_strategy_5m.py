"""Multi-Timeframe Confluence Strategy Research (M5) - Optimization.

This script implements a VectorBT optimization for a Multi-Timeframe (MTF) strategy
focusing on 5-minute (M5), 1-hour (H1), 4-hour (H4), and Daily (D) timeframes.

Optimization:
    - Sweeps `ma_type`: SMA, EMA, WMA.
    - Sweeps `confirmation_threshold`: 0.10 to 0.85.
    - Uses 70% In-Sample (IS) / 30% Out-of-Sample (OOS) split.
    - Selects best parameters based on IS performance and OOS Stability (Parity).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
    ma_type: Literal["SMA", "EMA", "WMA"] = "SMA"
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


def run_mtf_optimization(pair: str = "EUR_USD") -> None:
    """Run the MTF Strategy-5m optimization."""
    print(f"Running MTF Strategy-5m Optimization for {pair}...")
    
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
    # Reindex others to M5 for signal calculation alignment (optional, but good for cleanliness)
    
    print(f"Data Range: {start_dt} to {end_dt}")
    print(f"M5 Bars: {len(m5)}")

    # Split IS/OOS (70/30)
    split_idx = int(len(m5) * 0.70)
    split_dt = m5.index[split_idx]
    
    print(f"IS Range:  {m5.index[0]} -> {split_dt}")
    print(f"OOS Range: {split_dt} -> {m5.index[-1]}")

    # Optimization Parameters
    MA_TYPES = ["SMA", "EMA", "WMA"]
    THRESHOLDS = np.arange(0.10, 0.86, 0.05) # 0.10 to 0.85
    
    results = []
    
    # Defaults
    fast_ma, slow_ma, rsi_period = 20, 50, 14
    w_m5, w_h1, w_h4, w_d1 = 0.1, 0.3, 0.3, 0.3

    for ma_type in MA_TYPES:
        print(f"  > Processing {ma_type}...")
        
        # Calculate Signals for this MA Type
        sig_m5 = compute_rsa_ma_signal(m5["close"], fast_ma, slow_ma, rsi_period, ma_type)
        sig_h1 = compute_rsa_ma_signal(h1["close"], fast_ma, slow_ma, rsi_period, ma_type)
        sig_h4 = compute_rsa_ma_signal(h4["close"], fast_ma, slow_ma, rsi_period, ma_type)
        sig_d1 = compute_rsa_ma_signal(d1["close"], fast_ma, slow_ma, rsi_period, ma_type)

        # Broadcast/Resample to M5 index
        sig_h1_re = sig_h1.reindex(m5.index, method="ffill").fillna(0)
        sig_h4_re = sig_h4.reindex(m5.index, method="ffill").fillna(0)
        sig_d1_re = sig_d1.reindex(m5.index, method="ffill").fillna(0)
        
        # Confluence Score
        confluence_score = (
            sig_m5 * w_m5 +
            sig_h1_re * w_h1 +
            sig_h4_re * w_h4 +
            sig_d1_re * w_d1
        )
        
        # Split Score & Close for VBT
        is_score = confluence_score.iloc[:split_idx]
        oos_score = confluence_score.iloc[split_idx:]
        
        is_close = m5["close"].iloc[:split_idx]
        oos_close = m5["close"].iloc[split_idx:]
        
        # Iterate Thresholds
        for th in THRESHOLDS:
            th = round(th, 2)
            
            # --- In-Sample ---
            is_entries = is_score >= th
            is_short_entries = is_score <= -th
            is_exits = is_score < 0
            is_short_exits = is_score > 0
            
            pf_is = vbt.Portfolio.from_signals(
                is_close,
                entries=is_entries,
                exits=is_exits,
                short_entries=is_short_entries,
                short_exits=is_short_exits,
                freq="5min",
                init_cash=10000,
                fees=0.0003, # 2 pips spread + 1 pip slippage
                slippage=0.0000
            )
            stats_is = get_portfolio_stats(pf_is)
            
            # --- Out-of-Sample ---
            oos_entries = oos_score >= th
            oos_short_entries = oos_score <= -th
            oos_exits = oos_score < 0
            oos_short_exits = oos_score > 0
            
            pf_oos = vbt.Portfolio.from_signals(
                oos_close,
                entries=oos_entries,
                exits=oos_exits,
                short_entries=oos_short_entries,
                short_exits=oos_short_exits,
                freq="5min",
                init_cash=10000,
                fees=0.0003,
                slippage=0.0000
            )
            stats_oos = get_portfolio_stats(pf_oos)
            
            # --- Metrics ---
            parity = stats_oos["sharpe"] / stats_is["sharpe"] if stats_is["sharpe"] != 0 else 0
            
            results.append({
                "ma_type": ma_type,
                "threshold": th,
                "is_sharpe": stats_is["sharpe"],
                "is_ret": stats_is["total_return"],
                "is_dd": stats_is["max_drawdown"],
                "is_trades": stats_is["trades"],
                "oos_sharpe": stats_oos["sharpe"],
                "oos_ret": stats_oos["total_return"],
                "oos_dd": stats_oos["max_drawdown"],
                "oos_trades": stats_oos["trades"],
                "parity": parity
            })

    # Convert to DataFrame
    df_res = pd.DataFrame(results)
    
    # Sort by IS Sharpe descending
    df_res = df_res.sort_values("is_sharpe", ascending=False)
    
    print("\nTop 5 Candidates (by IS Sharpe):")
    print(df_res.head(5).to_string(index=False))
    
    # Save CSV
    csv_path = REPORTS_DIR / "mtf_strategy_5m_optimization.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\nOptimization results saved to {csv_path}")

    # Best Candidate Logic: 
    # Must have Parity > 0.5 (OOS Sharpe at least 50% of IS Sharpe)
    # Then highest IS Sharpe.
    candidates = df_res[df_res["parity"] > 0.5]
    
    if not candidates.empty:
        best = candidates.iloc[0]
        print(f"\n🏆 BEST STABLE CANDIDATE:")
        print(f"   MA Type: {best['ma_type']}")
        print(f"   Threshold: {best['threshold']}")
        print(f"   IS Sharpe: {best['is_sharpe']:.2f}")
        print(f"   OOS Sharpe: {best['oos_sharpe']:.2f}")
        print(f"   Parity: {best['parity']:.2f}")
    else:
        print("\n⚠️ No stable candidates found (Parity > 0.5).")

    # Heatmap of IS Sharpe
    try:
        heatmap_data = df_res.pivot(index="ma_type", columns="threshold", values="is_sharpe")
        fig = px.imshow(
            heatmap_data,
            title="MTF Strategy-5m: IS Sharpe Ratio (MA Type vs Threshold)",
            labels=dict(x="Threshold", y="MA Type", color="IS Sharpe"),
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        html_path = REPORTS_DIR / "mtf_strategy_5m_heatmap.html"
        fig.write_html(str(html_path))
        print(f"Heatmap saved to {html_path}")
    except Exception as e:
        print(f"Could not generate heatmap: {e}")


if __name__ == "__main__":
    run_mtf_optimization()
