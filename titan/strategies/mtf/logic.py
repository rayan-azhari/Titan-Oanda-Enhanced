"""mtf_confluence.py — Multi-Timeframe Confluence Signal Engine.

Aligns signals across H1, H4, and D timeframes to only trade when
all timeframes agree on direction. This filters out low-conviction
setups and dramatically reduces false signals.

Higher timeframes set the directional bias; lower timeframes time
the entry.

Directive: Multi-Timeframe Confluence.md

WARNING: This script is destructive. It will:
1. Cancel ALL pending orders for the account.
2. Market-close ALL open positions for the account.
Use only in emergency situations where the algorithm is out of control.
"""

import sys
import tomllib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# sys.path.insert(0, str(PROJECT_ROOT))

RAW_DATA_DIR = PROJECT_ROOT / ".tmp" / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / ".tmp" / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Indicator helpers (lookback-only, no future data)
# ---------------------------------------------------------------------------


def compute_trend(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Compute trend direction from dual-MA crossover.

    Args:
        close: Close price series.
        fast: Fast MA period.
        slow: Slow MA period.

    Returns:
        Series: +1 (bullish), -1 (bearish), 0 (neutral/cross zone).
    """
    ma_fast = close.rolling(window=fast).mean()
    ma_slow = close.rolling(window=slow).mean()

    # Require 0.1% separation between moving averages to confirm a trend.
    # This filter avoids false signals during choppy/sideways markets where MAs frequently cross.
    spread = (ma_fast - ma_slow) / ma_slow
    trend = pd.Series(0, index=close.index)
    trend[spread > 0.001] = 1
    trend[spread < -0.001] = -1
    return trend


def compute_momentum(close: pd.Series, rsi_period: int = 14) -> pd.Series:
    """Compute momentum bias from RSI.

    Args:
        close: Close price series.
        rsi_period: RSI lookback period.

    Returns:
        Series: +1 (bullish momentum), -1 (bearish), 0 (neutral 40-60 zone).
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi_val = 100 - (100 / (1 + rs))

    momentum = pd.Series(0, index=close.index)
    momentum[rsi_val > 60] = 1
    momentum[rsi_val < 40] = -1
    return momentum


def compute_structure(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """Compute market structure — higher highs/lows vs lower highs/lows.

    Args:
        high: High price series.
        low: Low price series.
        period: Lookback period for swing detection.

    Returns:
        Series: +1 (higher structure), -1 (lower structure), 0 (ranging).
    """
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()

    prev_highest = highest.shift(period)
    prev_lowest = lowest.shift(period)

    structure = pd.Series(0, index=high.index)
    structure[(highest > prev_highest) & (lowest > prev_lowest)] = 1  # Uptrend
    structure[(highest < prev_highest) & (lowest < prev_lowest)] = -1  # Downtrend
    return structure


# ---------------------------------------------------------------------------
# Timeframe signal builder
# ---------------------------------------------------------------------------


def build_timeframe_signal(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build directional signal components for a single timeframe.

    Args:
        df: OHLCV DataFrame for one timeframe.
        config: MTF config for this timeframe.

    Returns:
        DataFrame with trend, momentum, structure columns.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    fast_ma = config.get("fast_ma", 20)
    slow_ma = config.get("slow_ma", 50)
    rsi_period = config.get("rsi_period", 14)
    structure_period = config.get("structure_period", 20)

    signals = pd.DataFrame(index=df.index)
    signals["trend"] = compute_trend(close, fast_ma, slow_ma)
    signals["momentum"] = compute_momentum(close, rsi_period)
    signals["structure"] = compute_structure(high, low, structure_period)

    # Composite bias for this timeframe: average of components
    signals["bias"] = (signals["trend"] + signals["momentum"] + signals["structure"]) / 3.0
    return signals


def align_timeframes(
    h1_signals: pd.DataFrame,
    h4_signals: pd.DataFrame,
    d_signals: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """Align signals from multiple timeframes onto the H1 index.

    Higher timeframe signals are forward-filled onto the H1 timeline.
    # IMPORTANT: We use forward-fill to simulate real-time availability.
    # At any H1 timestamp, we only know the state of the H4/D candle that *closed* previously.
    # This prevents look-ahead bias.

    Args:
        h1_signals: H1 signal DataFrame.
        h4_signals: H4 signal DataFrame.
        d_signals: Daily signal DataFrame.
        weights: Timeframe weight dict, e.g. {"H1": 0.2, "H4": 0.4, "D": 0.4}.

    Returns:
        DataFrame with confluence score and individual timeframe biases.
    """
    # Reindex higher TFs onto H1 timeline with forward-fill
    h4_aligned = h4_signals["bias"].reindex(h1_signals.index, method="ffill")
    d_aligned = d_signals["bias"].reindex(h1_signals.index, method="ffill")

    confluence = pd.DataFrame(index=h1_signals.index)
    confluence["h1_bias"] = h1_signals["bias"]
    confluence["h4_bias"] = h4_aligned
    confluence["d_bias"] = d_aligned

    # Weighted confluence score
    w_h1 = weights.get("H1", 0.2)
    w_h4 = weights.get("H4", 0.4)
    w_d = weights.get("D", 0.4)

    confluence["confluence_score"] = (
        confluence["h1_bias"] * w_h1 + confluence["h4_bias"] * w_h4 + confluence["d_bias"] * w_d
    )

    # Agreement flag: all 3 same sign
    confluence["all_bullish"] = (
        (confluence["h1_bias"] > 0) & (confluence["h4_bias"] > 0) & (confluence["d_bias"] > 0)
    ).astype(int)

    confluence["all_bearish"] = (
        (confluence["h1_bias"] < 0) & (confluence["h4_bias"] < 0) & (confluence["d_bias"] < 0)
    ).astype(int)

    # Final signal: only trade with full confluence
    confluence["signal"] = 0
    confluence.loc[confluence["all_bullish"] == 1, "signal"] = 1
    confluence.loc[confluence["all_bearish"] == 1, "signal"] = -1

    return confluence


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_mtf_config() -> dict:
    """Load MTF configuration from config/mtf.toml."""
    config_path = PROJECT_ROOT / "config" / "mtf.toml"
    if not config_path.exists():
        print(f"WARNING: {config_path} not found. Using defaults.")
        return {
            "weights": {"H1": 0.2, "H4": 0.4, "D": 0.4},
            "confirmation_threshold": 0.3,
            "H1": {"fast_ma": 20, "slow_ma": 50, "rsi_period": 14, "structure_period": 20},
            "H4": {"fast_ma": 20, "slow_ma": 50, "rsi_period": 14, "structure_period": 20},
            "D": {"fast_ma": 10, "slow_ma": 30, "rsi_period": 14, "structure_period": 10},
        }
    with open(config_path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_pair_data(pair: str, granularity: str) -> pd.DataFrame | None:
    """Load raw data for a pair/granularity if available.

    Args:
        pair: Instrument name.
        granularity: Candle granularity.

    Returns:
        DataFrame or None if file doesn't exist.
    """
    path = RAW_DATA_DIR / f"{pair}_{granularity}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    return df


def main() -> None:
    """Generate multi-timeframe confluence features for all pairs."""
    config = load_mtf_config()
    weights = config.get("weights", {"H1": 0.2, "H4": 0.4, "D": 0.4})

    # Load instruments config
    inst_path = PROJECT_ROOT / "config" / "instruments.toml"
    if not inst_path.exists():
        print("ERROR: config/instruments.toml not found.")
        sys.exit(1)
    with open(inst_path, "rb") as f:
        inst_config = tomllib.load(f)

    pairs = inst_config.get("instruments", {}).get("pairs", [])
    print("🔀 Multi-Timeframe Confluence Engine\n")
    print(
        f"  Weights: H1={weights.get('H1', 0.2)}, "
        f"H4={weights.get('H4', 0.4)}, D={weights.get('D', 0.4)}\n"
    )

    for pair in pairs:
        print(f"  ━━━ {pair} ━━━")

        h1_df = load_pair_data(pair, "H1")
        h4_df = load_pair_data(pair, "H4")
        d_df = load_pair_data(pair, "D")

        if h1_df is None or h4_df is None or d_df is None:
            missing = []
            if h1_df is None:
                missing.append("H1")
            if h4_df is None:
                missing.append("H4")
            if d_df is None:
                missing.append("D")
            print(
                f"  ⚠️  Missing data: {', '.join(missing)}. "
                "Run download_oanda_data.py for all timeframes.\n"
            )
            continue

        # Build signals per timeframe
        h1_signals = build_timeframe_signal(h1_df, config.get("H1", {}))
        h4_signals = build_timeframe_signal(h4_df, config.get("H4", {}))
        d_signals = build_timeframe_signal(d_df, config.get("D", {}))

        # Align onto H1 timeline
        confluence = align_timeframes(h1_signals, h4_signals, d_signals, weights)

        # Report
        total_bars = len(confluence)
        bullish = (confluence["signal"] == 1).sum()
        bearish = (confluence["signal"] == -1).sum()
        neutral = total_bars - bullish - bearish
        print(f"    Total bars:  {total_bars}")
        print(f"    📈 Bullish:  {bullish} ({bullish / total_bars * 100:.1f}%)")
        print(f"    📉 Bearish:  {bearish} ({bearish / total_bars * 100:.1f}%)")
        print(f"    ⏸️  Neutral:  {neutral} ({neutral / total_bars * 100:.1f}%)")

        # Save confluence features
        out_path = FEATURES_DIR / f"{pair}_mtf_confluence.parquet"
        confluence.to_parquet(out_path)
        print(f"    ✓ Saved to {out_path.name}\n")

    print("✅ MTF confluence signals complete.\n")


if __name__ == "__main__":
    main()
