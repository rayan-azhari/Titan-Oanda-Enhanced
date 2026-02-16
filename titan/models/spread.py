"""spread_model.py — Time-varying spread and slippage estimation.

Models realistic trading costs by estimating spread variation across
trading sessions (Tokyo, London, New York) and applying slippage
based on position size.

Research Gap Fix #2: The original research only used a flat 0.0002 fee
with no consideration for variable spread or slippage.
"""

import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DATA_DIR = PROJECT_ROOT / ".tmp" / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Session definitions (UTC hours)
# ---------------------------------------------------------------------------
SESSIONS = {
    "tokyo": (0, 9),  # 00:00–09:00 UTC
    "london": (8, 16),  # 08:00–16:00 UTC
    "new_york": (13, 21),  # 13:00–21:00 UTC
    "off_hours": None,  # Everything else
}


def load_spread_config() -> dict:
    """Load spread configuration from config/spread.toml.

    Returns:
        Spread configuration dictionary.
    """
    config_path = PROJECT_ROOT / "config" / "spread.toml"
    if not config_path.exists():
        print(f"WARNING: {config_path} not found. Using defaults.")
        return {
            "EUR_USD": {
                "london": 0.00012,
                "new_york": 0.00014,
                "tokyo": 0.00025,
                "off_hours": 0.00035,
            },
            "GBP_USD": {
                "london": 0.00015,
                "new_york": 0.00018,
                "tokyo": 0.00030,
                "off_hours": 0.00045,
            },
            "AUD_USD": {
                "london": 0.00016,
                "new_york": 0.00018,
                "tokyo": 0.00020,
                "off_hours": 0.00040,
            },
        }
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def classify_session(hour: int) -> str:
    """Classify an hour (UTC) into a trading session.

    During overlap periods (London/NY), returns the lower-spread session.

    Args:
        hour: Hour of day in UTC (0–23).

    Returns:
        Session name: "tokyo", "london", "new_york", or "off_hours".
    """
    if 8 <= hour < 16:
        return "london"
    elif 13 <= hour < 21:
        return "new_york"
    elif 0 <= hour < 9:
        return "tokyo"
    else:
        return "off_hours"


def estimate_slippage(position_units: int, avg_volume: float) -> float:
    """Estimate slippage based on position size relative to average volume.

    Uses a square-root market impact model: slippage ∝ √(size/volume).

    Args:
        position_units: Number of units in the order.
        avg_volume: Average volume for the time period.

    Returns:
        Estimated slippage as a decimal (e.g., 0.00005 = 0.5 pips).
    """
    if avg_volume <= 0:
        return 0.0001  # Conservative default
    impact = 0.0001 * np.sqrt(position_units / avg_volume)
    return min(impact, 0.001)  # Cap at 10 pips


def build_spread_series(df: pd.DataFrame, pair: str) -> pd.Series:
    """Build a time-varying spread series for the given dataset.

    If the data contains both bid and ask prices, computes actual spread.
    Otherwise, estimates from the session-based lookup table.

    Args:
        df: OHLCV DataFrame with a timestamp column or DatetimeIndex.
        pair: Instrument name (e.g., "EUR_USD").

    Returns:
        Series of spread values aligned with the DataFrame index.
    """
    config = load_spread_config()
    pair_config = config.get(pair, config.get("EUR_USD", {}))

    # Check if we have actual bid-ask data
    if "bid_close" in df.columns and "ask_close" in df.columns:
        actual_spread = df["ask_close"].astype(float) - df["bid_close"].astype(float)
        print(f"  Using actual bid-ask spread (mean: {actual_spread.mean():.6f})")
        return actual_spread

    # Otherwise, use session-based estimates
    if isinstance(df.index, pd.DatetimeIndex):
        hours = df.index.hour
    elif "timestamp" in df.columns:
        hours = pd.to_datetime(df["timestamp"]).dt.hour
    else:
        print("  WARNING: No timestamp found. Using flat spread.")
        return pd.Series(0.0003, index=df.index)

    sessions = hours.map(classify_session)
    spreads = sessions.map(lambda s: pair_config.get(s, 0.0003))
    return pd.Series(spreads.values, index=df.index, name="spread")


def build_total_cost_series(df: pd.DataFrame, pair: str, position_size: int = 5000) -> pd.Series:
    """Build a total cost series (spread + slippage) per bar.

    Args:
        df: OHLCV DataFrame.
        pair: Instrument name.
        position_size: Intended position size in units.

    Returns:
        Series of total cost per bar (spread + slippage).
    """
    spread = build_spread_series(df, pair)

    # Estimate slippage from volume
    if "volume" in df.columns:
        avg_vol = df["volume"].astype(float).rolling(20).mean()
        slippage = avg_vol.apply(lambda v: estimate_slippage(position_size, v) if v > 0 else 0.0001)
    else:
        slippage = pd.Series(0.00005, index=df.index)

    total = spread + slippage
    return total


def generate_spread_report(pair: str, granularity: str) -> None:
    """Generate a spread analysis report for a given pair.

    Args:
        pair: Instrument name.
        granularity: Candle granularity.
    """
    path = RAW_DATA_DIR / f"{pair}_{granularity}.parquet"
    if not path.exists():
        print(f"  Skipping {pair}_{granularity}: no data file.")
        return

    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    spread = build_spread_series(df, pair)
    total_cost = build_total_cost_series(df, pair)

    _close = df["close"].astype(float)  # noqa: F841
    spread_pips = spread * 10_000  # Convert to pips
    cost_pips = total_cost * 10_000

    print(f"\n  {'=' * 50}")
    print(f"  📊 Spread Report: {pair} ({granularity})")
    print(f"  {'=' * 50}")
    print(f"  Mean spread:      {spread_pips.mean():.2f} pips")
    print(f"  Median spread:    {spread_pips.median():.2f} pips")
    print(f"  Max spread:       {spread_pips.max():.2f} pips")
    print(f"  Mean total cost:  {cost_pips.mean():.2f} pips")

    # Session breakdown
    if isinstance(df.index, pd.DatetimeIndex):
        hours = df.index.hour
        for session_name, (start, end) in [
            ("Tokyo", (0, 9)),
            ("London", (8, 16)),
            ("New York", (13, 21)),
        ]:
            mask = (hours >= start) & (hours < end)
            if mask.any():
                session_spread = spread_pips[mask].mean()
                print(f"  {session_name:12s} avg:  {session_spread:.2f} pips")


def main() -> None:
    """Generate spread reports for all configured instruments."""
    config_path = PROJECT_ROOT / "config" / "instruments.toml"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found.")
        sys.exit(1)

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    pairs = config.get("instruments", {}).get("pairs", [])
    granularities = config.get("instruments", {}).get("granularities", ["H4"])

    print("📈 Spread & Slippage Analysis\n")

    for pair in pairs:
        for gran in granularities:
            generate_spread_report(pair, gran)

    print("\n✅ Spread analysis complete.\n")


if __name__ == "__main__":
    main()
