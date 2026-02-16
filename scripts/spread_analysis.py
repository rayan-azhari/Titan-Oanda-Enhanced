"""spread_analysis.py — Generate spread and report analysis.

Moved from titan/models/spread.py to separate script logic from library code.
"""

import sys
import tomllib
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from titan.models.spread import build_spread_series, build_total_cost_series

RAW_DATA_DIR = PROJECT_ROOT / "data"

def generate_spread_report(pair: str, granularity: str) -> None:
    """Generate a spread analysis report for a given pair."""
    path = RAW_DATA_DIR / f"{pair}_{granularity}.parquet"
    if not path.exists():
        print(f"  Skipping {pair}_{granularity}: no data file at {path}")
        return

    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    # Pass config or let it use defaults/internal loading for now (step 2)
    # Ideally we pass config dict here.
    spread = build_spread_series(df, pair)
    total_cost = build_total_cost_series(df, pair)

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
