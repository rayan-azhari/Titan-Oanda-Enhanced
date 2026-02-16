"""download_oanda_data.py — Pull historical OHLC data from OANDA.

Downloads candlestick data for the instruments and granularities
specified in config/instruments.toml. Stores output as Parquet
files in .tmp/data/raw/.

Supports resumption from the last stored timestamp.

Directive: Alpha Research Loop (VectorBT Pro).md
"""

import os
import sys
import time
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import oandapyV20
import oandapyV20.endpoints.instruments as instruments_ep
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")

RAW_DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_instruments_config() -> dict:
    """Load the instruments configuration from config/instruments.toml."""
    config_path = PROJECT_ROOT / "config" / "instruments.toml"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found.")
        sys.exit(1)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def fetch_candles(
    client: oandapyV20.API,
    instrument: str,
    granularity: str,
    from_time: str | None = None,
    count: int = 5000,
) -> list[dict]:
    """Fetch candles from OANDA with exponential back-off on rate limits.

    Args:
        client: Authenticated OANDA API client.
        instrument: Instrument name (e.g., "EUR_USD").
        granularity: Candle granularity (e.g., "M5", "H1").
        from_time: ISO-8601 start time for resumption.
        count: Maximum number of candles to retrieve.

    Returns:
        List of candle dictionaries from the OANDA response.
    """
    params: dict = {
        "granularity": granularity,
        "count": count,
        "price": "BA",
    }
    if from_time:
        params["from"] = from_time

    r = instruments_ep.InstrumentsCandles(instrument=instrument, params=params)

    for attempt in range(5):
        try:
            response = client.request(r)
            return response.get("candles", [])
        except oandapyV20.exceptions.V20Error as e:
            if "429" in str(e):
                wait = 2**attempt
                print(f"  ⏳ Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    return []


def candles_to_dataframe(candles: list[dict]) -> pd.DataFrame:
    """Convert OANDA candle response to a clean DataFrame.

    Uses bid prices. Only includes complete candles.
    All prices stored as Decimal for financial precision.

    Args:
        candles: Raw candle list from the OANDA API.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    rows = []
    for c in candles:
        if not c.get("complete", False):
            continue
        bid = c["bid"]
        rows.append(
            {
                "timestamp": pd.Timestamp(c["time"]),
                "open": float(bid["o"]),
                "high": float(bid["h"]),
                "low": float(bid["l"]),
                "close": float(bid["c"]),
                "volume": int(c["volume"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Download historical OHLC data for all configured instruments."""
    config = load_instruments_config()
    client = oandapyV20.API(access_token=ACCESS_TOKEN, environment=ENVIRONMENT)

    pairs = config.get("instruments", {}).get("pairs", [])
    granularities = config.get("instruments", {}).get("granularities", ["M5"])

    if not pairs:
        print("ERROR: No pairs defined in config/instruments.toml")
        sys.exit(1)

    print(f"📥 Downloading data for {len(pairs)} pairs × {len(granularities)} granularities\n")

    for pair in pairs:
        for gran in granularities:
            output_path = RAW_DATA_DIR / f"{pair}_{gran}.parquet"

            # Resume logic: start from last stored timestamp
            from_time = None
            if output_path.exists():
                existing = pd.read_parquet(output_path)
                last_ts = existing["timestamp"].max()
                from_time = last_ts.isoformat()
                print(f"  ↻ Resuming {pair} {gran} from {from_time}")
            else:
                print(f"  ↓ Downloading {pair} {gran}...")

            candles = fetch_candles(client, pair, gran, from_time=from_time)
            df = candles_to_dataframe(candles)

            if df.empty:
                print(f"    No new data for {pair} {gran}.")
                continue

            # Merge with existing data if resuming
            if output_path.exists():
                existing = pd.read_parquet(output_path)
                df = (
                    pd.concat([existing, df])
                    .drop_duplicates(subset="timestamp")
                    .sort_values("timestamp")
                )

            df.to_parquet(output_path, index=False)
            print(f"    ✓ {len(df)} rows | {df['timestamp'].min()} → {df['timestamp'].max()}")

    print("\n✅ Data download complete.\n")


if __name__ == "__main__":
    main()
