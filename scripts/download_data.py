"""download_data.py — Pull historical OHLC data from OANDA.

Downloads candlestick data for the instruments and granularities
specified in config/instruments.toml. Stores output as Parquet
files in data/.
"""

import argparse
import sys
import tomllib
from datetime import timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import os

import oandapyV20
import pandas as pd

from titan.data.oanda import candles_to_dataframe, fetch_candles

# Config
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_instruments_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "instruments.toml"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found.")
        sys.exit(1)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def main() -> None:
    if not ACCESS_TOKEN:
        print("ERROR: OANDA_ACCESS_TOKEN not found in .env")
        sys.exit(1)

    config = load_instruments_config()
    client = oandapyV20.API(access_token=ACCESS_TOKEN, environment=ENVIRONMENT)

    pairs = config.get("instruments", {}).get("pairs", [])
    granularities = config.get("instruments", {}).get("granularities", ["M5"])

    parser = argparse.ArgumentParser(description="Download OANDA data")
    parser.add_argument("-i", "--instrument", help="Filter by instrument (e.g. EUR_USD)")
    parser.add_argument("-g", "--granularity", help="Filter by granularity (e.g. M5, H1)")
    args = parser.parse_args()

    if args.instrument:
        if args.instrument not in pairs:
            print(f"❌ Error: Instrument {args.instrument} not in instruments.toml")
            sys.exit(1)
        pairs = [args.instrument]

    if args.granularity:
        if args.granularity in granularities:
            granularities = [args.granularity]
        else:
            print(f"❌ Error: Granularity {args.granularity} not in instruments.toml")
            sys.exit(1)

    print(f"📥 Downloading data for {len(pairs)} pairs × {len(granularities)} granularities\n")

    for pair in pairs:
        for gran in granularities:
            output_path = DATA_DIR / f"{pair}_{gran}.parquet"

            # Resume logic or start from 2005
            from_time = "2005-01-01T00:00:00Z"
            if output_path.exists():
                try:
                    existing = pd.read_parquet(output_path)
                    # Enforce timestamp type
                    if not pd.api.types.is_datetime64_any_dtype(existing["timestamp"]):
                        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)

                    # Sanitize: Remove future data if any
                    now_utc = pd.Timestamp.now(timezone.utc)
                    existing = existing[existing["timestamp"] <= now_utc]

                    if not existing.empty:
                        last_ts = existing["timestamp"].max()
                        from_time = last_ts.isoformat()
                        print(f"  ↻ Resuming {pair} {gran} from {from_time}")
                except Exception as e:
                    print(f"    ⚠ Error reading existing file: {e}. Starting fresh.")
                    from_time = "2005-01-01T00:00:00Z"
            else:
                print(f"  ↓ Downloading {pair} {gran} from {from_time}...")

            # Pagination Loop
            total_downloaded = 0
            while True:
                try:
                    candles = fetch_candles(client, pair, gran, from_time=from_time, count=5000)
                    df = candles_to_dataframe(candles)

                    if df.empty:
                        print(f"    ✓ No more data for {pair} {gran}.")
                        break

                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

                    # Filter out records <= from_time to avoid duplicates/infinite loop
                    # OANDA 'from' is exclusive or inclusive? It seems inclusive sometimes.
                    if from_time:
                         df = df[df["timestamp"] > pd.Timestamp(from_time)]

                    if df.empty:
                        print("    ✓ Up to date.")
                        break

                    # Append to existing file on disk (read-modify-write pattern for simplicity)
                    if output_path.exists():
                        existing = pd.read_parquet(output_path)
                        if not pd.api.types.is_datetime64_any_dtype(existing["timestamp"]):
                            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)

                        df = (
                            pd.concat([existing, df])
                            .drop_duplicates(subset="timestamp")
                            .sort_values("timestamp")
                        )

                    # Type conversion
                    cols = ["open", "high", "low", "close", "volume"]
                    for col in cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")

                    now_utc = pd.Timestamp.now(timezone.utc)
                    df = df[df["timestamp"] <= now_utc]

                    if df.empty:
                         print("    ⚠ All fetched data was future-dated/filtered. Stop.")
                         break

                    df.to_parquet(output_path, index=False)


                    last_ts = df["timestamp"].max()
                    from_time = last_ts.isoformat()
                    total_downloaded += len(candles)

                    print(f"    → Saved {len(df)} rows total. Last: {last_ts}. (Batch: {len(candles)})")

                    # If we got fewer than requested, we are likely at the end
                    if len(candles) < 5000:
                        print("    ✓ Reached end of stream.")
                        break

                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    break

    print("\n✅ Data download complete.\n")


if __name__ == "__main__":
    main()
