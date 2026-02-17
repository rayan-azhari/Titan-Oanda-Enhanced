"""download_data.py — Pull historical OHLC data from OANDA.

Downloads candlestick data for the instruments and granularities
specified in config/instruments.toml. Stores output as Parquet
files in data/.
"""

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

    if not pairs:
        print("ERROR: No pairs defined in config/instruments.toml")
        sys.exit(1)

    print(f"📥 Downloading data for {len(pairs)} pairs × {len(granularities)} granularities\n")

    for pair in pairs:
        for gran in granularities:
            output_path = DATA_DIR / f"{pair}_{gran}.parquet"

            # Resume logic
            from_time = None
            if output_path.exists():
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
            else:
                print(f"  ↓ Downloading {pair} {gran}...")

            try:
                candles = fetch_candles(client, pair, gran, from_time=from_time, count=5000)
                df = candles_to_dataframe(candles)

                if df.empty:
                    print(f"    No new data for {pair} {gran}.")
                    continue

                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

                if output_path.exists():
                    existing = pd.read_parquet(output_path)
                    # Enforce timestamp type on existing data too, just in case
                    if not pd.api.types.is_datetime64_any_dtype(existing["timestamp"]):
                        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)

                    df = (
                        pd.concat([existing, df])
                        .drop_duplicates(subset="timestamp")
                        .sort_values("timestamp")
                    )

                # Ensure numeric types for OHLCV to avoid pyarrow issues
                cols = ["open", "high", "low", "close", "volume"]
                for col in cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                # Filter out future timestamps (e.g. from backtest leakage or OANDA glitches)
                now_utc = pd.Timestamp.now(timezone.utc)
                df = df[df["timestamp"] <= now_utc]

                if df.empty:
                    print("    ⚠ All data was future-dated. Skipping save.")
                    continue

                df.to_parquet(output_path, index=False)
                prior_count = 0
                if output_path.exists() and "existing" in locals():
                    prior_count = len(existing)
                new_count = len(df) - prior_count
                print(
                    f"    ✓ {len(df)} rows total | "
                    f"{df['timestamp'].min()} → {df['timestamp'].max()}"
                )

            except Exception as e:
                print(f"    ❌ Error: {e}")

    print("\n✅ Data download complete.\n")


if __name__ == "__main__":
    main()
