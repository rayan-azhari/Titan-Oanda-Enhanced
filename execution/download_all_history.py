"""download_all_history.py — Pull FULL historical OHLC data for EUR_USD H1.

Downloads all available H1 candlestick data for EUR_USD from OANDA,
starting from 2005-01-01 to present.
"""

import os
import sys
import time
from pathlib import Path

import oandapyV20
import oandapyV20.endpoints.instruments as instruments_ep
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")

INSTRUMENT = "EUR_USD"
GRANULARITY = "H1"
START_DATE = "2005-01-01T00:00:00Z"

# Ensure data directory exists
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = DATA_DIR / f"{INSTRUMENT}_{GRANULARITY}_all.parquet"


def fetch_candles(client, instrument, granularity, from_time, count=5000):
    """Fetch candles from OANDA with retry logic."""
    params = {
        "granularity": granularity,
        "count": count,
        "price": "B",  # Bid price
        "from": from_time,
        "includeFirst": False,
    }

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
                print(f"Error fetching data: {e}")
                raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
    return []


def candles_to_dataframe(candles):
    """Convert OANDA candle response to a DataFrame."""
    rows = []
    for c in candles:
        if not c.get("complete", False):
            continue

        bid = c.get("bid")
        if not bid:
            continue

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


def main():
    print(f"🚀 Starting full download for {INSTRUMENT} {GRANULARITY} from {START_DATE}")

    if not ACCESS_TOKEN:
        print("❌ OANDA_ACCESS_TOKEN not set in .env")
        return

    client = oandapyV20.API(access_token=ACCESS_TOKEN, environment=ENVIRONMENT)

    all_dfs = []
    next_time = START_DATE

    while True:
        print(f"  Fetching from {next_time}...")
        try:
            candles = fetch_candles(
                client,
                INSTRUMENT,
                GRANULARITY,
                from_time=next_time,
                count=5000,
            )
        except Exception as e:
            print(f"  ❌ Critical error: {e}")
            break

        if not candles:
            print("  No more candles received.")
            break

        df = candles_to_dataframe(candles)
        if df.empty:
            print("  Returned candles were empty or incomplete.")
            break

        all_dfs.append(df)
        count = len(df)

        last_time_ts = df.iloc[-1]["timestamp"]
        print(f"    Got {count} rows. Last: {last_time_ts}")

        # OANDA returns RFC3339 strings, but we parsed to Timestamp.
        # We need to convert back to ISO format for the next request.
        next_time = last_time_ts.isoformat()

        # Check if we are close to now
        if last_time_ts.tz_convert("UTC") >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1):
            print("  Reached present time.")
            break

        time.sleep(0.1)

    if not all_dfs:
        print("No data downloaded.")
        return

    print("Merging data...")
    final_df = pd.concat(all_dfs)
    final_df = final_df.drop_duplicates(subset="timestamp").sort_values("timestamp")

    print(f"💾 Saving {len(final_df)} rows to {OUTPUT_FILE}...")
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print("✅ Done.")


if __name__ == "__main__":
    main()
