"""fetch_eur_usd.py — Download 1 year of EUR/USD OHLC data from OANDA.

Fetches candles across all granularities defined in config/instruments.toml
using paginated requests (OANDA caps at 5000 candles per call).

Saves output as Parquet files in data/ folder.

Usage:
    uv run python titan/data/fetch_eur_usd.py
"""

import os
import sys
import time
import tomllib
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import oandapyV20
import oandapyV20.endpoints.instruments as instruments_ep
import pandas as pd

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Target instrument and lookback
INSTRUMENT = "EUR_USD"
LOOKBACK_DAYS = 365


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fetch_candles_page(
    client: oandapyV20.API,
    instrument: str,
    granularity: str,
    from_time: str,
    count: int = 500,
) -> list[dict]:
    """Fetch a single page of candles with retry on rate-limit.

    Uses count-based pagination (from + count), which is the most
    reliable method for the OANDA v20 API.  Max count is 5000 but
    we default to 500 to stay comfortably within limits.
    """
    params: dict = {
        "granularity": granularity,
        "count": count,
        "price": "BA",
        "from": from_time,
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
                raise
    return []


def fetch_all_candles(
    client: oandapyV20.API,
    instrument: str,
    granularity: str,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """Paginate through OANDA to fetch all candles between start and end."""
    all_candles: list[dict] = []
    current_from = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    page = 0

    while True:
        page += 1
        candles = fetch_candles_page(
            client,
            instrument,
            granularity,
            from_time=current_from,
        )

        if not candles:
            break

        all_candles.extend(candles)
        print(
            f"    Page {page}: {len(candles)} candles "
            f"({candles[0]['time'][:10]} → {candles[-1]['time'][:10]})"
        )

        # Stop if the last candle is past our end date
        last_time = candles[-1]["time"]
        if last_time >= end_iso:
            break

        # Advance past the last candle for next page
        current_from = last_time

        # If we got fewer than requested, we've reached the end
        if len(candles) < 500:
            break

        # Small delay to be kind to the API
        time.sleep(0.3)

    return all_candles


def candles_to_dataframe(candles: list[dict]) -> pd.DataFrame:
    """Convert OANDA candle JSON to a clean DataFrame.

    Uses bid prices. Only includes complete candles.
    All prices stored as Decimal for financial precision.
    """
    rows = []
    for c in candles:
        if not c.get("complete", False):
            continue
        bid = c["bid"]
        rows.append(
            {
                "timestamp": pd.Timestamp(c["time"]),
                "open": Decimal(bid["o"]),
                "high": Decimal(bid["h"]),
                "low": Decimal(bid["l"]),
                "close": Decimal(bid["c"]),
                "volume": int(c["volume"]),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Download 1 year of EUR/USD data for all configured granularities."""
    if not ACCESS_TOKEN:
        print("ERROR: OANDA_ACCESS_TOKEN not set in .env")
        sys.exit(1)

    # Load granularities from config
    config_path = PROJECT_ROOT / "config" / "instruments.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    granularities = config.get("instruments", {}).get("granularities", ["H1", "H4", "D", "W"])

    client = oandapyV20.API(access_token=ACCESS_TOKEN, environment=ENVIRONMENT)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)

    print(f"📥 Downloading {INSTRUMENT} data")
    print(f"   Period: {start.date()} → {end.date()} ({LOOKBACK_DAYS} days)")
    print(f"   Granularities: {', '.join(granularities)}")
    print(f"   Output: {DATA_DIR}/\n")

    for gran in granularities:
        print(f"  ▸ {INSTRUMENT} {gran}...")
        candles = fetch_all_candles(client, INSTRUMENT, gran, start, end)
        df = candles_to_dataframe(candles)

        if df.empty:
            print(f"    ⚠ No data returned for {gran}.")
            continue

        # Convert Decimal columns to float for Parquet storage
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)

        output_path = DATA_DIR / f"{INSTRUMENT}_{gran}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"    ✓ {len(df)} rows → {output_path.name}")
        print(f"      Range: {df['timestamp'].min()} → {df['timestamp'].max()}\n")

    print("✅ Done. Files saved to data/\n")

    # Summary
    print("📊 Summary:")
    for f in sorted(DATA_DIR.glob(f"{INSTRUMENT}_*.parquet")):
        size_kb = f.stat().st_size / 1024
        print(f"   {f.name:25s} {size_kb:>7.1f} KB")


if __name__ == "__main__":
    main()
