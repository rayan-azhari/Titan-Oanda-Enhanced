"""titan/data/oanda.py — OANDA Data Fetching Logic."""

import time
import oandapyV20
import oandapyV20.endpoints.instruments as instruments_ep
import pandas as pd
from decimal import Decimal


def fetch_candles(
    client: oandapyV20.API,
    instrument: str,
    granularity: str,
    from_time: str | None = None,
    count: int = 5000,
) -> list[dict]:
    """Fetch candles from OANDA with exponential back-off on rate limits."""
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
                continue
            else:
                raise
    return []


def candles_to_dataframe(candles: list[dict]) -> pd.DataFrame:
    """Convert OANDA candle response to a clean DataFrame."""
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
