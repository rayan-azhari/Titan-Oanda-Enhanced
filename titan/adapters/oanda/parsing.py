"""nautilus_oanda.parsing
----------------------

Utilities for parsing OANDA API data into NautilusTrader objects.
"""

import pandas as pd
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Price, Quantity

ENVIRONMENT_URLS = {
    "practice": {
        "rest": "api-fxpractice.oanda.com",
        "stream": "stream-fxpractice.oanda.com",
    },
    "live": {
        "rest": "api-fxtrade.oanda.com",
        "stream": "stream-fxtrade.oanda.com",
    },
}


def get_environment_url(environment: str, method: str = "rest") -> str:
    """Get the base URL for the OANDA environment."""
    return ENVIRONMENT_URLS.get(environment, {}).get(method, "")


def parse_instrument_id(oanda_symbol: str) -> InstrumentId:
    """Convert OANDA symbol (e.g., 'EUR_USD') to Nautilus InstrumentId.

    OANDA uses underscore separators (EUR_USD) but the OandaInstrumentProvider
    creates instruments with slash separators (EUR/USD). We must match that
    convention so cache lookups succeed.
    """
    # EUR_USD -> EUR/USD
    nautilus_symbol = oanda_symbol.replace("_", "/")
    symbol = Symbol(nautilus_symbol)
    venue = Venue("OANDA")
    return InstrumentId(symbol, venue)


def parse_datetime(oanda_time: str) -> pd.Timestamp:
    """Convert OANDA RFC3339 time string to pandas Timestamp."""
    return pd.Timestamp(oanda_time)


def parse_price(price_str: str) -> Price:
    """Convert OANDA price string to Nautilus Price."""
    return Price.from_str(price_str)


def parse_quantity(units_str: str) -> Quantity:
    """Convert OANDA units string to Nautilus Quantity."""
    # Use from_str to handle potential decimal points in units (if any, though usually int)
    return Quantity.from_str(units_str)
