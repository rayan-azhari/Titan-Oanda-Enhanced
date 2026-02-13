"""
nautilus_oanda.parsing
----------------------

Utilities for parsing OANDA API data into NautilusTrader objects.
"""

from decimal import Decimal
from typing import Optional

import pandas as pd
from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.enums import AssetClass
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


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
    """Convert OANDA symbol (e.g., 'EUR_USD') to Nautilus InstrumentId."""
    # OANDA uses underscore separator, Nautilus typically expects formatted symbols
    # Here we map OANDA format directly to Nautilus InstrumentId
    symbol = Symbol(oanda_symbol)
    venue = Venue("OANDA")
    return InstrumentId(symbol, venue)


def parse_datetime(oanda_time: str) -> pd.Timestamp:
    """Convert OANDA RFC3339 time string to pandas Timestamp."""
    return pd.Timestamp(oanda_time)


def parse_price(price_str: str) -> Price:
    """Convert OANDA price string to Nautilus Price."""
    return Price(Decimal(price_str), precision=None)  # Precision inferred


def parse_quantity(units_str: str) -> Quantity:
    """Convert OANDA units string to Nautilus Quantity."""
    return Quantity(Decimal(units_str), precision=0)  # OANDA units are integers? Verify later.
