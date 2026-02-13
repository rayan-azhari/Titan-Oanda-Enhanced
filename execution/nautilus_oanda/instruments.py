"""
nautilus_oanda.instruments
--------------------------

Instrument provider for OANDA.
"""

from decimal import Decimal
from typing import List

import oandapyV20
import oandapyV20.endpoints.accounts as accounts
from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity

from .config import OandaInstrumentProviderConfig
from .parsing import get_environment_url


class OandaInstrumentProvider:
    """Provides instruments from OANDA."""

    def __init__(self, config: OandaInstrumentProviderConfig):
        self._config = config
        self._client = oandapyV20.API(
            access_token=config.access_token,
            environment=config.environment
        )
        self._account_id = config.account_id

    def load_all(self) -> List[CurrencyPair]:
        """Fetch all available instruments and convert to Nautilus CurrencyPairs."""
        r = accounts.AccountInstruments(accountID=self._account_id)
        self._client.request(r)
        instruments = r.response.get("instruments", [])

        nautilus_instruments = []
        for inst in instruments:
            if inst["type"] != "CURRENCY":
                continue  # Skip CFDs/Metals for now, focus on FX

            # OANDA symbol: EUR_USD
            # Nautilus symbol: EUR/USD
            oanda_name = inst["name"]
            base, quote = oanda_name.split("_")
            
            # Create Identifiers
            venue = Venue("OANDA")
            symbol = Symbol(f"{base}/{quote}")
            instrument_id = InstrumentId(symbol, venue)
            
            # Parse precision and limits
            display_precision = inst["displayPrecision"]
            tick_size = Decimal(10) ** -display_precision
            
            # Lot size - OANDA allows 1 unit, but standard lot is 100,000
            # We set lot_size to 1 for OANDA (micro-lots/units)
            lot_size = Quantity(1, precision=0) 
            
            # create instrument
            native_symbol = Symbol(oanda_name)
            
            currency_pair = CurrencyPair(
                instrument_id=instrument_id,
                native_symbol=native_symbol,
                currency_base=Currency(base),
                currency_quote=Currency(quote),
                price_precision=display_precision,
                size_precision=0,  # OANDA units are integers
                price_increment=Price(tick_size, precision=display_precision),
                size_increment=lot_size,
                lot_size=lot_size,
                max_quantity=Quantity(1_000_000_000, precision=0), # Arbitrary large cap
                min_quantity=Quantity(1, precision=0),
                im_factor=Decimal(inst.get("marginRate", "0.02")),
                mm_factor=Decimal(inst.get("marginRate", "0.02")), # Approx same
            )
            
            nautilus_instruments.append(currency_pair)
            
        return nautilus_instruments
