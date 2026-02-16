"""nautilus_oanda.instruments
--------------------------

Instrument provider for OANDA.
"""

from decimal import Decimal
from typing import List

import oandapyV20
import oandapyV20.endpoints.accounts as accounts
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments.currency_pair import CurrencyPair

from .config import OandaInstrumentProviderConfig


class OandaInstrumentProvider(InstrumentProvider):
    """Provides instruments from OANDA."""

    def __init__(self, config: OandaInstrumentProviderConfig):
        super().__init__(config)
        self._config = config
        self._client = oandapyV20.API(
            access_token=config.access_token, environment=config.environment
        )
        self._account_id = config.account_id

    def load_all(self) -> List[CurrencyPair]:
        """Fetch all available instruments with retries."""
        r = accounts.AccountInstruments(accountID=self._account_id)
        
        # Retry mechanism for OANDA connection
        max_retries = 5
        import time
        import logging
        import socket
        logger = logging.getLogger(__name__)

        default_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(30)

        for attempt in range(1, max_retries + 1):
            try:
                print(f"INFO:titan.oanda:Loading instruments (Attempt {attempt}/{max_retries})...")
                self._client.request(r)
                print("INFO:titan.oanda:Instruments loaded successfully.")
                break
            except Exception as e:
                logger.warning(f"Failed to load instruments (Attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    socket.setdefaulttimeout(default_timeout)
                    raise e
                time.sleep(2 * attempt) # Exponential backoff
        
        socket.setdefaulttimeout(default_timeout)

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

            # create instrument

            # Use from_dict to handle internal CurrencyPair structure robustly
            margin_rate = inst.get("marginRate", "0.02")

            inst_dict = {
                "id": str(instrument_id),
                "symbol": oanda_name,
                "raw_symbol": oanda_name,
                "venue": "OANDA",
                "base_currency": base,
                "quote_currency": quote,
                "multiplier": "1",
                "price_precision": int(display_precision),
                "size_precision": 0,
                "price_increment": str(tick_size),
                "size_increment": "1",
                "lot_size": "1",
                "max_quantity": inst.get("maximumOrderUnits", "1000000000"),
                "min_quantity": inst.get("minimumOrderUnits", "1"),
                "max_price": "100000.00000",
                "min_price": "0.00001",
                "max_notional": f"1000000000.00 {quote}",
                "min_notional": f"1.00 {quote}",
                "maker_fee": "0.0000",
                "taker_fee": "0.0000",
                "ts_event": 0,
                "ts_init": 0,
                "margin_init": str(margin_rate),
                "margin_maint": str(margin_rate),
                "info": {},
            }

            currency_pair = CurrencyPair.from_dict(inst_dict)

            nautilus_instruments.append(currency_pair)

        return nautilus_instruments
