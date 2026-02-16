"""nautilus_oanda package
----------------------

NautilusTrader adapter for OANDA (v20 API).

Components:
- OandaInstrumentProvider: Fetches instrument definitions.
- OandaDataClient: Streams live quotes.
- OandaExecutionClient: Handles order execution.
"""

from .config import OandaDataClientConfig, OandaExecutionClientConfig, OandaInstrumentProviderConfig
from .data import OandaDataClient
from .execution import OandaExecutionClient
from .instruments import OandaInstrumentProvider

__all__ = [
    "OandaInstrumentProviderConfig",
    "OandaDataClientConfig",
    "OandaExecutionClientConfig",
    "OandaInstrumentProvider",
    "OandaDataClient",
    "OandaExecutionClient",
]
