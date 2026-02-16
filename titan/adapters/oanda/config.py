"""nautilus_oanda.config
---------------------

Configuration for the OANDA adapter.
"""

from nautilus_trader.live.config import (
    InstrumentProviderConfig,
    LiveDataClientConfig,
    LiveExecClientConfig,
)


class OandaInstrumentProviderConfig(InstrumentProviderConfig):
    """Configuration for OandaInstrumentProvider."""

    account_id: str = ""
    access_token: str = ""
    environment: str = "practice"  # "practice" or "live"


class OandaDataClientConfig(LiveDataClientConfig):
    """Configuration for OandaDataClient."""

    account_id: str = ""
    access_token: str = ""
    environment: str = "practice"
    # Reconnection logic
    reconnect_attempts: int = 10
    reconnect_delay: float = 1.0


class OandaExecutionClientConfig(LiveExecClientConfig):
    """Configuration for OandaExecutionClient."""

    account_id: str = ""
    access_token: str = ""
    environment: str = "practice"
