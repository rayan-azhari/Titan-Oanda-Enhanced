"""test_oanda_reconciliation.py — Tests for OANDA position reconciliation."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nautilus_trader.model.enums import AccountType, OmsType, PositionSide
from nautilus_trader.model.identifiers import ClientId, Venue
from nautilus_trader.model.objects import Currency

from titan.adapters.oanda.config import OandaExecutionClientConfig


class MockBase:
    def __init__(self, *args, **kwargs):
        pass

def _get_patched_client_class():
    """Import OandaExecutionClient with its base class patched to MockBase.

    This bypasses all Cython/Nautilus restrictions (readonly attrs, type checks)
    and allows pure logic testing of the python subclass.
    """
    with patch("nautilus_trader.live.execution_client.LiveExecutionClient", MockBase):
        # Force reload to apply patch to base class resolution
        if 'titan.adapters.oanda.execution' in sys.modules:
             del sys.modules['titan.adapters.oanda.execution']
        from titan.adapters.oanda.execution import OandaExecutionClient
        return OandaExecutionClient

def _make_client():
    """Create a test OandaExecutionClient with mocked base class."""
    loop = asyncio.new_event_loop()

    config = OandaExecutionClientConfig(
        account_id="101-001-1234567-001",
        access_token="fake-token",
        environment="practice",
    )

    ClientClass = _get_patched_client_class()

    # Instantiate with arbitrary arguments (type checks are mocked out)
    client = ClientClass(
        loop=loop,
        client_id=ClientId("TEST_CLIENT"),
        venue=Venue("OANDA"),
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        base_currency=Currency.from_str("USD"),
        instrument_provider=MagicMock(),
        config=config,
        msgbus=MagicMock(),
        cache=MagicMock(),
        clock=MagicMock(),
    )

    # Configure the python attributes needed by logic
    # client._clock is now a writable attribute on the Mock subclass
    client._loop = loop
    client._clock = MagicMock()
    client._clock.timestamp_ns.return_value = 1625097600000000000

    # Logic uses self._log to define errors
    client._log = MagicMock()

    # API mock (normally set by OandaExecutionClient.__init__)
    # We verify it exists or set it if needed (it should be set by init logic)
    if not hasattr(client, '_api'):
        client._api = MagicMock()
    else:
        # It was set by __init__, but we replace it with a fresh Mock for test control
        client._api = MagicMock()

    return client, loop


def test_long_position():
    """Long position should produce a report with LONG side and correct qty."""
    client, loop = _make_client()
    client._api.request.return_value = {
        "positions": [
            {
                "instrument": "EUR_USD",
                "long": {"units": "1000", "averagePrice": "1.1050"},
                "short": {"units": "0", "averagePrice": "0.0000"},
            }
        ]
    }
    reports = loop.run_until_complete(
        client.generate_position_status_reports(command=None)
    )
    assert len(reports) == 1
    assert reports[0].instrument_id.symbol.value == "EUR/USD"
    assert reports[0].position_side == PositionSide.LONG
    assert reports[0].quantity.as_double() == 1000.0
    loop.close()


def test_short_position():
    """Short position (negative units in V20) should produce LONG=0, SHORT=-500 → net SHORT."""
    client, loop = _make_client()
    client._api.request.return_value = {
        "positions": [
            {
                "instrument": "GBP_USD",
                "long": {"units": "0", "averagePrice": "0.0000"},
                "short": {"units": "-500", "averagePrice": "1.2500"},
            }
        ]
    }
    reports = loop.run_until_complete(
        client.generate_position_status_reports(command=None)
    )
    assert len(reports) == 1
    assert reports[0].instrument_id.symbol.value == "GBP/USD"
    assert reports[0].position_side == PositionSide.SHORT
    assert reports[0].quantity.as_double() == 500.0
    loop.close()


def test_net_position():
    """When both long and short exist, report the net direction and qty."""
    client, loop = _make_client()
    client._api.request.return_value = {
        "positions": [
            {
                "instrument": "USD_JPY",
                "long": {"units": "1000", "averagePrice": "145.00"},
                "short": {"units": "-1500", "averagePrice": "146.00"},
            }
        ]
    }
    reports = loop.run_until_complete(
        client.generate_position_status_reports(command=None)
    )
    assert len(reports) == 1
    assert reports[0].instrument_id.symbol.value == "USD/JPY"
    assert reports[0].position_side == PositionSide.SHORT
    assert reports[0].quantity.as_double() == 500.0
    loop.close()


def test_empty_positions():
    """No open positions should return an empty list."""
    client, loop = _make_client()
    client._api.request.return_value = {"positions": []}
    reports = loop.run_until_complete(
        client.generate_position_status_reports(command=None)
    )
    assert len(reports) == 0
    loop.close()


def test_flat_position_skipped():
    """A position with net_units == 0 should be skipped."""
    client, loop = _make_client()
    client._api.request.return_value = {
        "positions": [
            {
                "instrument": "AUD_USD",
                "long": {"units": "500", "averagePrice": "0.6500"},
                "short": {"units": "-500", "averagePrice": "0.6510"},
            }
        ]
    }
    reports = loop.run_until_complete(
        client.generate_position_status_reports(command=None)
    )
    assert len(reports) == 0
    loop.close()


def test_api_error_returns_empty():
    """API errors should be caught gracefully, returning empty list."""
    client, loop = _make_client()
    client._api.request.side_effect = Exception("Connection failed")
    reports = loop.run_until_complete(
        client.generate_position_status_reports(command=None)
    )
    assert len(reports) == 0
    loop.close()
