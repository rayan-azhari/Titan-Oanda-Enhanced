"""test_nautilus_instruments.py — Integration test for OandaInstrumentProvider.

Requires live OANDA credentials in .env — skips gracefully if unavailable.
"""

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from titan.adapters.oanda.config import OandaInstrumentProviderConfig
from titan.adapters.oanda.instruments import OandaInstrumentProvider

# Mark the entire module as requiring live credentials
pytestmark = pytest.mark.skipif(
    not os.getenv("OANDA_ACCOUNT_ID") or not os.getenv("OANDA_ACCESS_TOKEN"),
    reason="OANDA credentials not found in environment.",
)


@pytest.fixture(scope="module")
def instruments():
    """Load instruments once for all tests in this module."""
    config = OandaInstrumentProviderConfig(
        account_id=os.getenv("OANDA_ACCOUNT_ID"),
        access_token=os.getenv("OANDA_ACCESS_TOKEN"),
        environment="practice",
    )
    provider = OandaInstrumentProvider(config)
    return provider.load_all()


def test_instruments_loaded(instruments):
    """At least one instrument should be returned."""
    assert len(instruments) > 0


def test_eur_usd_exists(instruments):
    """EUR/USD must be present in the instrument list."""
    eur_usd = next((i for i in instruments if i.id.symbol.value == "EUR/USD"), None)
    assert eur_usd is not None, "EUR/USD not found in instruments"


def test_eur_usd_precision(instruments):
    """EUR/USD should have 5 decimal places."""
    eur_usd = next((i for i in instruments if i.id.symbol.value == "EUR/USD"), None)
    if eur_usd is None:
        pytest.skip("EUR/USD not available")
    assert eur_usd.price_precision == 5


def test_lot_size(instruments):
    """OANDA allows unit-level trading — lot_size should be 1."""
    eur_usd = next((i for i in instruments if i.id.symbol.value == "EUR/USD"), None)
    if eur_usd is None:
        pytest.skip("EUR/USD not available")
    assert eur_usd.lot_size.as_double() == 1.0


def test_venue_is_oanda(instruments):
    """All instruments should belong to OANDA venue."""
    for inst in instruments:
        assert inst.id.venue.value == "OANDA"
