"""
test_nautilus_instruments.py
----------------------------

Verify OandaInstrumentProvider functionality.
"""

import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

import os
from execution.nautilus_oanda.config import OandaInstrumentProviderConfig
from execution.nautilus_oanda.instruments import OandaInstrumentProvider


def test_instrument_loading():
    """Test loading instruments from OANDA."""
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    access_token = os.getenv("OANDA_ACCESS_TOKEN")
    
    if not account_id or not access_token:
        pytest.skip("OANDA credentials not found in environment.")

    config = OandaInstrumentProviderConfig(
        account_id=account_id,
        access_token=access_token,
        environment="practice"
    )
    
    try:
        provider = OandaInstrumentProvider(config)
        
        print("Fetching instruments...")
        instruments = provider.load_all()
        
        assert len(instruments) > 0
        
        # Check EUR/USD
        eur_usd = next((i for i in instruments if i.id.symbol.value == "EUR/USD"), None)
        assert eur_usd is not None
        assert eur_usd.price_precision == 5
        assert eur_usd.lot_size.as_double() == 1.0
        
        with open("test_success.txt", "w") as f:
            f.write(f"Successfully loaded {len(instruments)} instruments.\nEUR/USD found: {eur_usd}")
            
    except Exception as e:
        with open("test_failure.txt", "w") as f:
            f.write(str(e))
        raise e

if __name__ == "__main__":
    test_instrument_loading()
