"""parse_oanda_instruments.py — Generate the Nautilus instrument provider.

Fetches the full instrument list from OANDA and generates
oanda_instrument_provider.py with correct tick_size, lot_size,
and margin requirements for each instrument.

Directive: Nautilus-Oanda Adapter Construction.md (Phase 1)
"""

import os
import sys
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import oandapyV20
import oandapyV20.endpoints.accounts as accounts

ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")

OUTPUT_DIR = PROJECT_ROOT / "titan" / "adapters" / "oanda"


def fetch_instruments(client: oandapyV20.API) -> list[dict]:
    """Fetch all available instruments from OANDA.

    Args:
        client: Authenticated OANDA API client.

    Returns:
        List of instrument dictionaries from OANDA.
    """
    r = accounts.AccountInstruments(ACCOUNT_ID)
    response = client.request(r)
    return response.get("instruments", [])


def parse_tick_size(display_precision: int) -> Decimal:
    """Convert OANDA's displayPrecision to a tick size.

    Args:
        display_precision: Number of decimal places (e.g., 5 for EUR/USD).

    Returns:
        Tick size as Decimal (e.g., Decimal("0.00001")).
    """
    return Decimal(10) ** -display_precision


def generate_provider_code(instruments: list[dict]) -> str:
    """Generate Python source code for the instrument provider.

    Args:
        instruments: List of OANDA instrument dictionaries.

    Returns:
        Python source code string.
    """
    lines = [
        '"""oanda_instrument_provider.py — Auto-generated OANDA instrument definitions.',
        "",
        "DO NOT EDIT MANUALLY. Regenerate with: uv run python scripts/fetch_instruments.py",
        '"""',
        "",
        "from decimal import Decimal",
        "",
        "",
        "OANDA_INSTRUMENTS = {",
    ]

    for inst in sorted(instruments, key=lambda x: x["name"]):
        name = inst["name"]
        inst_type = inst["type"]
        precision = inst.get("displayPrecision", 5)
        tick_size = parse_tick_size(precision)
        pip_location = inst.get("pipLocation", -4)
        min_units = inst.get("minimumTradeSize", "1")
        margin_rate = inst.get("marginRate", "0.05")

        lines.append(f'    "{name}": {{')
        lines.append(f'        "type": "{inst_type}",')
        lines.append(f'        "display_precision": {precision},')
        lines.append(f'        "tick_size": Decimal("{tick_size}"),')
        lines.append(f'        "pip_location": {pip_location},')
        lines.append(f'        "min_trade_size": Decimal("{min_units}"),')
        lines.append(f'        "margin_rate": Decimal("{margin_rate}"),')
        lines.append("    },")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Fetch OANDA instruments and generate the instrument provider module."""
    if not ACCOUNT_ID or not ACCESS_TOKEN:
        print("ERROR: OANDA credentials not set in .env")
        sys.exit(1)

    client = oandapyV20.API(access_token=ACCESS_TOKEN, environment=ENVIRONMENT)

    print("🔧 Fetching OANDA instruments...")
    instruments = fetch_instruments(client)
    print(f"   Found {len(instruments)} instruments.")

    code = generate_provider_code(instruments)
    output_path = OUTPUT_DIR / "oanda_instrument_provider.py"
    output_path.write_text(code)
    print(f"   ✓ Generated {output_path}")
    print("   Validate with: uv run python -m pytest tests/test_instrument_parsing.py")

    print("\n✅ Instrument provider generated.\n")


if __name__ == "__main__":
    main()
