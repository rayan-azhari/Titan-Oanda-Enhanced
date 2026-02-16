"""verify_connection.py — Validate OANDA API connectivity.

Reads credentials from .env and prints account summary + available instruments.
Directive: 01_environment_setup.md
"""

import os
import sys
from decimal import Decimal
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: python-dotenv is not installed. Run `uv sync` first.")
    sys.exit(1)

try:
    import oandapyV20
    import oandapyV20.endpoints.accounts as accounts
except ImportError:
    print("ERROR: oandapyV20 is not installed. Run `uv sync` first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")


def main() -> None:
    if not ACCOUNT_ID or not ACCESS_TOKEN:
        print("ERROR: OANDA_ACCOUNT_ID and OANDA_ACCESS_TOKEN must be set in .env")
        sys.exit(1)

    client = oandapyV20.API(
        access_token=ACCESS_TOKEN,
        environment=ENVIRONMENT,
    )

    # --- Account Summary ---
    r = accounts.AccountSummary(ACCOUNT_ID)
    try:
        response = client.request(r)
    except oandapyV20.exceptions.V20Error as e:
        print(f"ERROR: OANDA API returned an error:\n{e}")
        sys.exit(1)

    acct = response["account"]
    print("=" * 50)
    print("  OANDA Connection Verified ✓")
    print("=" * 50)
    print(f"  Account ID  : {acct['id']}")
    print(f"  Currency    : {acct['currency']}")
    print(f"  Balance     : {Decimal(acct['balance'])}")
    print(f"  NAV         : {Decimal(acct['NAV'])}")
    print(f"  Open Trades : {acct['openTradeCount']}")
    print(f"  Environment : {ENVIRONMENT}")
    print("=" * 50)

    # --- Available Instruments ---
    r = accounts.AccountInstruments(ACCOUNT_ID)
    response = client.request(r)
    instruments = response["instruments"]
    print(f"\n  {len(instruments)} instruments available.")
    print("  First 10:")
    for inst in instruments[:10]:
        print(f"    • {inst['name']}  ({inst['type']})")

    print("\n✅ All checks passed. You are ready to go.\n")


if __name__ == "__main__":
    main()
