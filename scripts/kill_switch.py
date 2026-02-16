#!/usr/bin/env python
"""kill_switch.py — Emergency: flatten all positions and cancel all pending orders.

Wrapper for titan.utils.ops.
"""

import os
import sys
from pathlib import Path

# Add project root to path for local execution config loading
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment using the project root
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import oandapyV20
import titan.utils.ops

# Load environment variables
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")


def main():
    if not all([ACCOUNT_ID, ACCESS_TOKEN]):
        print("ERROR: OANDA_ACCOUNT_ID and OANDA_ACCESS_TOKEN must be set in .env")
        sys.exit(1)

    client = oandapyV20.API(access_token=ACCESS_TOKEN, environment=ENVIRONMENT)

    print("=" * 40)
    print("🚨 Initiating emergency kill switch...")
    print("=" * 40)

    # Cancel all pending orders
    titan.utils.ops.cancel_all_orders(client, ACCOUNT_ID)

    # Close all open positions
    titan.utils.ops.close_all_positions(client, ACCOUNT_ID)

    print("=" * 40)
    print("✅ All positions flattened. All orders cancelled.\n")


if __name__ == "__main__":
    main()
