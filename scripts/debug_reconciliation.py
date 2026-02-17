import sys
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nautilus_trader.core.uuid import UUID4
from nautilus_trader.execution.reports import PositionStatusReport
from nautilus_trader.model.enums import PositionSide
from nautilus_trader.model.identifiers import AccountId, InstrumentId, PositionId, Symbol, Venue
from nautilus_trader.model.objects import Price, Quantity


# Mock Parsing from titan.adapters.oanda.parsing
def parse_instrument_id(oanda_symbol: str) -> InstrumentId:
    nautilus_symbol = oanda_symbol.replace("_", "/")
    symbol = Symbol(nautilus_symbol)
    venue = Venue("OANDA")
    return InstrumentId(symbol, venue)


def main():
    # Mock Data from OANDA (from check_oanda_positions.py output)
    p_data = {
        "instrument": "EUR_USD",
        "long": {
            "units": "1000",
            "averagePrice": "1.18467",
            "pl": "-1.0545",
            "resettablePL": "-1.0545",
            "financing": "-0.0011",
            "dividendAdjustment": "0.0000",
            "guaranteedExecutionFees": "0.0000",
            "tradeIDs": ["91"],
            "trueUnrealizedPL": "0.0732",
            "unrealizedPL": "0.0732",
        },
        "short": {
            "units": "0",
            "pl": "0.0000",
            "resettablePL": "0.0000",
            "financing": "0.0000",
            "dividendAdjustment": "0.0000",
            "guaranteedExecutionFees": "0.0000",
            "trueUnrealizedPL": "0.0000",
            "unrealizedPL": "0.0000",
        },
        "pl": "-1.0545",
        "resettablePL": "-1.0545",
        "financing": "-0.0011",
        "commission": "0.0000",
        "dividendAdjustment": "0.0000",
        "guaranteedExecutionFees": "0.0000",
        "trueUnrealizedPL": "0.0732",
        "unrealizedPL": "0.0732",
        "marginUsed": "29.0567",
    }

    try:
        print("Starting Logic...")

        instrument_id = parse_instrument_id(p_data.get("instrument", ""))
        print(f"Instrument ID: {instrument_id}")

        long_units = int(p_data.get("long", {}).get("units", "0"))
        short_units = int(p_data.get("short", {}).get("units", "0"))
        net_units = long_units + short_units
        print(f"Net Units: {net_units}")

        if net_units > 0:
            position_side = PositionSide.LONG
            avg_px_str = p_data.get("long", {}).get("averagePrice", "0")
        else:
            position_side = PositionSide.SHORT
            avg_px_str = p_data.get("short", {}).get("averagePrice", "0")

        print(f"Avg Px Str: {avg_px_str}")
        print(f"Position Side: {position_side}")

        account_id_str = "101-004-38378020-001"

        # Test Price creation
        price = Price.from_str(avg_px_str)
        print(f"Price Object: {price}")

        # Test Quantity creation
        qty = Quantity(abs(net_units), precision=0)
        print(f"Quantity Object: {qty}")

        # Test VenuePositionId
        # NOTE: Nautilus PostionId must be a UUID or string?
        # PositionId(value: str)
        # In execution.py: venue_position_id=PositionId(instrument_id.value),

        vpid_str = instrument_id.value
        print(f"VPID Str: {vpid_str}")
        vpid = PositionId(vpid_str)
        print(f"VPID Object: {vpid}")

        report = PositionStatusReport(
            account_id=AccountId(f"OANDA-{account_id_str}"),
            instrument_id=instrument_id,
            position_side=position_side,
            quantity=qty,
            report_id=UUID4(),
            ts_last=123456789,
            ts_init=123456789,
            venue_position_id=vpid,
            avg_px_open=price,
        )
        print("Report created successfully!")
        print(report)

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
