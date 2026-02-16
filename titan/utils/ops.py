"""ops.py — Shared OANDA operations.

Provides functions to cancel orders and close positions.
"""

import oandapyV20
import oandapyV20.endpoints.orders as orders_ep
import oandapyV20.endpoints.positions as positions_ep


def cancel_all_orders(client: oandapyV20.API, account_id: str) -> None:
    """Cancel all pending orders for the account."""
    r = orders_ep.OrdersPending(account_id)
    try:
        response = client.request(r)
        pending = response.get("orders", [])
        for order in pending:
            cancel_r = orders_ep.OrderCancel(account_id, order["id"])
            client.request(cancel_r)
            print(f"  ✗ Cancelled order {order['id']} ({order['instrument']})")
        if not pending:
            print("  No pending orders.")
    except Exception as e:
        print(f"  ERROR cancelling orders: {e}")


def close_all_positions(client: oandapyV20.API, account_id: str) -> None:
    """Close all open positions for the account."""
    r = positions_ep.OpenPositions(account_id)
    try:
        response = client.request(r)
        open_positions = response.get("positions", [])
        for pos in open_positions:
            instrument = pos["instrument"]
            long_units = pos.get("long", {}).get("units", "0")
            short_units = pos.get("short", {}).get("units", "0")

            if long_units != "0":
                close_r = positions_ep.PositionClose(
                    account_id, instrument, data={"longUnits": "ALL"}
                )
                client.request(close_r)
                print(f"  ✗ Closed LONG {instrument} ({long_units} units)")

            if short_units != "0":
                close_r = positions_ep.PositionClose(
                    account_id, instrument, data={"shortUnits": "ALL"}
                )
                client.request(close_r)
                print(f"  ✗ Closed SHORT {instrument} ({short_units} units)")

        if not open_positions:
            print("  No open positions.")
    except Exception as e:
        print(f"  ERROR closing positions: {e}")
