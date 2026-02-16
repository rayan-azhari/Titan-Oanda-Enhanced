# Directive: Nautilus-Oanda Adapter Construction

## Goal

Build a high-performance, deterministic `nautilus_oanda` adapter package using a **hybrid wrapper approach**. This enables event-driven live trading directly from the Algo Engine.

## Architecture

The adapter connects NautilusTrader's abstract interfaces to OANDA's v20 REST and Streaming APIs.

| Component | Nautilus Interface | OANDA Endpoint | Implementation |
|---|---|---|---|
| **Instruments** | `InstrumentProvider` | `GET /v3/accounts/{id}/instruments` | `titan/adapters/oanda/instruments.py` |
| **Data** | `LiveDataClient` | `GET /v3/accounts/{id}/pricing/stream` | `titan/adapters/oanda/data.py` |
| **Execution** | `LiveExecutionClient` | `POST /v3/accounts/{id}/orders` | `titan/adapters/oanda/execution.py` |

## Implementation Details

### Phase 1 — Instrumentation (`instruments.py`)
- Fetches all tradeable currency pairs.
- Maps OANDA precision and margin rates to Nautilus `CurrencyPair` objects.
- Handles `tick_size` resolution (e.g., 0.00001 for EUR/USD).

### Phase 2 — Streaming Data (`data.py`)
- Wraps `oandapyV20` streaming endpoints.
- Runs blocking stream consumption in a separate thread (executor).
- Pushes `QuoteTick` events to the Nautilus message bus.
- **Constraint:** Uses `oandapyV20` for connection handling to maximize compatibility.

### Phase 3 — Execution Logic (`execution.py`)
- Handles order submission via `LiveExecutionClient`.
- Maps Nautilus `Order` objects to OANDA JSON payload.
- Listens to the **Transactions Stream** for execution reports (fills, cancels).

### Phase 4 — Integration (`run_nautilus_live.py`)
- New entry point `scripts/run_live_ml.py`.
- Configures `TradingNode` with the custom adapter components.
- Loads instruments and starts the event loop.

## Validation

- **Environment:** Requires `nautilus_trader` (Rust extension). verified installed.
- **Unit Tests:** `tests/test_nautilus_instruments.py` (and others).
- **Live Test:** `scripts/run_live_ml.py` (interactive).