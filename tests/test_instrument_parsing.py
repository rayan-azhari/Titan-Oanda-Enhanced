"""test_instrument_parsing.py — Unit tests for OANDA parsing utilities.

Validates the parsing module's symbol conversion, price/quantity parsing,
and environment URL resolution — all without requiring a live API connection.

Directive: Nautilus-Oanda Adapter Construction.md (Phase 1)
"""

from decimal import Decimal

from titan.adapters.oanda.parsing import (
    get_environment_url,
    parse_instrument_id,
    parse_price,
    parse_quantity,
)


class TestParseInstrumentId:
    """Test OANDA -> Nautilus symbol conversion."""

    def test_standard_pair(self):
        """EUR_USD should become EUR/USD.OANDA."""
        result = parse_instrument_id("EUR_USD")
        assert result.symbol.value == "EUR/USD"
        assert result.venue.value == "OANDA"

    def test_jpy_pair(self):
        """USD_JPY should map correctly."""
        result = parse_instrument_id("USD_JPY")
        assert result.symbol.value == "USD/JPY"

    def test_cross_pair(self):
        """GBP_AUD cross pair."""
        result = parse_instrument_id("GBP_AUD")
        assert result.symbol.value == "GBP/AUD"


class TestParsePrice:
    """Test OANDA price string -> Nautilus Price."""

    def test_five_decimal(self):
        """EUR/USD-style 5-digit price."""
        price = parse_price("1.10523")
        assert price.as_decimal() == Decimal("1.10523")

    def test_three_decimal(self):
        """JPY-style 3-digit price."""
        price = parse_price("149.123")
        assert price.as_decimal() == Decimal("149.123")


class TestParseQuantity:
    """Test OANDA units string -> Nautilus Quantity."""

    def test_positive_units(self):
        qty = parse_quantity("10000")
        assert int(qty) == 10000

    def test_single_unit(self):
        qty = parse_quantity("1")
        assert int(qty) == 1


class TestEnvironmentUrls:
    """Test OANDA environment URL resolution."""

    def test_practice_rest(self):
        url = get_environment_url("practice", "rest")
        assert "fxpractice" in url

    def test_live_rest(self):
        url = get_environment_url("live", "rest")
        assert "fxtrade" in url

    def test_practice_stream(self):
        url = get_environment_url("practice", "stream")
        assert "stream-fxpractice" in url

    def test_invalid_environment(self):
        url = get_environment_url("invalid", "rest")
        assert url == ""
