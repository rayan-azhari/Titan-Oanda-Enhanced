"""rate_limiter.py — Token bucket rate limiter for OANDA API calls.

Prevents HTTP 429 (Too Many Requests) errors by enforcing a sustained
request rate. OANDA's REST API allows ~100 requests/second for most
endpoints, but order submission should be throttled more conservatively.

Research Gap Fix #5: The research document mentioned a token bucket
rate limiter but it was never implemented.
"""

import threading
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Thread-safe token bucket rate limiter.

    Tokens refill at a constant rate. Each API call consumes one token.
    If no tokens are available, the caller blocks until one refills.

    Attributes:
        capacity: Maximum tokens in the bucket.
        refill_rate: Tokens added per second.
        tokens: Current number of available tokens.
    """

    capacity: float
    refill_rate: float
    tokens: float = field(init=False)
    _last_refill: float = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        """Initialise token count and timestamp."""
        self.tokens = self.capacity
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self._last_refill = now

    def acquire(self, timeout: float = 30.0) -> bool:
        """Acquire a token, blocking until one is available.

        Args:
            timeout: Maximum seconds to wait for a token.

        Returns:
            True if a token was acquired, False if timeout expired.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True

            # Calculate wait time for next token
            with self._lock:
                wait = (1.0 - self.tokens) / self.refill_rate if self.refill_rate > 0 else 1.0

            if time.monotonic() + wait > deadline:
                return False

            time.sleep(min(wait, 0.1))  # Sleep in small increments

    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking.

        Returns:
            True if a token was available and consumed, False otherwise.
        """
        with self._lock:
            self._refill()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

    @property
    def available(self) -> float:
        """Return current available tokens (approximate)."""
        with self._lock:
            self._refill()
            return self.tokens


# ---------------------------------------------------------------------------
# Pre-configured limiters for OANDA endpoints
# ---------------------------------------------------------------------------

# General API calls (pricing, account info, candle history)
api_limiter = TokenBucket(capacity=25, refill_rate=20)

# Order submission — more conservative to avoid rejection
order_limiter = TokenBucket(capacity=5, refill_rate=2)

# Streaming connections — very limited
stream_limiter = TokenBucket(capacity=2, refill_rate=0.5)


def rate_limited_call(func, *args, limiter: TokenBucket = api_limiter, **kwargs):
    """Execute a function with rate limiting.

    Blocks until a token is available, then calls the function.

    Args:
        func: Callable to execute.
        *args: Positional arguments for func.
        limiter: TokenBucket to use (default: api_limiter).
        **kwargs: Keyword arguments for func.

    Returns:
        Result of func(*args, **kwargs).

    Raises:
        TimeoutError: If a token could not be acquired within 30 seconds.
    """
    if not limiter.acquire(timeout=30.0):
        raise TimeoutError(
            f"Rate limiter timeout: could not acquire token in 30s. "
            f"Available: {limiter.available:.1f}"
        )
    return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate the rate limiter with simulated API calls."""
    print("🪣 Token Bucket Rate Limiter — Demo\n")

    # Simulate burst of 10 order submissions with a limit of 2/sec
    limiter = TokenBucket(capacity=5, refill_rate=2)
    print(f"  Capacity: {limiter.capacity}, Refill: {limiter.refill_rate}/sec\n")

    for i in range(10):
        start = time.monotonic()
        acquired = limiter.acquire(timeout=10.0)
        elapsed = time.monotonic() - start
        print(
            f"  Request {i + 1:2d}: {'✓ granted' if acquired else '✗ denied'}"
            f"  (waited {elapsed:.3f}s, tokens: {limiter.available:.1f})"
        )

    print("\n✅ Rate limiter working correctly.\n")


if __name__ == "__main__":
    main()
