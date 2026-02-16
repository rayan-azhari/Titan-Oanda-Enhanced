"""Tests for the Ehlers Gaussian Channel indicator."""

import numpy as np

from titan.indicators.gaussian_filter import (
    GaussianChannel,
    _gaussian_channel_kernel,
    get_gaussian_alpha,
)


def test_alpha_calculation():
    """Alpha should be 1.0 for sub-Nyquist periods and decrease for larger ones."""
    # Period < 2 → Nyquist guard → alpha = 1.0
    assert get_gaussian_alpha(1.0, 1) == 1.0

    # Large period → small alpha (heavy smoothing)
    a_large = get_gaussian_alpha(100.0, 1)
    assert 0 < a_large < 0.1

    # For same period, higher poles → higher per-stage alpha
    a1 = get_gaussian_alpha(20.0, 1)
    a4 = get_gaussian_alpha(20.0, 4)
    assert a4 > a1


def test_kernel_output_shapes_and_bounds():
    """Kernel must return 3 arrays; upper >= middle >= lower."""
    np.random.seed(42)
    close = np.random.randn(200).cumsum() + 100.0
    high = close + np.abs(np.random.randn(200)) * 0.5
    low = close - np.abs(np.random.randn(200)) * 0.5

    upper, lower, mid = _gaussian_channel_kernel(high, low, close, 20.0, 4, 2.0)

    assert upper.shape == close.shape
    assert lower.shape == close.shape
    assert mid.shape == close.shape

    # TR is always >= 0, so upper >= middle >= lower everywhere
    assert np.all(upper >= mid)
    assert np.all(mid >= lower)

    # No NaNs after some warmup
    assert not np.isnan(mid[50])


def test_vbt_factory_single_param():
    """GaussianChannel.run() should accept pandas-like inputs."""
    np.random.seed(42)
    close = np.random.randn(100).cumsum() + 100.0
    high = close + 1.0
    low = close - 1.0

    gc = GaussianChannel.run(high, low, close, period=20, poles=4, sigma=2.0)

    assert gc.upper.shape == (100,)
    assert gc.lower.shape == (100,)
    assert gc.middle.shape == (100,)


def test_vbt_factory_multi_param():
    """Multiple period values should broadcast to a 2-D output."""
    np.random.seed(42)
    close = np.arange(100, dtype=np.float64)
    high = close + 1.0
    low = close - 1.0

    gc = GaussianChannel.run(high, low, close, period=[10, 20], poles=2, sigma=2.0)
    assert gc.middle.shape == (100, 2)
