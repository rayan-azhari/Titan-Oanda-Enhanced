"""Ehlers Gaussian Filter & Channel indicator for VectorBT.

Implements the "Gaussian Channel [DW]" (DonovanWall) logic:
  1. Compute alpha from period + poles via the Ehlers -3 dB formula.
  2. Cascade a 1-pole EMA `poles` times (= N-pole Gaussian filter).
  3. Filter True Range the same way, then build upper/lower bands.

All heavy loops use Numba @njit; numpy handles array allocation.
"""

import numpy as np
import vectorbt as vbt
from numba import njit

# ---------------------------------------------------------------------------
# Numba kernels — operate on raw float64 arrays only
# ---------------------------------------------------------------------------


@njit(cache=True)
def get_gaussian_alpha(period, poles):
    """Return the EMA alpha for an N-pole Gaussian filter at *period*."""
    if period < 2.0:
        return 1.0  # Nyquist guard — pass everything through

    w = 2.0 * np.pi / period
    c = np.cos(w)
    k = 0.5 ** (1.0 / poles)

    # Quadratic:  (1-k)*a^2 + 2k(1-c)*a - 2k(1-c) = 0
    a_coef = 1.0 - k
    b_coef = 2.0 * k * (1.0 - c)
    c_coef = -b_coef

    if np.abs(a_coef) < 1e-12:
        return 1.0

    disc = b_coef * b_coef - 4.0 * a_coef * c_coef
    if disc < 0.0:
        return 1.0

    alpha = (-b_coef + np.sqrt(disc)) / (2.0 * a_coef)

    # Clamp to [0, 1]
    if alpha < 0.0:
        alpha = 0.0
    if alpha > 1.0:
        alpha = 1.0
    return alpha


@njit(cache=True)
def _ema_cascade(data, alpha, poles):
    """Apply a 1-pole EMA *poles* times (cascade → N-pole Gaussian)."""
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    for j in range(n):
        out[j] = data[j]

    for _ in range(int(poles)):
        prev = np.nan
        for i in range(n):
            v = out[i]
            if np.isnan(v):
                out[i] = np.nan
            else:
                if np.isnan(prev):
                    prev = v
                else:
                    v = alpha * v + (1.0 - alpha) * prev
                    prev = v
                out[i] = v
    return out


@njit(cache=True)
def _true_range(high, low, close):
    """Compute True Range array."""
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = np.abs(high[i] - close[i - 1])
        lc = np.abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
    return tr


@njit(cache=True)
def _gaussian_channel_kernel(high, low, close, period, poles, sigma):
    """Pure-Numba kernel: returns (upper, lower, middle) as float64 arrays."""
    alpha = get_gaussian_alpha(period, poles)
    mid = _ema_cascade(close, alpha, int(poles))
    tr = _true_range(high, low, close)
    ftr = _ema_cascade(tr, alpha, int(poles))
    upper = mid + sigma * ftr
    lower = mid - sigma * ftr
    return upper, lower, mid


# ---------------------------------------------------------------------------
# Python wrapper — converts pandas → numpy for Numba, then returns arrays
# ---------------------------------------------------------------------------


def _gaussian_channel_custom(high_2d, low_2d, close_2d, periods, poles_arr, sigmas, **kwargs):
    """Custom function for VBT ``IndicatorFactory.from_custom_func``.

    VBT 0.28 broadcasts inputs to 2-D with ``n_input_cols`` columns (from the
    original data) and passes each param as an array of ``n_param_combos``
    values.  The required output shape is ``(n_rows, n_input_cols × n_param_combos)``.
    """
    high_2d = np.asarray(high_2d, dtype=np.float64)
    low_2d = np.asarray(low_2d, dtype=np.float64)
    close_2d = np.asarray(close_2d, dtype=np.float64)

    # Ensure 2-D even for single-column input
    if high_2d.ndim == 1:
        high_2d = high_2d[:, np.newaxis]
        low_2d = low_2d[:, np.newaxis]
        close_2d = close_2d[:, np.newaxis]

    n_rows, n_input_cols = close_2d.shape
    n_param_combos = len(periods)
    n_out_cols = n_input_cols * n_param_combos

    upper_out = np.empty((n_rows, n_out_cols), dtype=np.float64)
    lower_out = np.empty((n_rows, n_out_cols), dtype=np.float64)
    mid_out = np.empty((n_rows, n_out_cols), dtype=np.float64)

    out_col = 0
    for ic in range(n_input_cols):
        h = np.ascontiguousarray(high_2d[:, ic])
        lo_arr = np.ascontiguousarray(low_2d[:, ic])
        c = np.ascontiguousarray(close_2d[:, ic])
        for pc in range(n_param_combos):
            u, lo, m = _gaussian_channel_kernel(
                h,
                lo_arr,
                c,
                float(periods[pc]),
                int(poles_arr[pc]),
                float(sigmas[pc]),
            )
            upper_out[:, out_col] = u
            lower_out[:, out_col] = lo
            mid_out[:, out_col] = m
            out_col += 1

    return upper_out, lower_out, mid_out


# ---------------------------------------------------------------------------
# VectorBT Indicator Factory
# ---------------------------------------------------------------------------

GaussianChannel = vbt.IndicatorFactory(
    class_name="GaussianChannel",
    short_name="gauss",
    input_names=["high", "low", "close"],
    param_names=["period", "poles", "sigma"],
    output_names=["upper", "lower", "middle"],
).from_custom_func(
    _gaussian_channel_custom,
    param_defaults={"period": 144, "poles": 4, "sigma": 2.0},
    keep_pd=True,
)
