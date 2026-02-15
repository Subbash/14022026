"""Utilities for ADQ adjustable analog bias (input offset)."""

INT16_MIN = -32768
INT16_MAX = 32767


def compute_bias_codes(bias_mV: float, actual_range_mVpp: float) -> int:
    """Convert analog bias in mV to ADQ adjustable-bias codes (int16-safe)."""
    if actual_range_mVpp <= 0:
        raise ValueError("actual_range_mVpp must be > 0")

    full_scale_half_mV = actual_range_mVpp / 2.0
    bias_codes = int(round((bias_mV / full_scale_half_mV) * (2 ** 15)))
    return max(INT16_MIN, min(INT16_MAX, bias_codes))


def bias_conversion_demo() -> str:
    """Small self-check/demo for bias conversion."""
    demo_codes = compute_bias_codes(-100.0, 500.0)
    return f"Bias conversion demo: bias_mV=-100.0, range_mVpp=500.0 -> codes={demo_codes}"
