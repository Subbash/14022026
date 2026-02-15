# ADQ14 Analog DC Bias / Input Offset

This project now supports ADQAPI adjustable **analog bias** (`SetAdjustableBias`) per channel.

## Why this matters

- **Analog bias** shifts the ADC input window before conversion and can prevent clipping.
- **Digital/FPGA offset** only shifts already-digitized samples and does **not** prevent ADC clipping.

## Bias code conversion

Use calibrated range (`actual_range_mVpp`) from `SetInputRange(...)` return value, or `GetInputRange(...)` if range is already set:

```python
bias_codes = round((bias_mV / (actual_range_mVpp / 2.0)) * (2**15))
```

Then clamp to int16 safe range `[-32768, 32767]`.

Example:

- `bias_mV = -100`
- `actual_range_mVpp = 500`
- `bias_codes ~= -13107`

## Application order

1. Set/read input range.
2. Validate support with `HasAdjustableBias()`.
3. Apply bias with `SetAdjustableBias(channel, bias_codes)`.
4. Wait ~0.5 s for analog bias filter settling.
5. Arm and record as usual.
