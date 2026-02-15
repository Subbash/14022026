import time
import numpy as np
from adq_controller import (
    adq_arm, adq_disarm, adq_acquire_once, adq_close, adq_connect,
    adq_acquire_data, adq_acquire_ch0, adq_create_buffers,
)
from logger_utils import log_message

# ---- Try importing CuPy for GPU-accelerated averaging ----
try:
    import cupy as cp
    HAS_CUPY = True
    print("CuPy available - GPU acceleration enabled for trace processing")
except ImportError:
    HAS_CUPY = False
    print("CuPy not available - using NumPy (CPU) for trace processing")


def measure_pump_pulse(adq_config, attenuation_db: float = 0.0, pd_gain_v_per_w: float = 2250.0):
    
    print("\n--- Measuring pump pulse on Channel 1 ---")

    # Uses adq_acquire_once (arms internally, returns both channels)
    handle = adq_connect(adq_config)
    ch0, ch1 = adq_acquire_once(handle)
    adq_disarm(handle)
    adq_close(handle)

    pulse_trace = ch1  # already float32
    fs = adq_config["sample_rate"]
    t_us = np.arange(len(pulse_trace)) / fs * 1e6

    peak_value_mv = float(np.max(pulse_trace))
    peak_index = int(np.argmax(pulse_trace))
    peak_time_us = t_us[peak_index]
    peak_voltage_v = peak_value_mv / 1000.0

    peak_power_w = peak_voltage_v / pd_gain_v_per_w
    incident_power_w = peak_power_w * (10 ** (attenuation_db / 10.0))
    incident_power_dbm = 10 * np.log10(incident_power_w / 1e-3)

    print(f"Samples: {len(pulse_trace)} @ {fs/1e6:.1f} MS/s")
    print(f"Peak: {peak_value_mv:.2f} mV at {peak_time_us:.3f} us")
    print(f"-> Optical power after attenuator: {peak_power_w*1e3:.3f} mW")
    print(f"-> Optical power before attenuator ({attenuation_db:.1f} dB): "
          f"{incident_power_w*1e3:.3f} mW  ({incident_power_dbm:.2f} dBm)")

    return (
        peak_value_mv,
        peak_time_us,
        peak_power_w,
        incident_power_w,
        incident_power_dbm,
        pulse_trace,
    )


# ---- Legacy acquire functions (kept for backward compatibility) ----
def acquire_single_trace_raw_ch0(handle) -> np.ndarray:
    adq_arm(handle)
    ch0, ch1 = adq_acquire_once(handle)
    adq_disarm(handle)
    return ch0.astype(np.float32)

def acquire_single_trace_raw_ch1(handle) -> np.ndarray:
    adq_arm(handle)
    ch0, ch1 = adq_acquire_once(handle)
    adq_disarm(handle)
    return ch1.astype(np.float32)


def average_spatial_samples(trace: np.ndarray, group_size) -> np.ndarray:
    n = len(trace)
    usable = n - (n % group_size)
    reshaped = trace[:usable].reshape(-1, group_size)
    return reshaped.mean(axis=1)


def compute_processed_length(cfg: dict, valid_window_us: float) -> int:
    """
    Compute the length of a processed trace BEFORE acquiring any data.
    Use this to pre-allocate the output array.
    """
    fs = float(cfg["sample_rate"])
    prf = float(cfg["PRF"])
    samples_per_period = int(round(fs / prf))
    valid_samples = int(round(valid_window_us * fs))
    valid_samples = min(valid_samples, samples_per_period)
    return valid_samples


def process_trace(trace: np.ndarray, cfg: dict, valid_window_us: float, apply_dc: bool = True):
    """
    Process a raw trace: reshape into PRF periods, average, remove DC.
    
    NOTE: trace can be the shared pre-allocated buffer from adq_acquire_ch0.
    This is safe because .mean() creates a new independent array before
    the next acquire call overwrites the buffer.
    """
    fs = float(cfg["sample_rate"])
    prf = float(cfg["PRF"])
    pw = float(cfg.get("pulselength"))

    group_size = int(pw * fs * 1e-9)

    # Derived sizes
    samples_per_period = int(round(fs / prf))
    valid_samples = int(round(valid_window_us * fs))

    n = trace.size

    # Trim to whole PRF periods (views only, no copies)
    n_periods = n // samples_per_period
    if n_periods == 0:
        raise RuntimeError("Not enough samples for a single PRF period.")
    use_samples = n_periods * samples_per_period
    diff = trace[:use_samples]

    # Reshape into (n_periods, samples_per_period) — view, no copy
    periods = diff.reshape(n_periods, samples_per_period)
    valid_samples = min(valid_samples, samples_per_period)
    periods_window = periods[:, :valid_samples]

    # Average across periods (this creates a NEW array — safe)
    if HAS_CUPY:
        pw_gpu = cp.asarray(periods_window)
        processed = cp.mean(pw_gpu, axis=0, dtype=cp.float64).astype(cp.float32).get()
    else:
        processed = periods_window.mean(axis=0, dtype=np.float64).astype(np.float32)

    # DC offset removal
    dc_offset = None
    if apply_dc:
        tail_region = periods[:, valid_samples:]
        if tail_region.size == 0:
            raise RuntimeError("No tail samples available for DC estimation.")

        if HAS_CUPY:
            dc_offset = float(cp.mean(cp.asarray(tail_region)))
        else:
            dc_offset = float(np.mean(tail_region))

        processed = processed - dc_offset

    meta_info = dict(
        sample_rate=fs,
        prf=prf,
        samples_per_period=samples_per_period,
        valid_samples=valid_samples,
        valid_window_us=valid_window_us,
        raw_length=n,
        usable_samples=use_samples,
        n_periods=n_periods,
        periods_window_shape=periods_window.shape,
        processed_shape=processed.shape,
        dc_offset=dc_offset,
    )

    return processed, meta_info


def bulk_process_traces(raw_traces, cfg, valid_window_us, apply_dc=True):
    """Kept for backward compatibility."""
    processed_traces = []
    meta_list = []
    for raw in raw_traces:
        proc, meta = process_trace(raw, cfg, valid_window_us, apply_dc=apply_dc)
        processed_traces.append(proc)
        meta_list.append(meta)
    return np.vstack(processed_traces), meta_list
