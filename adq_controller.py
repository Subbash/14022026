import ctypes as ct
import numpy as np
import pyadq
import time
from typing import Dict, Tuple, Optional, Iterable, List
import gc

from adq_bias import compute_bias_codes, bias_conversion_demo


BIAS_SETTLE_SECONDS = 0.5


def _get_range_getter(dev):
    for name in ("GetInputRange", "ADQ_GetInputRange"):
        if hasattr(dev, name):
            return getattr(dev, name)
    return None


def _get_range_setter(dev):
    for name in ("SetInputRange", "ADQ_SetInputRange"):
        if hasattr(dev, name):
            return getattr(dev, name)
    return None


def adq_get_input_range_mVpp(handle: Dict, channel: int) -> float:
    """Read the calibrated/actual input range for a channel in mVpp."""
    dev = handle["dev"]
    getter = _get_range_getter(dev)
    if getter is None:
        raise RuntimeError("Input range getter is not available in this pyadq/ADQAPI build")

    value = getter(channel)
    if isinstance(value, tuple):
        actual_range = value[0]
    else:
        actual_range = value

    return float(actual_range)


def adq_set_input_range_mVpp(handle: Dict, channel: int, requested_range_mVpp: float) -> float:
    """Set range and return actual/calibrated range mVpp reported by ADQAPI."""
    dev = handle["dev"]
    setter = _get_range_setter(dev)
    if setter is None:
        raise RuntimeError("Input range setter is not available in this pyadq/ADQAPI build")

    value = setter(channel, float(requested_range_mVpp))
    if isinstance(value, tuple):
        actual_range = value[0]
    else:
        actual_range = value

    actual_range = float(actual_range)
    handle.setdefault("actual_input_ranges_mVpp", {})[channel] = actual_range
    print(
        f"Input range set: channel={channel}, requested_mVpp={requested_range_mVpp}, "
        f"actual_mVpp={actual_range}"
    )
    return actual_range


def _has_adjustable_bias(dev) -> bool:
    for name in ("HasAdjustableBias", "ADQ_HasAdjustableBias"):
        if hasattr(dev, name):
            return bool(getattr(dev, name)())
    raise RuntimeError("HasAdjustableBias() is not available in this pyadq/ADQAPI build")


def adq_apply_adjustable_bias(
    handle: Dict,
    channels: Iterable[int],
    bias_mV: float,
    settle_seconds: float = BIAS_SETTLE_SECONDS,
    read_back: bool = True,
) -> List[Dict]:
    """Apply ADQAPI adjustable analog bias for selected channel(s)."""
    dev = handle["dev"]
    if not _has_adjustable_bias(dev):
        raise RuntimeError(
            "This ADQ14 variant/firmware does not support adjustable analog bias (DC offset)."
        )

    if not hasattr(dev, "SetAdjustableBias"):
        raise RuntimeError("SetAdjustableBias() is not available in this pyadq/ADQAPI build")

    get_bias = getattr(dev, "GetAdjustableBias", None)
    results = []

    for channel in channels:
        actual_range_mVpp = handle.get("actual_input_ranges_mVpp", {}).get(channel)
        if actual_range_mVpp is None:
            actual_range_mVpp = adq_get_input_range_mVpp(handle, channel)
            handle.setdefault("actual_input_ranges_mVpp", {})[channel] = actual_range_mVpp

        bias_codes = compute_bias_codes(bias_mV, actual_range_mVpp)
        status = dev.SetAdjustableBias(channel, bias_codes)
        rb_codes = get_bias(channel) if (read_back and callable(get_bias)) else None
        print(
            "Apply adjustable bias: "
            f"channel={channel}, bias_mV={bias_mV}, actual_range_mVpp={actual_range_mVpp}, "
            f"bias_codes={bias_codes}, status={status}, readback={rb_codes}"
        )
        results.append(
            {
                "channel": channel,
                "bias_mV": bias_mV,
                "actual_range_mVpp": actual_range_mVpp,
                "bias_codes": bias_codes,
                "status": status,
                "readback_codes": rb_codes,
            }
        )

    if settle_seconds > 0:
        time.sleep(settle_seconds)

    return results


def _calculate_derived_params(cfg: Dict) -> Dict:
    cfg = dict(cfg)  # don't mutate caller's dict
    cfg.setdefault('sample_rate',      100e6)
    cfg.setdefault('sample_length',    100e6)
    cfg.setdefault('PRF',              5000)
    cfg.setdefault('fiblen_actual',    10000)
    cfg.setdefault('number_of_channels', 2)
    cfg.setdefault('pretrigger',       0)
    cfg.setdefault('trigger_delay',    00)
    cfg.setdefault('pulselength',      200)
    cfg.setdefault('trigger_mode',     pyadq.ADQ_INTERNAL_TRIGGER_MODE)
    cfg.setdefault('timeout_seconds',  10)
    cfg.setdefault('filter_size',      1)

    cfg['samples_per_record'] = int(cfg['sample_length'])
    cfg['sample_skip']        = int(500e6 / cfg['sample_rate'])
    cfg['trigger_period']     = int(1 / cfg['PRF'] / 2e-9)
    cfg['channel_mask']       = 2 ** (cfg['number_of_channels']) - 1
    return cfg


def adq_connect(config: Dict) -> Dict:
    """
    Initialize ADQ control unit + device with the given config.
    Returns a simple handle dict you pass to other functions.
    """
    cfg = _calculate_derived_params(config)

    acu = pyadq.ADQControlUnit()
    acu.ADQControlUnit_EnableErrorTrace(pyadq.LOG_LEVEL_INFO, ".")

    dev_list = acu.ListDevices()
    if not dev_list:
        raise RuntimeError("No ADQ devices found")

    idx = next(
        (i for i, info in enumerate(dev_list)
         if info.ProductID in [pyadq.PID_ADQ14, pyadq.PID_ADQ7, pyadq.PID_ADQ8]),
        -1
    )
    if idx < 0:
        raise RuntimeError("No supported ADQ device found (expect ADQ14/7/8)")

    dev = acu.SetupDevice(idx)

    if not dev.ADQ_SetTriggerMode(cfg['trigger_mode']):
        raise RuntimeError("Failed to set trigger mode")

    dev.ADQ_SetInternalTriggerPeriod(cfg['trigger_period'])
    dev.ADQ_SetupTriggerOutput(0, 5, cfg['pulselength'], 0)
    dev.ADQ_SetSampleSkip(cfg['sample_skip'])
    dev.ADQ_SetPreTrigSamples(cfg['pretrigger'])
    dev.ADQ_SetTriggerDelay(cfg['trigger_delay'])
    dev.ADQ_MultiRecordSetChannelMask(cfg['channel_mask'])
    dev.ADQ_MultiRecordSetup(1, cfg['samples_per_record'])

    print("ADQ connected and configured.")
    return {"acu": acu, "dev": dev, "config": cfg}


def adq_set_trigger_output(handle: Dict, level_v: int = 5, pulse_len_ns: Optional[int] = None, invert: int = 0):
    """Enable/shape the front-panel trigger output (TTL)."""
    dev = handle["dev"]
    cfg = handle["config"]
    if pulse_len_ns is None:
        pulse_len_ns = cfg['pulselength']
    dev.ADQ_SetupTriggerOutput(0, level_v, pulse_len_ns, invert)


def adq_arm(handle: Dict):
    """Arm the device for acquisition."""
    dev = handle["dev"]
    dev.ADQ_DisarmTrigger()
    dev.ADQ_ArmTrigger()


def adq_disarm(handle: Dict):
    """Disarm the device."""
    dev = handle["dev"]
    dev.ADQ_DisarmTrigger()


# =============================================================
#  NEW: Pre-allocated buffer system for sweep loops
# =============================================================

def adq_create_buffers(handle: Dict) -> Dict:
    """
    Allocate ctypes + numpy buffers ONCE before the sweep loop.
    Call this once, then pass the returned dict to adq_acquire_ch0().
    
    This avoids re-allocating ~1 GB of buffers every step.
    """
    cfg = handle["config"]
    N = cfg['samples_per_record']
    n_ch = cfg['number_of_channels']

    # ctypes buffers (required by hardware, both channels always needed)
    ct_buffers = (ct.POINTER(ct.c_int16 * N) * n_ch)()
    for i in range(n_ch):
        ct_buffers[i] = ct.pointer((ct.c_int16 * N)())

    header = (pyadq.structs._ADQRecordHeader * 1)()
    ct_buffers_vp = ct.cast(ct_buffers, ct.POINTER(ct.c_void_p))

    # Pre-allocated float32 output for ch0 (avoids intermediate arrays)
    ch0_out = np.empty(N, dtype=np.float32)

    print(f"ADQ buffers pre-allocated: {N} samples, ~{N * 2 * n_ch / 1e9:.2f} GB ctypes + {N * 4 / 1e9:.2f} GB float32")

    return {
        "ct_buffers": ct_buffers,
        "ct_buffers_vp": ct_buffers_vp,
        "header": header,
        "ch0_out": ch0_out,
        "N": N,
    }


def adq_acquire_ch0(handle: Dict, bufs: Dict) -> np.ndarray:
    """
    Arm, acquire, and return ch0 as float32 (mV) using pre-allocated buffers.
    ch1 data goes into ctypes buffer (required by hardware) but is NOT converted.
    
    Returns: bufs["ch0_out"] — the SAME pre-allocated array, overwritten each call.
    If you need to keep a copy, do .copy() on the result.
    """
    dev = handle["dev"]
    cfg = handle["config"]

    # Arm
    dev.ADQ_DisarmTrigger()
    dev.ADQ_ArmTrigger()

    N = bufs["N"]

    # Poll for data
    start = time.time()
    while True:
        if dev.ADQ_GetAcquiredRecords() > 0:
            ok = dev.ADQ_GetDataWHTS(
                bufs["ct_buffers_vp"],
                ct.cast(bufs["header"], ct.c_void_p),
                None,
                N,
                2,      # 2 bytes/sample (int16)
                0, 1,   # from record 0, 1 record
                cfg['channel_mask'],
                0,
                N,
                0x00,
            )
            if not ok:
                raise RuntimeError("Data acquisition failed")
            break

        if (time.time() - start) > cfg['timeout_seconds']:
            raise TimeoutError("Data acquisition timeout")

    # Convert ch0 only, in-place into pre-allocated buffer
    # np.frombuffer gives a view (no copy), then multiply into ch0_out (no intermediate)
    int16_view = np.frombuffer(bufs["ct_buffers"][0].contents, dtype=np.int16, count=N)
    np.multiply(int16_view, np.float32(8e-3), out=bufs["ch0_out"], casting='unsafe')

    return bufs["ch0_out"]


# =============================================================
#  Original functions (kept for backward compatibility)
# =============================================================

def adq_acquire_data(handle: Dict):
    """
    Acquire both channels (no pre-allocated buffers).
    Used by measure_pump_pulse which needs ch1.
    """
    dev = handle["dev"]
    cfg = handle["config"]

    N = cfg['samples_per_record']
    n_ch = cfg['number_of_channels']

    target_buffers = (ct.POINTER(ct.c_int16 * N) * n_ch)()
    for i in range(n_ch):
        target_buffers[i] = ct.pointer((ct.c_int16 * N)())

    header_list = (pyadq.structs._ADQRecordHeader * 1)()
    target_buffers_vp = ct.cast(target_buffers, ct.POINTER(ct.c_void_p))

    start = time.time()
    while True:
        if dev.ADQ_GetAcquiredRecords() > 0:
            ok = dev.ADQ_GetDataWHTS(
                target_buffers_vp,
                ct.cast(header_list, ct.c_void_p),
                None,
                N,
                2,
                0, 1,
                cfg['channel_mask'],
                0,
                N,
                0x00,
            )
            if not ok:
                raise RuntimeError("Data acquisition failed")
            break

        if (time.time() - start) > cfg['timeout_seconds']:
            raise TimeoutError("Data acquisition timeout")

    data0 = np.frombuffer(target_buffers[0].contents, dtype=np.int16, count=N).astype(np.float32) * 8e-3
    data1 = np.frombuffer(target_buffers[1].contents, dtype=np.int16, count=N).astype(np.float32) * 8e-3
    return data0, data1


def adq_acquire_once(handle: Dict):
    """Original acquire function. Kept for backward compatibility."""
    dev = handle["dev"]
    cfg = handle["config"]

    dev.ADQ_DisarmTrigger()
    dev.ADQ_ArmTrigger()

    N = cfg['samples_per_record']
    n_ch = cfg['number_of_channels']

    target_buffers = (ct.POINTER(ct.c_int16 * N) * n_ch)()
    for i in range(n_ch):
        target_buffers[i] = ct.pointer((ct.c_int16 * N)())

    header_list = (pyadq.structs._ADQRecordHeader * 1)()
    target_buffers_vp = ct.cast(target_buffers, ct.POINTER(ct.c_void_p))

    start = time.time()
    while True:
        if dev.ADQ_GetAcquiredRecords() > 0:
            ok = dev.ADQ_GetDataWHTS(
                target_buffers_vp,
                ct.cast(header_list, ct.c_void_p),
                None,
                N,
                2,
                0, 1,
                cfg['channel_mask'],
                0,
                N,
                0x00,
            )
            if not ok:
                raise RuntimeError("Data acquisition failed")
            break

        if (time.time() - start) > cfg['timeout_seconds']:
            raise TimeoutError("Data acquisition timeout")

    data0_np = np.frombuffer(target_buffers[0].contents, dtype=np.int16, count=N)
    data0_cp = data0_np * 8e-3
    data1_np = np.frombuffer(target_buffers[1].contents, dtype=np.int16, count=N)
    data1_cp = data1_np * 8e-3
    return data0_cp, data1_cp


def adq_close(handle: Dict):
    """Tear down multi-record and disable trigger output."""
    dev = handle.get("dev")
    if dev is not None:
        cfg = handle["config"]
        dev.ADQ_SetupTriggerOutput(0, 0, cfg['pulselength'], 0)
        dev.ADQ_MultiRecordClose()
