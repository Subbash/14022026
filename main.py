# sweep_rf_capture_once.py
import time
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from datetime import datetime

from rf_controller import connect_rf_controller, set_rf, set_rf_freq_only, enable_rf, get_status
from adq_controller import adq_connect, adq_set_trigger_output, adq_create_buffers, adq_acquire_ch0
from processing import (
    measure_pump_pulse,
    process_trace,
    compute_processed_length,
)
from plotter import (
    save_traces, plot_traces_3d_dist, save_raw_traces_h5,
    save_pump_data_h5, save_dc_offsets_h5, save_single_raw_trace_h5,
)
from logger_utils import log_sweep_session

# ========== SAVE CONFIGURATION ==========
SAVE_CONFIG = {
    'save_raw': False,
    'save_processed': True,
    'save_pump': True,
    #'raw_save_dir': r"F:\01_02_2026_data\saved_raw",
    'processed_save_dir': r"E:\15_02_2026_data\20ns\4096"
}
# ========== END SAVE CONFIGURATION ==========

# ========== COMMENTS ==========
PERSONAL_COMMENTS = """

DSB Trial
100ns pulse width trial

"""
# ========== END COMMENTS ==========

fiber_len = 50e3

rf_params = dict(
    start_freq=10.70,
    stop_freq=10.90,
    step_mhz=1,
    sweep_power_dbm=-8.0,
    com_port="COM19"
)


# ==========================================================
#  Chunked ADQ config
#  
#  The hardware has a DMA limit on how many samples it can
#  transfer in one record. max_chunk controls the max periods
#  per single acquisition. For large averages (e.g., 2048),
#  the acquisition repeats in chunks and results are averaged.
#
#  Example: averages=2048, max_chunk=512
#   -> hardware acquires 512 periods per call
#   -> 4 chunks acquired per frequency step
#   -> results averaged together = equivalent to 2048 averages
# ==========================================================
def compute_adq_config(fiber_len_actual_m, averages=4096, max_chunk=512, sample_rate=500e6):
   
    valid_window_us = fiber_len_actual_m / 1e8
    tail_us = 0.1 * valid_window_us
    prp_us = valid_window_us + tail_us
    prp_s = prp_us
    prf = 1 / prp_s
    sample_period_s = 1 / sample_rate

    # Chunk calculation
    chunk_averages = min(averages, max_chunk)
    n_chunks = averages // chunk_averages
    
    # Hardware record size based on chunk (not total averages)
    samples_per_record = int(prp_s * chunk_averages / sample_period_s)

    adq_config = dict(
        sample_rate=sample_rate,
        sample_length=samples_per_record,
        PRF=prf,
        fiblen_actual=fiber_len_actual_m,
        number_of_channels=2,
        pretrigger=0,
        trigger_delay=0,
        pulselength=20,
        timeout_seconds=2,
        valid_window_us=valid_window_us,
        tail_window_us=tail_us,
        prp_us=prp_us,
        # Averaging info
        averages=averages,              # total desired averages
        chunk_averages=chunk_averages,  # periods per hardware acquisition
        n_chunks=n_chunks,              # number of acquisitions to reach total
    )
    
    return adq_config

adq_config = compute_adq_config(fiber_len)

for k, v in adq_config.items():
    print(f"{k}: {v}")


# ==========================================================
#  Sweep + process with chunked averaging
# ==========================================================
def sweep_and_process(
    rf_params,
    adq_config,
    dwell_time,
    arm_before_each,
    inter_capture_delay,
    settle_extra,
    save_raw=False,
    raw_save_dir=None,
):
    start_freq = rf_params["start_freq"]
    stop_freq = rf_params["stop_freq"]
    step_mhz = rf_params["step_mhz"]
    sweep_power_dbm = rf_params["sweep_power_dbm"]
    com_port = rf_params["com_port"]
    valid_window_us = adq_config["valid_window_us"]
    n_chunks = adq_config["n_chunks"]

    # --- RF Setup ---
    connect_rf_controller(com_port)
    rf_status = get_status(channel=0)

    if rf_status and rf_status["enabled"]:
        current_freq = rf_status["frequency_ghz"]
        print(f"RF already enabled at {current_freq:.6f} GHz")
        if abs(current_freq - start_freq) < 1e-6:
            print("Current frequency matches start frequency. Continuing...")
        else:
            print(f"Frequency differs ({current_freq:.6f} -> {start_freq:.6f} GHz). Updating...")
            time.sleep(max(dwell_time, 0))
            set_rf(channel=0, freq_ghz=start_freq, power_dbm=sweep_power_dbm)
    else:
        print("RF not enabled. Initializing...")
        set_rf(channel=0, freq_ghz=start_freq, power_dbm=sweep_power_dbm)
        enable_rf(channel=0, enable=True)
        print("RF initialized and enabled.")
        time.sleep(max(dwell_time, 0))
        if settle_extra > 0:
            time.sleep(settle_extra)

    # --- ADQ Setup ---
    handle = adq_connect(adq_config or {})
    adq_set_trigger_output(handle, level_v=5)
    cfg = handle["config"]

    # --- Allocate hardware buffers ONCE ---
    bufs = adq_create_buffers(handle)

    # --- Step math ---
    step_ghz = float(step_mhz) * 1e-3
    n_steps = int(((stop_freq - start_freq) / step_ghz) + 1)

    # --- Pre-allocate output arrays ---
    proc_len = compute_processed_length(cfg, valid_window_us)
    processed_traces = np.empty((n_steps, proc_len), dtype=np.float32)
    freqs = np.linspace(start_freq, stop_freq, n_steps)
    meta_list = []
    dc_offsets = []

    # --- Accumulator for chunked averaging (reused each frequency step) ---
    accum = np.zeros(proc_len, dtype=np.float64)

    # --- Optional: prepare raw save folder ---
    raw_session_dir = None
    if save_raw and raw_save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_session_dir = os.path.join(raw_save_dir, f"raw_sweep_{timestamp}")
        os.makedirs(raw_session_dir, exist_ok=True)

    print(f"\nStarting sweep: {n_steps} freq steps x {n_chunks} chunks = {n_steps * n_chunks} total acquisitions")

    try:
        for i in range(n_steps):
            freq = freqs[i]
            set_rf_freq_only(channel=0, freq_ghz=float(freq))

            # Reset accumulator for this frequency
            accum[:] = 0.0
            dc_accum = 0.0
            last_meta = None

            # --- Acquire n_chunks and average ---
            for c in range(n_chunks):
                raw_trace = adq_acquire_ch0(handle, bufs)

                # Save raw if enabled (first chunk only to avoid huge files)
                if save_raw and raw_session_dir and c == 0:
                    save_single_raw_trace_h5(raw_trace.copy(), freq, i, cfg, raw_session_dir)

                # Process this chunk
                proc, meta = process_trace(raw_trace, cfg, valid_window_us, apply_dc=True)

                # Accumulate
                accum += proc
                dc_accum += meta["dc_offset"] if meta["dc_offset"] is not None else 0.0
                last_meta = meta

            # --- Average across chunks ---
            processed_traces[i, :] = (accum / n_chunks).astype(np.float32)
            meta_list.append(last_meta)
            dc_offsets.append(dc_accum / n_chunks)

            # Print progress every 10 steps
            if (i + 1) % 10 == 0 or (i + 1) == n_steps:
                print(
                    f"\r[{i+1}/{n_steps}] RF -> {freq:.6f} GHz | {n_chunks} chunks done",
                    end="", flush=True
                )

            # Garbage collection every 20 steps
            if (i + 1) % 20 == 0:
                gc.collect()

        print()
        return processed_traces, freqs, meta_list, dc_offsets, cfg

    finally:
        try: set_rf(channel=0, freq_ghz=start_freq, power_dbm=sweep_power_dbm) 
        except: pass
        try: enable_rf(channel=0, enable=True)
        except: pass


# ==========================================================
#  MAIN
# ==========================================================
if __name__ == "__main__":
    # --- 1. Setup Session Directory ---
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(SAVE_CONFIG['processed_save_dir'], f"run_{timestamp_str}")
    os.makedirs(session_dir, exist_ok=True)
    print(f"Session Folder created: {session_dir}")

    adq_config = compute_adq_config(fiber_len)

    # --- 2. Measure & Save Pump pulse ---
    if SAVE_CONFIG.get('save_pump', False):
        pump_data = measure_pump_pulse(
            adq_config,
            attenuation_db=40.0,
            pd_gain_v_per_w=2250.0
        )
        save_pump_data_h5(pump_data, adq_config, session_dir)
    else:
        print("Skipping pump pulse measurement.")

    gc.collect()

    # --- 3. Acquire + Process ---
    start_time = time.time()

    processed_traces, freqs, meta_list, dc_offsets, cfg = sweep_and_process(
        rf_params,
        adq_config,
        dwell_time=10.0,
        arm_before_each=True,
        inter_capture_delay=0.0,
        settle_extra=0.0,
        save_raw=SAVE_CONFIG['save_raw'],
        raw_save_dir=SAVE_CONFIG.get('raw_save_dir'),
    )

    total_elapsed = time.time() - start_time
    print(f"Sweep + Processing completed in {total_elapsed:.3f} s")

    # --- 4. Plotting ---
    fs = cfg["sample_rate"]
    t_us = np.arange(processed_traces.shape[1]) / fs * 1e6
    
    plot_traces_3d_dist(processed_traces, freqs, adq_config["fiblen_actual"], adq_config["pulselength"])

    # --- 5. Save processed data ---
    if SAVE_CONFIG['save_processed']:
        log_file = log_sweep_session(
            rf_params=rf_params,
            adq_config=adq_config,
            traces=processed_traces,
            freqs=freqs,
            elapsed_time=total_elapsed,
            meta_info=meta_list[-1],
            session_dir=session_dir,
            personal_comments=PERSONAL_COMMENTS
        )

        save_traces(processed_traces, freqs, t_us, save_dir=session_dir, log_file=log_file)
        save_dc_offsets_h5(freqs, dc_offsets, log_file)
        
        try:
            print("\n--- Add comments for this sweep session (ENTER twice to finish) ---")
            user_comments = []
            while True:
                line = input()
                if line.strip() == "": break
                user_comments.append(line)
            
            if user_comments:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write("\n\n# ===== USER COMMENTS =====\n" + "\n".join(user_comments) + "\n")
                print(f"Comments added to: {log_file}")
        except Exception as e:
            print(f"Could not append comments: {e}")
    else:
        print("Processed data saving is disabled.")
