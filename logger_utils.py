import os
from datetime import datetime

def get_log_filepath(session_dir, prefix="Log"):
    """Generates log filepath directly inside the provided session directory."""
    # Timestamp for filename
    timestamp = datetime.now().strftime("%H%M%S")


    # Build log filename
    log_filename = f"{prefix}_{timestamp}.txt"
    return os.path.join(session_dir, log_filename)

def log_message(msg, mode="a", log_file=None):
    """Write a message to the log file."""
    if log_file:
        with open(log_file, mode) as f:
            f.write(msg + "\n")

def log_sweep_session(rf_params, adq_config, traces, freqs, elapsed_time, meta_info, session_dir, prefix="processed", personal_comments=""):
    """
    Logs sweep parameters directly into the session_dir.
    """
    # Pass session_dir directly to get the path
    log_file = get_log_filepath(session_dir, prefix)
    
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    header = f"=== New Sweep Session started at {timestamp} ==="
    log_message(header, mode="w", log_file=log_file)

    # Personal Comments section (if provided)
    if personal_comments.strip():
        comments_header = "\n=== PERSONAL COMMENTS & NOTES ==="
        log_message(comments_header, log_file=log_file)
        log_message(personal_comments.strip(), log_file=log_file)
        log_message("=" * 40, log_file=log_file)

    # RF Parameters section
    rf_text = (f"\nRF Sweep Parameters:\n"
               f"Start frequency   : {rf_params['start_freq']:.6f} GHz\n"
               f"Stop frequency    : {rf_params['stop_freq']:.6f} GHz\n"
               f"Step size         : {rf_params['step_mhz']} MHz\n"
               f"RF Power          : {rf_params['sweep_power_dbm']:.1f} dBm\n"
               f"COM Port          : {rf_params['com_port']}\n")

    # ADQ Configuration section
    adq_text = (f"ADQ Configuration:\n"
                f"Sample rate       : {adq_config['sample_rate']/1e6:.2f} MS/s\n"
                f"Sampling period   : {1/adq_config['sample_rate']*1e9:.2f} ns\n"
                f"Sample length     : {adq_config['sample_length']:.0f} samples\n"
                f"PRF               : {adq_config['PRF']} Hz\n"
                f"Pulse Width       : {adq_config['pulselength']} ns\n"
                f"Averages          : {adq_config.get('averages', 'N/A')}\n"
                f"Fiber length      : {adq_config['fiblen_actual']/1000:.2f} km\n")

    # Calculate summary metrics
    fs = adq_config['sample_rate']
    prf = adq_config['PRF']

    Ts = 1 / fs
    t_trace = meta_info['raw_length'] * Ts
    period_time = 1 / prf
    valid_fiber_time = meta_info['valid_window_us']
    tail_time = period_time - valid_fiber_time
    tail_samples = tail_time / Ts

    # Summary section
    summary = (f"\n===== Sweep Summary =====\n"
               f"Time Taken to Acquire 1 Trace : {t_trace*1e3:.3f} ms\n"
               f"Period time           : {period_time*1e6:.3f} µs\n"
               f"Valid fiber time      : {valid_fiber_time*1e6:.3f} µs\n"
               f"Tail time             : {tail_time*1e6:.3f} µs\n"
               f"Samples Acquired      : {meta_info['raw_length']}\n" 
               f"Periods               : {meta_info['n_periods']}\n"
               f"Samples/period        : {meta_info['samples_per_period']}\n"
               f"Valid fiber samples   : {meta_info['valid_samples']}\n"
               f"Tail samples          : {tail_samples}\n"
               f"Reshaping(Periods,Valid_samples/Period)  : {meta_info['periods_window_shape']}\n"
               f"Final Processed trace : {meta_info['processed_shape']}\n"
               f"Acquisition time (sweep): {elapsed_time:.3f} s\n"
               f"Trace shape           : {traces.shape}\n"
               "==========================\n")

    # Write all sections to the log file
    log_message(rf_text, log_file=log_file)
    log_message(adq_text, log_file=log_file)
    log_message(summary, log_file=log_file)
    
    print(f"Created log file: {log_file}")
    return log_file