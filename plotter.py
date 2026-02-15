import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
from datetime import datetime
from logger_utils import log_message
import h5py
def plot_traces_3d_dist(traces,freqs ,distance, pulse_width):

    fiber_length_actual = 50000
    #sampling_rate = 50e6
    # sampling_period = 1/sampling_rate  # 20ns
    # spatial_resl = pulse_width/10
    # samples_per_pulsewidth = pulse_width/sampling_period  #100ns/20ns = 5
   
    n_steps, n_samples = traces.shape
    mtr_conv_factor = n_samples/fiber_length_actual
    print('Traces shape',n_samples) #
    # Distance axis should have 1000 points for every 10m of 10000m
    distance_m = np.arange(n_samples)#*10#/fiber_length_actual
    
    # Meshgrid for plotting
    D, F = np.meshgrid(distance_m, freqs)

    # 3D Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(D, F, traces, cmap="viridis", linewidth=0, antialiased=True)

    ax.set_xlabel("Distance Samples")
    ax.set_ylabel("Frequency (GHz)")
    ax.set_zlabel("Amplitude")
    ax.set_title("3D Traces vs Frequency")

    fig.colorbar(surf, shrink=0.5, aspect=10, label="Amplitude")
    plt.show()


# def save_traces(traces, freqs, t_us, save_dir, prefix="processed", log_file=None):
#     # Folder named by today's date (YYYY-MM-DD)
#     today_str = datetime.today().strftime("%Y-%m-%d")
#     day_folder = os.path.join(save_dir, today_str)

#     # Create folder if not exists
#     if not os.path.exists(day_folder):
#         os.makedirs(day_folder)
#         print(f"Created folder: {day_folder}")
#     else:
#         print(f"Using existing folder: {day_folder}")

#     # Timestamp for filename
#     timestamp = datetime.now().strftime("%H%M%S")

#     # Build DataFrame:
#     # rows = freqs, columns = time values
#     df = pd.DataFrame(traces, index=freqs, columns=np.round(t_us, 5))
#     df.index.name = "Frequency"

#     # Filepath
#     filename = f"{prefix}_{timestamp}.csv"
#     filepath = os.path.join(day_folder, filename)

#     # Save to CSV
#     df.to_csv(filepath)

#     print(f"Saved matrix CSV: {filepath}")
    
#     # Log the file saving event if log_file is provided
#     if log_file:
#         log_message(f"Saved trace data to: {filepath}", log_file=log_file)


def save_single_raw_trace_h5(raw_trace, freq_ghz, index, cfg, session_dir):
    """
    Save a single raw trace to H5 file. Called inside the sweep loop
    so the raw trace can be discarded from memory right after saving.
    """
    freq_str = f"{freq_ghz:.6f}".replace('.', 'p')
    filepath = os.path.join(session_dir, f"freq_{freq_str}_GHz.h5")

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('raw_trace', data=raw_trace, compression='gzip', compression_opts=4)
        f.attrs['frequency_ghz'] = freq_ghz
        f.attrs['index'] = index
        f.attrs['trace_length'] = len(raw_trace)
        f.attrs['sample_rate'] = cfg.get('sample_rate', 0)
        f.attrs['prf'] = cfg.get('PRF', 0)
        f.attrs['pulse_width_ns'] = cfg.get('pulselength', 0)


def save_raw_traces_h5(raw_traces, freqs, cfg, save_dir, tag="raw"):
    """
    Save raw traces using h5py format with individual files per frequency.
    Creates a session folder and saves each frequency's data separately.
    
    Args:
        raw_traces: List of numpy arrays, one per frequency
        freqs: Array of frequencies (GHz)
        cfg: ADQ configuration dictionary
        save_dir: Base directory for saving
        tag: Tag for the session folder name
    
    Returns:
        session_folder: Path to the created session folder
    """
    # Create timestamp-based session folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{tag}_sweep_{timestamp}"
    session_folder = os.path.join(save_dir, session_name)
    os.makedirs(session_folder, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Saving raw traces to session folder:")
    print(f"{session_folder}")
    print(f"{'='*60}")
    
    # Save metadata file for the entire session
    metadata_file = os.path.join(session_folder, "session_metadata.h5")
    with h5py.File(metadata_file, 'w') as f:
        # Store session-level metadata
        f.attrs['timestamp'] = timestamp
        f.attrs['n_frequencies'] = len(freqs)
        f.attrs['n_traces'] = len(raw_traces)
        
        # Store configuration parameters
        cfg_group = f.create_group('config')
        for key, value in cfg.items():
            try:
                cfg_group.attrs[key] = value
            except (TypeError, ValueError):
                cfg_group.attrs[key] = str(value)
        
        # Store frequency array
        f.create_dataset('frequencies_ghz', data=np.array(freqs, dtype=np.float64))
    
    # Save each frequency's data in a separate file
    saved_files = []
    for idx, (freq, raw_trace) in enumerate(zip(freqs, raw_traces)):
        # Create filename with frequency value
        freq_str = f"{freq:.6f}".replace('.', 'p')  # Replace . with p for filename
        filename = f"freq_{freq_str}_GHz.h5"
        filepath = os.path.join(session_folder, filename)
        
        # Save trace data
        with h5py.File(filepath, 'w') as f:
            # Store the raw trace data
            f.create_dataset('raw_trace', data=raw_trace, compression='gzip', compression_opts=4)
            
            # Store metadata as attributes
            f.attrs['frequency_ghz'] = freq
            f.attrs['index'] = idx
            f.attrs['timestamp'] = timestamp
            f.attrs['trace_length'] = len(raw_trace)
            
            # Add some key configuration parameters
            f.attrs['sample_rate'] = cfg.get('sample_rate', 0)
            f.attrs['prf'] = cfg.get('PRF', 0)
            f.attrs['pulse_width_ns'] = cfg.get('pulselength', 0)
        
        saved_files.append(filename)
        
        # Progress indicator
        if (idx + 1) % 10 == 0 or (idx + 1) == len(freqs):
            print(f"Saved {idx + 1}/{len(freqs)} frequency files...", end='\r')
    
    print(f"\n✅ Successfully saved {len(saved_files)} raw trace files")
    print(f"{'='*60}\n")
    
    # Create a summary text file
    summary_file = os.path.join(session_folder, "README.txt")
    with open(summary_file, 'w') as f:
        f.write(f"BOTDA Raw Traces - Session {timestamp}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Number of frequencies: {len(freqs)}\n")
        f.write(f"Frequency range: {freqs[0]:.6f} - {freqs[-1]:.6f} GHz\n")
        f.write(f"Sample rate: {cfg.get('sample_rate', 0)/1e6:.2f} MS/s\n")
        f.write(f"PRF: {cfg.get('PRF', 0)} Hz\n")
        f.write(f"Pulse width: {cfg.get('pulselength', 0)} ns\n")
        f.write(f"Averages: {cfg.get('averages', 'N/A')}\n")
        f.write(f"\nFiles saved:\n")
        f.write(f"  - session_metadata.h5 (contains all frequencies and session config)\n")
        f.write(f"  - freq_*.h5 ({len(saved_files)} files, one per frequency)\n")
    
    return session_folder


def save_traces(traces, freqs, t_us, save_dir, prefix="processed_trace", log_file=None):
    
    """Saves processed traces directly into the session folder."""
    timestamp = datetime.now().strftime("%H%M%S")
    filepath = os.path.join(save_dir, f"{prefix}_{timestamp}.h5")

    with h5py.File(filepath, 'w') as f:
        # Create datasets
        f.create_dataset('traces', data=traces, compression='gzip', compression_opts=4)
        f.create_dataset('frequencies_ghz', data=freqs)
        f.create_dataset('time_us', data=t_us)
        
        print(f"✅ Processed data saved to: {filepath}")
    if log_file:
        log_message(f"Saved trace data to: {filepath}", log_file=log_file)
    
    return filepath

def save_dc_offsets_h5(freqs, dc_offsets, base_log_path):
    """Saves DC offsets to an HDF5 file instead of CSV."""
    # Derive filename: replace .log or .csv with .h5
    dc_file = os.path.splitext(base_log_path)[0] + "_dc_offset.h5"
    
    with h5py.File(dc_file, 'w') as f:
        f.create_dataset('frequency_ghz', data=np.array(freqs, dtype=np.float64))
        f.create_dataset('dc_offset', data=np.array(dc_offsets, dtype=np.float64))
        f.attrs['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"✅ DC offsets saved to H5: {dc_file}")
    return dc_file



def save_pump_data_h5(pump_results, adq_config, session_dir):
    """Saves the first 3 pump pulse traces to an HDF5 file."""
 
    filepath = os.path.join(session_dir, "pump_pulse.h5")
    # Unpack results
    peak_mv, t_peak, p_pd_w, p_inc_w, p_inc_dbm, full_trace = pump_results
    
    # Slice for first 3 periods
    fs = adq_config["sample_rate"]
    samples_per_period = int(round(fs / adq_config["PRF"]))
    short_trace = full_trace[:samples_per_period * 3]

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('pump_trace', data=short_trace, compression='gzip')
        f.attrs['sample_rate'] = fs
        f.attrs['optical_power_dbm'] = p_inc_dbm
        f.attrs['timestamp'] = datetime.now().strftime("%H:%M:%S")

    print(f"✅ Pump pulse saved to: {filepath}")



# def save_pump_data_h5(pump_results, save_dir):
#     """
#     Saves pump pulse measurement results to an HDF5 file.
#     pump_results is the tuple returned by measure_pump_pulse.
#     """
#     today_str = datetime.today().strftime("%Y-%m-%d")
#     day_folder = os.path.join(save_dir, today_str)
#     os.makedirs(day_folder, exist_ok=True)

#     timestamp = datetime.now().strftime("%H%M%S")
#     filepath = os.path.join(day_folder, f"pump_pulse_{timestamp}.h5")

#     # Unpack the tuple from measure_pump_pulse
#     peak_mv, t_peak, p_pd_w, p_inc_w, p_inc_dbm, trace = pump_results

#     with h5py.File(filepath, 'w') as f:
#         f.create_dataset('pump_trace', data=trace, compression='gzip')
#         # Store calculated metrics as attributes
#         f.attrs['peak_mv'] = peak_mv
#         f.attrs['peak_time_us'] = t_peak
#         f.attrs['optical_power_mw'] = p_inc_w * 1000
#         f.attrs['optical_power_dbm'] = p_inc_dbm
#         f.attrs['timestamp'] = timestamp

#     print(f"✅ Pump pulse data saved to: {filepath}")
#     return filepath


# def save_raw_traces(raw_traces, freqs, cfg, save_dir, tag="raw"):
#     """Legacy npz-based saving (kept for backward compatibility)."""
#     os.makedirs(save_dir, exist_ok=True)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     fname = f"{tag}_traces_{timestamp}.npz"
#     fpath = os.path.join(save_dir, fname)

#     np.savez_compressed(
#         fpath,
#         raw_traces=raw_traces,
#         freqs=freqs,
#         cfg=cfg
#     )

#     print(f"✅ Raw traces saved to:\n{fpath}")
#     return fpath