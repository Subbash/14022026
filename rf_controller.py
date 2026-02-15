# import time
# import serial
# from windfreak import SynthHD

# # Store the SynthHD object globally
# rf_device = None  

# def connect_rf_controller(com_port, retries=5, delay=1.0):
#     """Try to connect to RF controller with retries and return the device handle."""
#     global rf_device
    
#     for attempt in range(1, retries+1):
#         try:
#             print(f"[Attempt {attempt}] Connecting to RF controller on {com_port}...")
            
#             # Create object
#             rf_device = SynthHD(com_port)

#             # Optional: flush buffers in case of leftover junk
#             # if hasattr(rf_device, "ser") and isinstance(rf_device.ser, serial.Serial):
#             #     rf_device.ser.reset_input_buffer()
#             #     rf_device.ser.reset_output_buffer()

#             # Initialize
#             #rf_device.init()
#             # Force a query to check if it is alive
#             # _ = rf_device.hardware_version  
            
#             print(f"✅ Connected successfully on {com_port}")
#             return rf_device
        
#         except (serial.SerialException, UnicodeDecodeError, OSError) as e:
#             print(f"❌ Failed: {e}")
#             time.sleep(delay)
    
#     raise RuntimeError(f"Could not connect to RF Synth on {com_port} after {retries} attempts")


# def set_rf(channel, freq_ghz, power_dbm):
#     """Set frequency and power on given channel."""
#     if rf_device is None:
#         raise RuntimeError("RF controller not connected. Call connect_rf_controller first.")
#     rf_device[channel].frequency = freq_ghz * 1e9
#     rf_device[channel].power = power_dbm
#     # print(f"Channel {channel}: {freq_ghz} GHz, {power_dbm} dBm")


# def enable_rf(channel=0, enable=True):
#     """Enable or disable RF output on given channel."""
#     if rf_device is None:
#         raise RuntimeError("RF controller not connected. Call connect_rf_controller first.")
#     rf_device[channel].enable = enable
#     state = "ENABLED" if enable else "DISABLED"
#     print(f"RF output {state} on channel {channel}")


# def get_status(channel=0):
#     """
#     Return detailed RF channel status.

#     Returns:
#         dict: {
#             "enabled": bool,
#             "frequency_ghz": float,
#             "power_dbm": float
#         }
#     If the controller is not connected or an error occurs, returns None.
#     """
#     if rf_device is None:
#         raise RuntimeError("RF controller not connected. Call connect_rf_controller first.")
    
#     try:
#         freq_hz = rf_device[channel].frequency
#         power_dbm = rf_device[channel].power
#         enabled = rf_device[channel].enable

#         print("===== RF STATUS =====")
#         print(f" RF Output : {'ON' if enabled else 'OFF'}")
#         print(f" Frequency : {freq_hz / 1e9:.6f} GHz")
#         print(f" Power     : {power_dbm:.2f} dBm")
#         print("======================")

#         return {
#             "enabled": enabled,
#             "frequency_ghz": freq_hz / 1e9,
#             "power_dbm": power_dbm,
#         }

#     except Exception as e:
#         print(f"⚠️ Could not read channel {channel} status: {e}")
#         return None

# def get_reference(self):
#     """
#     Get current reference source and frequency.
#     Returns:
#         dict: {"mode": str, "frequency_hz": float}
#     """
#     mode = self.reference_mode
#     freq = self.reference_frequency
#     print("===== RF REFERENCE =====")
#     print(f" Mode       : {mode}")
#     print(f" Frequency  : {freq/1e6:.3f} MHz")
#     print("========================")
#     return {"mode": mode, "frequency_hz": freq}


# def set_reference(self, mode: str, freq_hz: float = None):
#     """
#     Set both reference mode and frequency if supported.

#     Args:
#         mode (str): one of ['external', 'internal 27mhz', 'internal 10mhz']
#         freq_hz (float, optional): override frequency (in Hz)
#     """
#     if mode not in self.reference_modes:
#         raise ValueError(f"Invalid mode. Allowed: {self.reference_modes}")

#     self.reference_mode = mode
#     if freq_hz:
#         self.reference_frequency = freq_hz
#     else:
#         # choose default based on mode
#         if mode == 'internal 27mhz':
#             self.reference_frequency = 27e6
#         elif mode == 'internal 10mhz':
#             self.reference_frequency = 10e6
#         elif mode == 'external':
#             # external ref usually 10 MHz
#             self.reference_frequency = 10e6

#     print(f"✅ Reference set to {mode} ({self.reference_frequency/1e6:.3f} MHz)")

import time
import serial
from windfreak import SynthHD

# Global SynthHD object
rf_device = None


# ------------------------------------------------------
# CONNECT / DISCONNECT
# ------------------------------------------------------
def connect_rf_controller(com_port, retries=5, delay=1.0):
    """Try to connect to RF controller with retries and return the device handle."""
    global rf_device

    for attempt in range(1, retries + 1):
        try:
            print(f"[Attempt {attempt}] Connecting to RF controller on {com_port}...")

            rf_device = SynthHD(com_port)
            _ = rf_device[0].frequency  # probe connection
            print(f"✅ Connected successfully on {com_port}")
            return rf_device

        except (serial.SerialException, UnicodeDecodeError, OSError) as e:
            print(f"❌ Failed: {e}")
            time.sleep(delay)

    raise RuntimeError(f"Could not connect to RF Synth on {com_port} after {retries} attempts")


def disconnect_rf_controller():
    """Safely close connection and release COM port."""
    global rf_device
    if rf_device is not None:
        try:
            if hasattr(rf_device, "ser") and isinstance(rf_device.ser, serial.Serial):
                rf_device.ser.reset_input_buffer()
                rf_device.ser.reset_output_buffer()
                rf_device.ser.close()
                print("🔌 RF controller serial port closed.")
            rf_device = None
            time.sleep(1.0)
            print("✅ RF controller disconnected and port released.")
        except Exception as e:
            print(f"⚠️ Error while disconnecting: {e}")


# ------------------------------------------------------
# RF CONTROL
# ------------------------------------------------------
def set_rf(channel, freq_ghz, power_dbm):
    """Set frequency and power on given channel."""
    if rf_device is None:
        raise RuntimeError("RF controller not connected. Call connect_rf_controller() first.")

    rf_device[channel].frequency = freq_ghz * 1e9
    rf_device[channel].power = power_dbm


# ---- NEW: Frequency-only update (skips redundant power write) ----
def set_rf_freq_only(channel, freq_ghz):
    """Set only frequency on given channel. Use inside sweep loops
    where power doesn't change between steps."""
    if rf_device is None:
        raise RuntimeError("RF controller not connected. Call connect_rf_controller() first.")

    rf_device[channel].frequency = freq_ghz * 1e9
    #
    # 
    # 
    # print(f"Channel {channel}: Frequency = {freq_ghz:.6f} GHz, Power = {power_dbm:.2f} dBm")


def enable_rf(channel=0, enable=True):
    """Enable or disable RF output on given channel."""
    if rf_device is None:
        raise RuntimeError("RF controller not connected. Call connect_rf_controller() first.")

    rf_device[channel].enable = enable
    state = "ENABLED" if enable else "DISABLED"
    print(f"RF output {state} on channel {channel}")


# ------------------------------------------------------
# REFERENCE MANAGEMENT
# ------------------------------------------------------
def get_reference():
    """Get current reference source and frequency."""
    if rf_device is None:
        raise RuntimeError("RF controller not connected. Call connect_rf_controller() first.")

    try:
        mode = rf_device.reference_mode
        freq = rf_device.reference_frequency
        print("===== RF REFERENCE =====")
        print(f" Mode       : {mode}")
        print(f" Frequency  : {freq / 1e6:.3f} MHz")
        print("========================")
        return {"mode": mode, "frequency_hz": freq}

    except Exception as e:
        print(f"⚠️ Error reading reference: {e}")
        return {"mode": None, "frequency_hz": None}


def set_reference(mode: str, freq_hz: float = None):
    """Set both reference mode and frequency."""
    if rf_device is None:
        raise RuntimeError("RF controller not connected. Call connect_rf_controller() first.")

    try:
        if mode not in rf_device.reference_modes:
            raise ValueError(f"Invalid mode. Allowed: {rf_device.reference_modes}")

        rf_device.reference_mode = mode

        # Default or custom frequency
        if freq_hz:
            rf_device.reference_frequency = freq_hz
        else:
            if mode == "internal 27mhz":
                rf_device.reference_frequency = 27e6
            elif mode == "internal 10mhz":
                rf_device.reference_frequency = 10e6
            elif mode == "external":
                rf_device.reference_frequency = 10e6

        print(f"✅ Reference set to {mode} ({rf_device.reference_frequency / 1e6:.3f} MHz)")

    except Exception as e:
        print(f"⚠️ Error setting reference: {e}")


# ------------------------------------------------------
# STATUS (now includes REFERENCE info)
# ------------------------------------------------------
def get_status(channel=0):
    """
    Return detailed RF status including reference info.
    Returns:
        dict: {
            "enabled": bool,
            "frequency_ghz": float,
            "power_dbm": float,
            "reference_mode": str,
            "reference_freq_mhz": float
        }
    """
    if rf_device is None:
        raise RuntimeError("RF controller not connected. Call connect_rf_controller() first.")

    try:
        freq_hz = rf_device[channel].frequency
        power_dbm = rf_device[channel].power
        enabled = rf_device[channel].enable

        ref_info = get_reference()  # reuse function
        ref_mode = ref_info.get("mode")
        ref_freq = ref_info.get("frequency_hz")

        print("===== RF STATUS =====")
        print(f" RF Output : {'ON' if enabled else 'OFF'}")
        print(f" Frequency : {freq_hz / 1e9:.6f} GHz")
        print(f" Power     : {power_dbm:.2f} dBm")
        print(f" Ref Mode  : {ref_mode}")
        if ref_freq:
            print(f" Ref Freq  : {ref_freq / 1e6:.3f} MHz")
        print("======================")

        return {
            "enabled": enabled,
            "frequency_ghz": freq_hz / 1e9,
            "power_dbm": power_dbm,
            "reference_mode": ref_mode,
            "reference_freq_mhz": ref_freq / 1e6 if ref_freq else None,
        }

    except Exception as e:
        print(f"⚠️ Could not read status: {e}")
        return None


# ------------------------------------------------------
# Example usage
# ------------------------------------------------------
if __name__ == "__main__":
    com_port = input("Enter COM port (e.g., COM5): ").strip()
    connect_rf_controller(com_port)

    # Example: Show current reference and status
    get_reference()
    get_status()

    # Change reference mode
    set_reference("internal 27mhz")
    get_status()

    # Example: Basic RF control
    set_rf(0, 10.8, 0)
    enable_rf(0, True)
    get_status()
    enable_rf(0, False)

    disconnect_rf_controller()
