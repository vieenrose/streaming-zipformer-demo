#!/usr/bin/env python3

"""
Audio input device enumeration utilities.

Provides functions to query and list available audio input devices
on the system using sounddevice.
"""

from typing import List, Dict, Optional
import sounddevice as sd


class AudioDeviceInfo:
    """Container for audio device information."""

    def __init__(self, device_id: int, device_data: Dict):
        """
        Initialize device info from sounddevice query result.

        Args:
            device_id: Device index
            device_data: Device info dict from sd.query_devices()
        """
        self.device_id = device_id
        self.name = device_data.get("name", "Unknown")
        self.channels = device_data.get("max_input_channels", 0)
        self.sample_rate = device_data.get("default_samplerate", 16000)
        self.api = device_data.get("hostapi", -1)
        self.device_data = device_data

    def __repr__(self) -> str:
        return (
            f"AudioDeviceInfo(id={self.device_id}, name={self.name}, "
            f"channels={self.channels}, rate={self.sample_rate})"
        )

    def __str__(self) -> str:
        return (
            f"[{self.device_id}] {self.name}\n"
            f"    Channels: {self.channels}, Sample Rate: {self.sample_rate} Hz"
        )


def enumerate_input_devices() -> List[AudioDeviceInfo]:
    """
    Enumerate all available audio input devices.

    Returns:
        List of AudioDeviceInfo for all input devices with at least 1 channel
    """
    try:
        all_devices = sd.query_devices()
    except Exception as e:
        print(f"Error querying devices: {e}")
        return []

    input_devices = []

    # sd.query_devices() returns a DeviceList which is iterable
    for device_data in all_devices:
        device_id = device_data.get("index", -1)

        # Filter for input devices (have input channels)
        if device_data.get("max_input_channels", 0) > 0:
            input_devices.append(AudioDeviceInfo(device_id, device_data))

    return input_devices


def get_device_by_id(device_id: int) -> Optional[AudioDeviceInfo]:
    """
    Get device info by device ID.

    Args:
        device_id: Device index

    Returns:
        AudioDeviceInfo if found, None otherwise
    """
    try:
        device_data = sd.query_devices(device_id)
        if device_data.get("max_input_channels", 0) > 0:
            return AudioDeviceInfo(device_id, device_data)
    except Exception:
        pass

    return None


def get_default_input_device() -> Optional[AudioDeviceInfo]:
    """
    Get the default input device.

    Returns:
        AudioDeviceInfo for default device, None if not found
    """
    try:
        default_id = sd.default.device[0]
        return get_device_by_id(default_id)
    except Exception:
        return None


def print_devices():
    """Print all available input devices in a formatted table."""
    devices = enumerate_input_devices()

    if not devices:
        print("No audio input devices found!")
        return

    print("\nAvailable Audio Input Devices:")
    print("=" * 70)

    for device in devices:
        print(device)

    print("=" * 70)

    default = get_default_input_device()
    if default:
        print(f"\nDefault Device: [{default.device_id}] {default.name}")
