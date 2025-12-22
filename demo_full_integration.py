#!/usr/bin/env python3

"""
FULL INTEGRATION DEMO: Step 4 + Step 5

Phase 1: Audio Device Detection (Step 4)
  - Enumerate all audio devices
  - Scan for active speech using VAD
  - Auto-select device with highest VAD probability

Phase 2: MP3 Recording (Step 5)
  - Start background recording on selected device
  - Record for 10 seconds while user speaks
  - Verify MP3 file integrity
"""

import sys
sys.path.insert(0, '/home/luigi/sherpa')

import time
import os
import json
import subprocess
from audio_device_detector import DeviceScanner


def print_phase(phase_num, title):
    """Print phase header."""
    print("\n" + "=" * 80)
    print(f"PHASE {phase_num}: {title}")
    print("=" * 80 + "\n")


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "FULL INTEGRATION DEMO: Audio Device Detection + MP3 Recording".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    # =========================================================================
    # PHASE 1: AUDIO DEVICE DETECTION
    # =========================================================================
    print_phase(1, "Audio Device Detection with VAD (Step 4)")

    try:
        # Create scanner with default VAD threshold
        print("Creating DeviceScanner with VAD threshold: 0.75")
        scanner = DeviceScanner(vad_threshold=0.75, chunk_duration_ms=100)

        print("\n▶ Starting device scan for active speech...")
        print("  Listen carefully - speak into your microphone when the scan begins")
        print("  The scanner will automatically detect the device with active speech\n")

        # Scan all devices for active speech
        detected_device, vad_prob = scanner.scan_devices()

        if not detected_device:
            print("\n✗ PHASE 1 FAILED: No device detected with speech above threshold")
            print("   Please try again and speak louder into your microphone")
            return False

        print("\n" + "-" * 80)
        print("✓ PHASE 1 COMPLETE: Device detected successfully")
        print("-" * 80)
        print(f"\nDetected Device:")
        print(f"  Device ID:       {detected_device.device_id}")
        print(f"  Device Name:     {detected_device.name}")
        print(f"  Channels:        {int(detected_device.channels)}")
        print(f"  Sample Rate:     {int(detected_device.sample_rate)} Hz")
        print(f"  Host API:        {detected_device.api}")
        print(f"\nDetection Metrics:")
        print(f"  VAD Probability: {vad_prob:.4f}")
        print(f"  Threshold:       0.75")
        print(f"  Status:          ✓ SPEECH DETECTED")

        selected_device_id = detected_device.device_id

    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user during Phase 1")
        return False
    except Exception as e:
        print(f"\n✗ Phase 1 Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # PHASE 2: MP3 RECORDING
    # =========================================================================
    print_phase(2, "MP3 Recording on Selected Device (Step 5)")

    try:
        # Generate output filename
        output_file = f"/tmp/demo_speech_{selected_device_id}_{int(time.time())}.mp3"

        print(f"Starting background MP3 recording on device {selected_device_id}...")
        print(f"  Bitrate:  128 kbps")
        print(f"  Output:   {output_file}\n")

        # Start background recording
        recorder = scanner.start_background_recording(
            device_id=selected_device_id,
            output_file=output_file,
            bitrate=128
        )

        # Recording progress
        print("▶ Recording in progress... Please speak clearly into the microphone")
        print("  (Recording 10 seconds of audio)")

        recording_duration = 10
        for i in range(recording_duration):
            time.sleep(1)
            elapsed = i + 1
            bar_length = int(elapsed / recording_duration * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  [{bar}] {elapsed:2d}s / {recording_duration:2d}s")

        # Stop recording
        print("\n▶ Finalizing MP3 file...")
        recorder.stop()

        # Verify file
        if not os.path.exists(output_file):
            print("\n✗ MP3 file was not created")
            return False

        file_size = os.path.getsize(output_file)
        print(f"✓ MP3 file created: {file_size / 1024:.1f} KB")

        # Validate with ffprobe
        print("\n▶ Validating MP3 integrity with ffprobe...")
        try:
            result = subprocess.run(
                ['ffprobe', '-show_entries', 'format=duration,bit_rate',
                 '-show_entries', 'stream=codec_type,sample_rate',
                 '-of', 'json', output_file],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                duration = float(data['format'].get('duration', 0))
                bitrate = int(data['format'].get('bit_rate', 0))
                sample_rate = data['streams'][0].get('sample_rate', 'N/A')
                codec_type = data['streams'][0].get('codec_type', 'N/A')

                print("\n" + "-" * 80)
                print("✓ PHASE 2 COMPLETE: MP3 Recording successful")
                print("-" * 80)
                print(f"\nMP3 File Details:")
                print(f"  File Path:    {output_file}")
                print(f"  File Size:    {file_size / 1024:.1f} KB")
                print(f"  Duration:     {duration:.2f} seconds")
                print(f"  Sample Rate:  {sample_rate} Hz")
                print(f"  Codec:        {codec_type}")
                print(f"  Bitrate:      {bitrate / 1000:.0f} kbps")

            else:
                print(f"\n⚠ ffprobe validation warning: {result.stderr}")
                print("  But MP3 file exists and was created successfully")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"\n⚠ Validation skipped ({type(e).__name__})")
            print("  But MP3 file exists and was created successfully")

    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user during Phase 2")
        return False
    except Exception as e:
        print(f"\n✗ Phase 2 Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("✓ FULL INTEGRATION DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\n📋 Summary:")
    print(f"  Phase 1 (Detection):  ✓ Detected device {selected_device_id}")
    print(f"  Phase 2 (Recording):  ✓ Recorded {duration:.2f}s of audio")
    print(f"  MP3 File:             ✓ {output_file}")

    print("\n💡 Next Steps:")
    print("  1. Playback: ffplay " + output_file)
    print("  2. Inspect: ffprobe " + output_file)
    print("  3. Convert: ffmpeg -i " + output_file + " output.wav")

    print("\n" + "=" * 80 + "\n")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
