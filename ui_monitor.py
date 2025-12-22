#!/usr/bin/env python3

"""
Real-time Monitor UI for ASR System with Fixed-Grid Layout

Implements the three-zone UI layout as specified in the README:
3.2.0 header showing system status
3.2.1 2 rows of dynamic bar charts for VAD and RMS
3.2.2 N rows of left-truncated, per-engine streaming transcripts
"""

import time
import numpy as np
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque

from asr_engine import StreamResult


@dataclass
class DeviceStatus:
    """Information about the active audio device."""
    device_id: int
    name: str
    sample_rate: int
    channels: int
    dtype: str
    api: str


class DockerCompatibleUI:
    """Docker-compatible text-based UI with fixed-grid layout for real-time monitoring."""

    def __init__(self):
        self.device_status: Optional[DeviceStatus] = None
        self.vad_levels: deque = deque(maxlen=50)  # Store last 50 VAD values
        self.rms_levels: deque = deque(maxlen=50)  # Store last 50 RMS values
        self.transcripts: Dict[str, str] = {}  # Model ID -> latest transcript
        self.hotwords: List[str] = []  # Currently active hotwords
        self.recognized_hotwords: set = set()  # Hotwords that have been recognized
        self.all_hotwords: List[str] = []  # Complete list of configured hotwords
        self.lock = threading.Lock()
        self.refresh_rate = 0.5  # UI refresh rate in seconds
        self.last_ui_output = ""  # Store last UI output to avoid flickering

        # Initialize OpenCC for zh-CN to zh-TW conversion
        try:
            import opencc
            self.opencc_converter = opencc.OpenCC('s2t')  # Simplified to Traditional
            self.opencc_available = True
        except ImportError:
            print("Warning: opencc-python-reimplemented not available. zh-CN to zh-TW conversion disabled.")
            self.opencc_converter = None
            self.opencc_available = False
        except Exception as e:
            print(f"Warning: OpenCC initialization failed: {e}. zh-CN to zh-TW conversion disabled.")
            self.opencc_converter = None
            self.opencc_available = False
        
    def set_device_status(self, device_status: DeviceStatus):
        """Update device status information."""
        with self.lock:
            self.device_status = device_status
    
    def update_audio_levels(self, vad_level: float, rms_level: float):
        """Update VAD and RMS levels for the bar charts."""
        with self.lock:
            self.vad_levels.append(vad_level)
            self.rms_levels.append(rms_level)
    
    def update_transcript(self, model_id: str, transcript: str):
        """Update transcript for a specific model."""
        with self.lock:
            # Convert transcript from zh-CN to zh-TW if OpenCC is available
            if self.opencc_available and transcript.strip():
                try:
                    transcript = self.opencc_converter.convert(transcript)
                except Exception as e:
                    print(f"Warning: OpenCC conversion failed: {e}")

            # Keep only the most recent portion of the transcript (left-truncated)
            max_length = 60  # Show last 60 characters
            if len(transcript) > max_length:
                transcript = transcript[-max_length:]
            self.transcripts[model_id] = transcript

            # Check if any hotwords are in this transcript
            if self.all_hotwords:
                for hw in self.all_hotwords:
                    if hw.lower() in transcript.lower():
                        self.recognized_hotwords.add(hw)
    
    def update_hotwords(self, hotwords: List[str]):
        """Update the list of active hotwords."""
        with self.lock:
            self.hotwords = hotwords
            self.all_hotwords = hotwords[:]  # Make a copy of the full list
    
    def draw_ui(self, asr_results: Optional[Dict[str, StreamResult]] = None) -> str:
        """Draw the complete UI with all three zones and return as string."""
        ui_parts = []
        
        # Zone 3.2.0: Header with system status
        ui_parts.append(self._draw_header_zone())
        
        # Zone 3.2.1: VAD and RMS bar charts
        ui_parts.append(self._draw_audio_charts_zone())
        
        # Zone 3.2.2: Per-engine streaming transcripts
        ui_parts.append(self._draw_transcripts_zone(asr_results))
        
        # Add hotwords info at the bottom
        ui_parts.append(self._draw_hotwords_info())
        
        # Join all parts with newlines
        full_ui = "\n".join(ui_parts)
        return full_ui
    
    def _draw_header_zone(self) -> str:
        """Draw the header zone with device status."""
        lines = []
        lines.append("┌" + "─" * 80 + "┐")
        lines.append("│ AUDIO DEVICE STATUS                                                      │")
        lines.append("├" + "─" * 80 + "┤")
        
        if self.device_status:
            lines.append(f"│ Device:     {self.device_status.name[:60]:<60} │")
            lines.append(f"│ ID:         {self.device_status.device_id:<66} │")
            lines.append(f"│ Sample Rate: {self.device_status.sample_rate} Hz{'':<59} │")
            lines.append(f"│ Channels:   {self.device_status.channels}{'':<67} │")
            lines.append(f"│ Data Type:  {self.device_status.dtype:<67} │")
            lines.append(f"│ API:        {self.device_status.api:<67} │")
        else:
            lines.append("│ No device selected                                                     │")
        
        lines.append("├" + "─" * 80 + "┤")
        return "\n".join(lines)
    
    def _draw_audio_charts_zone(self) -> str:
        """Draw the VAD and RMS bar charts."""
        lines = []
        lines.append("│ AUDIO LEVELS (VAD & RMS)                                                 │")
        lines.append("├" + "─" * 80 + "┤")

        # Draw VAD chart
        lines.append("│ VAD LEVEL:                                                              │")
        if self.vad_levels:
            vad_chart = self._draw_single_bar_chart(list(self.vad_levels), max_val=1.0, width=76)
            lines.extend([f"│ {line:<76} │" for line in vad_chart])
        else:
            lines.append("│ [No VAD data available]                                                │")

        # Draw RMS chart with logarithmic scaling for better visibility of low levels
        lines.append("│ RMS LEVEL (log scale):                                                  │")
        if self.rms_levels:
            rms_chart = self._draw_single_bar_chart(list(self.rms_levels), max_val=1.0, width=76, use_log_scale=True)
            lines.extend([f"│ {line:<76} │" for line in rms_chart])
        else:
            lines.append("│ [No RMS data available]                                                │")

        lines.append("├" + "─" * 80 + "┤")
        return "\n".join(lines)
    
    def _draw_single_bar_chart(self, values: List[float], max_val: float, width: int, use_log_scale: bool = False) -> List[str]:
        """Draw a single horizontal bar chart using text characters."""
        if not values:
            return [" " * width]

        # Get the most recent values (up to the width)
        recent_values = values[-width:] if len(values) > width else values

        # Create a simple text-based bar chart
        chart_lines = []
        chart_line = ""
        for val in recent_values:
            if use_log_scale and val > 0:
                # Use logarithmic scaling: log10(val) normalized to useful range
                # Shift and scale to map small positive values to visible range
                log_val = max(0.0, (np.log10(max(val, 1e-6)) + 6) / 6)  # Map from 1e-6 (0) to 1.0 (1)
                norm_val = min(1.0, max(0.0, log_val))
            else:
                # Normalize value to 0-10 range for character selection
                norm_val = min(1.0, max(0.0, val / max_val))

            if norm_val < 0.1:
                char = ' '
            elif norm_val < 0.3:
                char = '░'
            elif norm_val < 0.6:
                char = '▒'
            elif norm_val < 0.9:
                char = '▓'
            else:
                char = '█'
            chart_line += char

        # Pad to full width
        chart_line += ' ' * (width - len(recent_values))
        chart_lines.append(chart_line)

        return chart_lines
    
    def _draw_transcripts_zone(self, asr_results: Optional[Dict[str, StreamResult]] = None) -> str:
        """Draw the per-engine streaming transcripts."""
        lines = []
        lines.append("│ ASR TRANSCRIPTS (Left-truncated, Most Recent Text)                       │")
        lines.append("├" + "─" * 80 + "┤")

        if not self.transcripts and not asr_results:
            lines.append("│ [No transcripts available]                                             │")
        else:
            # Use asr_results if available, otherwise fall back to self.transcripts
            results_to_display = asr_results if asr_results else {}
            if not results_to_display:
                # If no asr_results provided, use self.transcripts
                for model_id, transcript in self.transcripts.items():
                    display_text = f"{model_id}: {transcript}"
                    if len(display_text) > 76:
                        display_text = display_text[:76]
                    else:
                        display_text = display_text.ljust(76)
                    lines.append(f"│ {display_text} │")
            else:
                # Use asr_results with additional information
                for model_id, result in results_to_display.items():
                    # Combine partial and final results, showing the most recent
                    transcript = result.partial if result.partial else result.final
                    if not transcript:
                        transcript = "(empty)"

                    # Convert transcript from zh-CN to zh-TW if OpenCC is available
                    if self.opencc_available and transcript and transcript != "(empty)":
                        try:
                            transcript = self.opencc_converter.convert(transcript)
                        except Exception as e:
                            print(f"Warning: OpenCC conversion failed in UI: {e}")

                    # Left-truncate to show most recent text
                    max_length = 50
                    if len(transcript) > max_length:
                        transcript = f"...{transcript[-(max_length-3):]}"

                    # Add model name and latency
                    display_text = f"{model_id}: {transcript} [{result.latency_ms:.1f}ms]"
                    if len(display_text) > 76:
                        display_text = display_text[:76]
                    else:
                        display_text = display_text.ljust(76)
                    lines.append(f"│ {display_text} │")

        # Fill remaining space if needed (max 6 transcript rows)
        max_rows = 6  # Maximum number of transcript rows
        actual_rows = len(asr_results) if asr_results else len(self.transcripts)
        for _ in range(max(0, max_rows - actual_rows)):
            lines.append("│                                                                        │")

        lines.append("└" + "─" * 80 + "┘")
        return "\n".join(lines)
    
    def _draw_hotwords_info(self) -> str:
        """Draw information about active hotwords."""
        lines = []
        lines.append("\nHOTWORDS STATUS:")
        
        if self.all_hotwords:
            # Group hotwords by recognition status
            recognized = []
            not_recognized = []
            
            for hw in self.all_hotwords:
                if hw in self.recognized_hotwords:
                    recognized.append(hw)
                else:
                    not_recognized.append(hw)
            
            if recognized:
                lines.append(f"  RECOGNIZED ({len(recognized)}): {', '.join(recognized[:5])}" + 
                           ("..." if len(recognized) > 5 else ""))
            
            if not_recognized:
                lines.append(f"  NOT RECOGNIZED ({len(not_recognized)}): {', '.join(not_recognized[:5])}" + 
                           ("..." if len(not_recognized) > 5 else ""))
            
            lines.append(f"  TOTAL: {len(self.all_hotwords)} hotwords configured")
        else:
            lines.append("  No hotwords configured")
        
        return "\n".join(lines)
    
    def get_status_text(self) -> str:
        """Get a text representation of the current status for logging."""
        status = []
        if self.device_status:
            status.append(f"Device: {self.device_status.name} ({self.device_status.device_id})")
            status.append(f"Sample Rate: {self.device_status.sample_rate}Hz")
        
        status.append(f"Active Models: {len(self.transcripts)}")
        status.append(f"Hotwords: {len(self.hotwords)}")
        status.append(f"Recognized: {len(self.recognized_hotwords)}/{len(self.all_hotwords) if self.all_hotwords else 0}")
        
        return " | ".join(status)


class UIThread:
    """Manages the UI in a separate thread."""
    
    def __init__(self, ui: DockerCompatibleUI):
        self.ui = ui
        self.running = False
        self.thread = None
        self.asr_results = {}
        self.update_lock = threading.Lock()
        
    def start(self):
        """Start the UI thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the UI thread."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def update_results(self, results: Dict[str, StreamResult]):
        """Update the ASR results to display."""
        with self.update_lock:
            self.asr_results = results.copy()
    
    def _run(self):
        """Main UI update loop."""
        while self.running:
            try:
                with self.update_lock:
                    current_results = self.asr_results.copy()
                
                ui_output = self.ui.draw_ui(current_results)
                
                # Only print if the UI output has changed to reduce log noise in Docker
                if ui_output != self.ui.last_ui_output:
                    print(ui_output)
                    self.ui.last_ui_output = ui_output
                
                time.sleep(self.ui.refresh_rate)
            except Exception as e:
                print(f"UI Error: {e}")
                break


if __name__ == "__main__":
    # Demo/test the UI
    print("Testing DockerCompatibleUI...")
    
    ui = DockerCompatibleUI()
    
    # Simulate some data
    ui.set_device_status(DeviceStatus(
        device_id=1,
        name="Built-in Microphone",
        sample_rate=16000,
        channels=1,
        dtype="float32",
        api="MME"
    ))
    
    ui.update_hotwords(["张伟明", "LILY", "MICHAEL", "李建国", "EMMA"])
    
    # Simulate updating data
    for i in range(20):
        # Update audio levels with simulated data
        vad = (np.sin(i * 0.2) + 1) / 2  # Oscillating between 0 and 1
        rms = (np.sin(i * 0.25) + 1) / 2 * 0.8  # Different oscillation
        ui.update_audio_levels(vad, rms)
        
        # Update transcripts
        ui.update_transcript("model1", f"Transcript from model 1 - iteration {i}")
        ui.update_transcript("model2", f"Another transcript from model 2 - {i}")
        
        ui_output = ui.draw_ui()
        print(ui_output)
        print("\n" + "="*80 + "\n")
        
        time.sleep(1)
    
    print("UI demo finished.")