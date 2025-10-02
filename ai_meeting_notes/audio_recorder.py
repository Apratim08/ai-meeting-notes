"""Audio recording functionality with BlackHole integration for macOS."""

import os
import time
import wave
import threading
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta

import sounddevice as sd
import numpy as np

from .config import config


class AudioRecorderError(Exception):
    """Custom exception for audio recording errors."""
    pass


class RecordingSession:
    """Manages the lifecycle of a single recording session."""
    
    def __init__(self, filename: str, sample_rate: int = 16000, channels: int = 1):
        self.filename = filename
        self.sample_rate = sample_rate
        self.channels = channels
        self.start_time = datetime.now()
        self.audio_data: List[np.ndarray] = []
        self.is_active = False
        self.total_frames = 0
        
    def add_audio_chunk(self, chunk: np.ndarray) -> None:
        """Add an audio chunk to the session."""
        if self.is_active:
            self.audio_data.append(chunk.copy())
            self.total_frames += len(chunk)
    
    def get_duration(self) -> float:
        """Get the current recording duration in seconds."""
        if self.total_frames == 0:
            return 0.0
        return self.total_frames / self.sample_rate
    
    def save_to_file(self) -> str:
        """Save the recorded audio to a WAV file."""
        if not self.audio_data:
            raise AudioRecorderError("No audio data to save")
        
        # Ensure directory exists
        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Concatenate all audio chunks
        full_audio = np.concatenate(self.audio_data)
        
        # Convert to int16 for WAV format
        audio_int16 = (full_audio * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(self.filename, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return self.filename
    
    def cleanup(self) -> None:
        """Clean up session resources."""
        self.audio_data.clear()
        self.is_active = False


class AudioRecorder:
    """Audio recorder with BlackHole integration for capturing system audio."""
    
    def __init__(self):
        self.config = config.audio
        self.current_session: Optional[RecordingSession] = None
        self.recording_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self._stream: Optional[sd.InputStream] = None

        # Dual-stream mixing state
        self._blackhole_stream: Optional[sd.InputStream] = None
        self._mic_stream: Optional[sd.InputStream] = None
        self._latest_blackhole: Optional[np.ndarray] = None
        self._latest_mic: Optional[np.ndarray] = None
        self._mix_lock = threading.Lock()
        self._recording_mode: str = "single"  # "single" or "dual"
        
    def check_blackhole_available(self) -> bool:
        """Check if BlackHole virtual audio device is available."""
        try:
            devices = sd.query_devices()
            for device in devices:
                if (isinstance(device, dict) and 
                    self.config.blackhole_device_name.lower() in device['name'].lower() and
                    device['max_input_channels'] > 0):
                    return True
            return False
        except Exception:
            return False
    
    def get_blackhole_device_id(self) -> Optional[int]:
        """Get the device ID for BlackHole input device."""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if (isinstance(device, dict) and 
                    self.config.blackhole_device_name.lower() in device['name'].lower() and
                    device['max_input_channels'] > 0):
                    return i
            return None
        except Exception:
            return None
    
    def check_multi_output_setup(self) -> bool:
        """Check if Multi-Output Device is properly configured."""
        try:
            devices = sd.query_devices()
            for device in devices:
                if (isinstance(device, dict) and 
                    'multi-output' in device['name'].lower() and
                    device['max_output_channels'] > 0):
                    return True
            return False
        except Exception:
            return False
    
    def get_setup_instructions(self) -> str:
        """Get detailed setup instructions for BlackHole and Multi-Output Device."""
        return "Setup instructions for BlackHole and Multi-Output Device configuration."
    
    def get_microphone_device_id(self) -> Optional[int]:
        """Get the device ID for microphone input device."""
        try:
            devices = sd.query_devices()

            # If specific microphone name is configured, find it
            if self.config.microphone_device_name:
                for i, device in enumerate(devices):
                    if (isinstance(device, dict) and
                        self.config.microphone_device_name.lower() in device['name'].lower() and
                        device['max_input_channels'] > 0):
                        return i
                return None

            # Otherwise, find the default microphone (exclude BlackHole)
            default_input = sd.query_devices(kind='input')
            if isinstance(default_input, dict):
                device_name = default_input['name'].lower()
                # Skip if it's BlackHole
                if self.config.blackhole_device_name.lower() not in device_name:
                    return default_input['index'] if 'index' in default_input else None

            # Fallback: find first non-BlackHole input device
            for i, device in enumerate(devices):
                if (isinstance(device, dict) and
                    device['max_input_channels'] > 0 and
                    self.config.blackhole_device_name.lower() not in device['name'].lower()):
                    return i

            return None
        except Exception:
            return None

    def check_microphone_available(self) -> bool:
        """Check if a microphone device is available."""
        return self.get_microphone_device_id() is not None

    def get_available_devices(self) -> List[Dict]:
        """Get list of available audio input devices."""
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, device in enumerate(devices):
                if isinstance(device, dict) and device['max_input_channels'] > 0:
                    is_blackhole = self.config.blackhole_device_name.lower() in device['name'].lower()
                    input_devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate'],
                        'is_blackhole': is_blackhole,
                        'is_microphone': not is_blackhole
                    })
            return input_devices
        except Exception as e:
            raise AudioRecorderError(f"Failed to query audio devices: {e}")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback function for single-stream audio (legacy)."""
        if status:
            print(f"Audio callback status: {status}")

        if self.current_session and self.current_session.is_active:
            # Convert to mono if needed
            if indata.shape[1] > 1:
                audio_chunk = np.mean(indata, axis=1)
            else:
                audio_chunk = indata[:, 0]

            self.current_session.add_audio_chunk(audio_chunk)

    def _blackhole_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback for BlackHole stream in dual-stream mode."""
        if status:
            print(f"BlackHole callback status: {status}")

        if not self.current_session or not self.current_session.is_active:
            return

        # Convert to mono
        if indata.shape[1] > 1:
            blackhole_mono = np.mean(indata, axis=1)
        else:
            blackhole_mono = indata[:, 0]

        with self._mix_lock:
            self._latest_blackhole = blackhole_mono.copy()

            # Mix with latest mic data if available
            if self._latest_mic is not None:
                # Apply mic gain from config
                mic_adjusted = self._latest_mic * self.config.mic_gain
                # Mix: average both sources
                mixed = (blackhole_mono + mic_adjusted) / 2.0
                self.current_session.add_audio_chunk(mixed)
            else:
                # No mic data yet, use BlackHole only
                self.current_session.add_audio_chunk(blackhole_mono)

    def _mic_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback for microphone stream in dual-stream mode."""
        if status:
            print(f"Microphone callback status: {status}")

        if not self.current_session or not self.current_session.is_active:
            return

        # Convert to mono
        if indata.shape[1] > 1:
            mic_mono = np.mean(indata, axis=1)
        else:
            mic_mono = indata[:, 0]

        with self._mix_lock:
            self._latest_mic = mic_mono.copy()

            # Mix with latest BlackHole data if available
            if self._latest_blackhole is not None:
                # Apply mic gain from config
                mic_adjusted = mic_mono * self.config.mic_gain
                # Mix: average both sources
                mixed = (self._latest_blackhole + mic_adjusted) / 2.0
                self.current_session.add_audio_chunk(mixed)
            else:
                # No BlackHole data yet, use mic only
                mic_adjusted = mic_mono * self.config.mic_gain
                self.current_session.add_audio_chunk(mic_adjusted)
    
    def start_recording(self, filename: str) -> None:
        """Start audio recording to the specified file."""
        if self.is_recording():
            raise AudioRecorderError("Recording is already in progress")

        # Get available devices
        blackhole_id = self.get_blackhole_device_id()
        mic_id = self.get_microphone_device_id()

        # Determine recording mode based on config and device availability
        use_dual_mode = (
            self.config.enable_microphone and
            blackhole_id is not None and
            mic_id is not None
        )

        # Fallback logic
        if use_dual_mode:
            self._recording_mode = "dual"
            print("🎙️  Recording mode: Dual-stream (BlackHole + Microphone)")
        elif blackhole_id is not None:
            self._recording_mode = "blackhole_only"
            print("🔊 Recording mode: BlackHole only (system audio)")
            if self.config.enable_microphone:
                print("⚠️  Warning: Microphone enabled but not found. Using BlackHole only.")
        elif mic_id is not None:
            self._recording_mode = "mic_only"
            print("🎤 Recording mode: Microphone only")
            print("⚠️  Warning: BlackHole not found. Recording microphone only.")
        else:
            raise AudioRecorderError(
                "No audio input devices found. Please check your audio setup."
            )

        # Validate file path
        file_path = Path(filename)
        if not file_path.suffix.lower() == '.wav':
            filename = str(file_path.with_suffix('.wav'))

        # Create recording session
        self.current_session = RecordingSession(
            filename=filename,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels
        )

        try:
            if self._recording_mode == "dual":
                self._start_dual_stream(blackhole_id, mic_id, filename)
            elif self._recording_mode == "blackhole_only":
                self._start_single_stream(blackhole_id, filename, "BlackHole")
            elif self._recording_mode == "mic_only":
                self._start_single_stream(mic_id, filename, "Microphone")

            self.current_session.is_active = True
            self.stop_event.clear()

        except Exception as e:
            self.current_session = None
            raise AudioRecorderError(f"Failed to start recording: {e}")

    def _start_dual_stream(self, blackhole_id: int, mic_id: int, filename: str) -> None:
        """Start dual-stream recording with real-time mixing."""
        # Initialize mixing buffers
        self._latest_blackhole = np.zeros(self.config.chunk_size, dtype=np.float32)
        self._latest_mic = np.zeros(self.config.chunk_size, dtype=np.float32)

        # Start BlackHole stream
        self._blackhole_stream = sd.InputStream(
            device=blackhole_id,
            channels=2,  # BlackHole provides stereo
            samplerate=self.config.sample_rate,
            callback=self._blackhole_callback,
            blocksize=self.config.chunk_size,
            dtype=np.float32
        )

        # Start microphone stream
        self._mic_stream = sd.InputStream(
            device=mic_id,
            channels=1,  # Mono mic input
            samplerate=self.config.sample_rate,
            callback=self._mic_callback,
            blocksize=self.config.chunk_size,
            dtype=np.float32
        )

        self._blackhole_stream.start()
        self._mic_stream.start()

        print(f"Started recording to: {filename}")

        blackhole_info = sd.query_devices(blackhole_id)
        mic_info = sd.query_devices(mic_id)

        blackhole_name = blackhole_info['name'] if isinstance(blackhole_info, dict) else "Unknown"
        mic_name = mic_info['name'] if isinstance(mic_info, dict) else "Unknown"

        print(f"Using BlackHole: {blackhole_name}")
        print(f"Using Microphone: {mic_name}")

    def _start_single_stream(self, device_id: int, filename: str, device_type: str) -> None:
        """Start single-stream recording (legacy mode)."""
        self._stream = sd.InputStream(
            device=device_id,
            channels=2,  # Assume stereo, convert to mono in callback
            samplerate=self.config.sample_rate,
            callback=self._audio_callback,
            blocksize=self.config.chunk_size,
            dtype=np.float32
        )

        self._stream.start()

        print(f"Started recording to: {filename}")
        device_info = sd.query_devices(device_id)
        device_name = device_info['name'] if isinstance(device_info, dict) else "Unknown Device"
        print(f"Using device ({device_type}): {device_name}")
    
    def stop_recording(self) -> str:
        """Stop recording and save the audio file."""
        if not self.is_recording():
            raise AudioRecorderError("No recording in progress")

        try:
            # Stop dual streams if active
            if self._blackhole_stream:
                self._blackhole_stream.stop()
                self._blackhole_stream.close()
                self._blackhole_stream = None

            if self._mic_stream:
                self._mic_stream.stop()
                self._mic_stream.close()
                self._mic_stream = None

            # Stop single stream if active
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            # Mark session as inactive
            self.current_session.is_active = False
            self.stop_event.set()

            # Save the recording
            filename = self.current_session.save_to_file()
            duration = self.current_session.get_duration()

            print(f"Recording stopped. Duration: {duration:.2f} seconds")
            print(f"Saved to: {filename}")

            # Clean up session
            self.current_session.cleanup()
            self.current_session = None

            # Reset mixing state
            self._latest_blackhole = None
            self._latest_mic = None
            self._recording_mode = "single"

            return filename

        except Exception as e:
            # Clean up on error
            if self.current_session:
                self.current_session.cleanup()
                self.current_session = None
            raise AudioRecorderError(f"Failed to stop recording: {e}")
    
    def is_recording(self) -> bool:
        """Check if recording is currently active."""
        if self.current_session is None or not self.current_session.is_active:
            return False

        # Check dual-stream mode
        if self._recording_mode == "dual":
            return (
                self._blackhole_stream is not None and
                self._mic_stream is not None and
                self._blackhole_stream.active and
                self._mic_stream.active
            )

        # Check single-stream mode
        return self._stream is not None and self._stream.active
    
    def get_recording_duration(self) -> float:
        """Get the current recording duration in seconds."""
        if self.current_session:
            return self.current_session.get_duration()
        return 0.0
    
    def get_recording_info(self) -> Dict:
        """Get information about the current recording session."""
        if not self.current_session:
            return {
                'is_recording': False,
                'duration': 0.0,
                'filename': None,
                'start_time': None
            }
        
        return {
            'is_recording': self.is_recording(),
            'duration': self.current_session.get_duration(),
            'filename': self.current_session.filename,
            'start_time': self.current_session.start_time.isoformat(),
            'sample_rate': self.current_session.sample_rate,
            'channels': self.current_session.channels
        }
    
    def cleanup(self) -> None:
        """Clean up recorder resources."""
        if self.is_recording():
            try:
                self.stop_recording()
            except Exception:
                pass  # Best effort cleanup

        # Clean up dual streams
        if self._blackhole_stream:
            try:
                self._blackhole_stream.close()
            except Exception:
                pass
            self._blackhole_stream = None

        if self._mic_stream:
            try:
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

        # Clean up single stream
        if self._stream:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self.current_session:
            self.current_session.cleanup()
            self.current_session = None

        # Reset mixing state
        self._latest_blackhole = None
        self._latest_mic = None
        self._recording_mode = "single"
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()