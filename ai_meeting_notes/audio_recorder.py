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
    
    def get_available_devices(self) -> List[Dict]:
        """Get list of available audio input devices."""
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, device in enumerate(devices):
                if isinstance(device, dict) and device['max_input_channels'] > 0:
                    input_devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate'],
                        'is_blackhole': self.config.blackhole_device_name.lower() in device['name'].lower()
                    })
            return input_devices
        except Exception as e:
            raise AudioRecorderError(f"Failed to query audio devices: {e}")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback function for audio stream."""
        if status:
            print(f"Audio callback status: {status}")
        
        if self.current_session and self.current_session.is_active:
            # Convert to mono if needed
            if indata.shape[1] > 1:
                audio_chunk = np.mean(indata, axis=1)
            else:
                audio_chunk = indata[:, 0]
            
            self.current_session.add_audio_chunk(audio_chunk)
    
    def start_recording(self, filename: str) -> None:
        """Start audio recording to the specified file."""
        if self.is_recording():
            raise AudioRecorderError("Recording is already in progress")
        
        # Check if BlackHole is available
        if not self.check_blackhole_available():
            raise AudioRecorderError(
                "BlackHole device not found. Please install BlackHole and create a Multi-Output Device."
            )
        
        # Get BlackHole device ID
        device_id = self.get_blackhole_device_id()
        if device_id is None:
            raise AudioRecorderError("Could not find BlackHole input device")
        
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
            # Start audio stream
            self._stream = sd.InputStream(
                device=device_id,
                channels=2,  # BlackHole provides stereo, we'll convert to mono
                samplerate=self.config.sample_rate,
                callback=self._audio_callback,
                blocksize=self.config.chunk_size,
                dtype=np.float32
            )
            
            self._stream.start()
            self.current_session.is_active = True
            self.stop_event.clear()
            
            print(f"Started recording to: {filename}")
            device_info = sd.query_devices(device_id)
            device_name = device_info['name'] if isinstance(device_info, dict) else "Unknown Device"
            print(f"Using device: {device_name}")
            
        except Exception as e:
            self.current_session = None
            raise AudioRecorderError(f"Failed to start recording: {e}")
    
    def stop_recording(self) -> str:
        """Stop recording and save the audio file."""
        if not self.is_recording():
            raise AudioRecorderError("No recording in progress")
        
        try:
            # Stop the audio stream
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
            
            return filename
            
        except Exception as e:
            # Clean up on error
            if self.current_session:
                self.current_session.cleanup()
                self.current_session = None
            raise AudioRecorderError(f"Failed to stop recording: {e}")
    
    def is_recording(self) -> bool:
        """Check if recording is currently active."""
        return (self.current_session is not None and 
                self.current_session.is_active and
                self._stream is not None and
                self._stream.active)
    
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
        
        if self._stream:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        
        if self.current_session:
            self.current_session.cleanup()
            self.current_session = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()