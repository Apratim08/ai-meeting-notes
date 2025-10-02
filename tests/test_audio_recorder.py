"""Unit tests for AudioRecorder class."""

import os
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import sounddevice as sd

from ai_meeting_notes.audio_recorder import (
    AudioRecorder, 
    RecordingSession, 
    AudioRecorderError
)


class TestRecordingSession:
    """Test cases for RecordingSession class."""
    
    def test_recording_session_initialization(self):
        """Test RecordingSession initialization."""
        session = RecordingSession("test.wav", sample_rate=16000, channels=1)
        
        assert session.filename == "test.wav"
        assert session.sample_rate == 16000
        assert session.channels == 1
        assert session.total_frames == 0
        assert not session.is_active
        assert len(session.audio_data) == 0
    
    def test_add_audio_chunk_when_active(self):
        """Test adding audio chunks when session is active."""
        session = RecordingSession("test.wav")
        session.is_active = True
        
        # Add some test audio data
        chunk1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        chunk2 = np.array([0.4, 0.5], dtype=np.float32)
        
        session.add_audio_chunk(chunk1)
        session.add_audio_chunk(chunk2)
        
        assert len(session.audio_data) == 2
        assert session.total_frames == 5
        assert np.array_equal(session.audio_data[0], chunk1)
        assert np.array_equal(session.audio_data[1], chunk2)
    
    def test_add_audio_chunk_when_inactive(self):
        """Test that audio chunks are not added when session is inactive."""
        session = RecordingSession("test.wav")
        session.is_active = False
        
        chunk = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        session.add_audio_chunk(chunk)
        
        assert len(session.audio_data) == 0
        assert session.total_frames == 0
    
    def test_get_duration(self):
        """Test duration calculation."""
        session = RecordingSession("test.wav", sample_rate=16000)
        
        # No audio data
        assert session.get_duration() == 0.0
        
        # Add some frames
        session.total_frames = 16000  # 1 second at 16kHz
        assert session.get_duration() == 1.0
        
        session.total_frames = 8000   # 0.5 seconds at 16kHz
        assert session.get_duration() == 0.5
    
    def test_save_to_file(self):
        """Test saving audio data to WAV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test_recording.wav")
            session = RecordingSession(filename, sample_rate=16000, channels=1)
            
            # Add some test audio data
            session.audio_data = [
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                np.array([0.4, 0.5], dtype=np.float32)
            ]
            session.total_frames = 5
            
            result_filename = session.save_to_file()
            
            assert result_filename == filename
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
    
    def test_save_to_file_no_data(self):
        """Test saving when no audio data exists."""
        session = RecordingSession("test.wav")
        
        with pytest.raises(AudioRecorderError, match="No audio data to save"):
            session.save_to_file()
    
    def test_cleanup(self):
        """Test session cleanup."""
        session = RecordingSession("test.wav")
        session.is_active = True
        session.audio_data = [np.array([1, 2, 3])]
        
        session.cleanup()
        
        assert not session.is_active
        assert len(session.audio_data) == 0


class TestAudioRecorder:
    """Test cases for AudioRecorder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.recorder = AudioRecorder()
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'recorder'):
            self.recorder.cleanup()
    
    @patch('sounddevice.query_devices')
    def test_check_blackhole_available_true(self, mock_query):
        """Test BlackHole detection when available."""
        mock_query.return_value = [
            {'name': 'Built-in Microphone', 'max_input_channels': 1},
            {'name': 'BlackHole 2ch', 'max_input_channels': 2},
            {'name': 'Built-in Output', 'max_input_channels': 0}
        ]
        
        assert self.recorder.check_blackhole_available() is True
    
    @patch('sounddevice.query_devices')
    def test_check_blackhole_available_false(self, mock_query):
        """Test BlackHole detection when not available."""
        mock_query.return_value = [
            {'name': 'Built-in Microphone', 'max_input_channels': 1},
            {'name': 'Built-in Output', 'max_input_channels': 0}
        ]
        
        assert self.recorder.check_blackhole_available() is False
    
    @patch('sounddevice.query_devices')
    def test_check_blackhole_available_exception(self, mock_query):
        """Test BlackHole detection when query_devices raises exception."""
        mock_query.side_effect = Exception("Device query failed")
        
        assert self.recorder.check_blackhole_available() is False
    
    @patch('sounddevice.query_devices')
    def test_get_blackhole_device_id(self, mock_query):
        """Test getting BlackHole device ID."""
        mock_query.return_value = [
            {'name': 'Built-in Microphone', 'max_input_channels': 1},
            {'name': 'BlackHole 2ch', 'max_input_channels': 2},
            {'name': 'Built-in Output', 'max_input_channels': 0}
        ]
        
        device_id = self.recorder.get_blackhole_device_id()
        assert device_id == 1
    
    @patch('sounddevice.query_devices')
    def test_get_blackhole_device_id_not_found(self, mock_query):
        """Test getting BlackHole device ID when not found."""
        mock_query.return_value = [
            {'name': 'Built-in Microphone', 'max_input_channels': 1},
            {'name': 'Built-in Output', 'max_input_channels': 0}
        ]
        
        device_id = self.recorder.get_blackhole_device_id()
        assert device_id is None
    
    @patch('sounddevice.query_devices')
    def test_check_multi_output_setup(self, mock_query):
        """Test Multi-Output Device detection."""
        mock_query.return_value = [
            {'name': 'Built-in Output', 'max_output_channels': 2},
            {'name': 'Multi-Output Device', 'max_output_channels': 4},
            {'name': 'BlackHole 2ch', 'max_output_channels': 2}
        ]
        
        assert self.recorder.check_multi_output_setup() is True
    
    def test_get_setup_instructions(self):
        """Test getting setup instructions."""
        instructions = self.recorder.get_setup_instructions()
        
        assert "BlackHole" in instructions
        assert "Multi-Output Device" in instructions
        assert "Audio MIDI Setup" in instructions
        assert "Step 1" in instructions
        assert "Step 2" in instructions
    
    @patch('sounddevice.query_devices')
    def test_get_available_devices(self, mock_query):
        """Test getting available input devices."""
        mock_query.return_value = [
            {'name': 'Built-in Microphone', 'max_input_channels': 1, 'default_samplerate': 44100},
            {'name': 'BlackHole 2ch', 'max_input_channels': 2, 'default_samplerate': 48000},
            {'name': 'Built-in Output', 'max_input_channels': 0, 'default_samplerate': 44100}
        ]
        
        devices = self.recorder.get_available_devices()
        
        assert len(devices) == 2
        assert devices[0]['name'] == 'Built-in Microphone'
        assert devices[0]['is_blackhole'] is False
        assert devices[1]['name'] == 'BlackHole 2ch'
        assert devices[1]['is_blackhole'] is True
    
    def test_is_recording_false_initially(self):
        """Test that recorder is not recording initially."""
        assert self.recorder.is_recording() is False
    
    def test_get_recording_duration_zero_initially(self):
        """Test that recording duration is zero initially."""
        assert self.recorder.get_recording_duration() == 0.0
    
    def test_get_recording_info_no_session(self):
        """Test getting recording info when no session exists."""
        info = self.recorder.get_recording_info()
        
        expected = {
            'is_recording': False,
            'duration': 0.0,
            'filename': None,
            'start_time': None
        }
        assert info == expected
    
    @patch('sounddevice.query_devices')
    @patch('sounddevice.InputStream')
    def test_start_recording_success(self, mock_stream_class, mock_query):
        """Test successful recording start."""
        # Mock BlackHole device availability - handle both list and single device queries
        def mock_query_devices(device_id=None):
            devices = [{'name': 'BlackHole 2ch', 'max_input_channels': 2}]
            if device_id is not None:
                return devices[device_id] if device_id < len(devices) else devices[0]
            return devices
        
        mock_query.side_effect = mock_query_devices
        
        # Mock audio stream
        mock_stream = Mock()
        mock_stream.active = True  # Mock the active property
        mock_stream_class.return_value = mock_stream
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test.wav")
            
            self.recorder.start_recording(filename)
            
            assert self.recorder.is_recording() is True
            assert self.recorder.current_session is not None
            assert self.recorder.current_session.filename == filename
            mock_stream.start.assert_called_once()
    
    def test_start_recording_no_blackhole(self):
        """Test recording start when BlackHole is not available."""
        with patch.object(self.recorder, 'check_blackhole_available', return_value=False):
            with pytest.raises(AudioRecorderError, match="BlackHole device not found"):
                self.recorder.start_recording("test.wav")
    
    def test_start_recording_already_recording(self):
        """Test starting recording when already recording."""
        # Mock that recording is already in progress
        self.recorder.current_session = Mock()
        self.recorder.current_session.is_active = True
        self.recorder._stream = Mock()
        self.recorder._stream.active = True
        
        with pytest.raises(AudioRecorderError, match="Recording is already in progress"):
            self.recorder.start_recording("test.wav")
    
    @patch('sounddevice.query_devices')
    @patch('sounddevice.InputStream')
    def test_stop_recording_success(self, mock_stream_class, mock_query):
        """Test successful recording stop."""
        # Setup mocks - handle both list and single device queries
        def mock_query_devices(device_id=None):
            devices = [{'name': 'BlackHole 2ch', 'max_input_channels': 2}]
            if device_id is not None:
                return devices[device_id] if device_id < len(devices) else devices[0]
            return devices
        
        mock_query.side_effect = mock_query_devices
        mock_stream = Mock()
        mock_stream.active = True  # Mock the active property
        mock_stream_class.return_value = mock_stream
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test.wav")
            
            # Start recording
            self.recorder.start_recording(filename)
            
            # Add some mock audio data
            self.recorder.current_session.audio_data = [
                np.array([0.1, 0.2, 0.3], dtype=np.float32)
            ]
            self.recorder.current_session.total_frames = 3
            
            # Stop recording
            result_filename = self.recorder.stop_recording()
            
            assert result_filename == filename
            assert self.recorder.is_recording() is False
            assert self.recorder.current_session is None
            mock_stream.stop.assert_called_once()
            mock_stream.close.assert_called_once()
    
    def test_stop_recording_not_recording(self):
        """Test stopping recording when not recording."""
        with pytest.raises(AudioRecorderError, match="No recording in progress"):
            self.recorder.stop_recording()
    
    def test_audio_callback_mono_conversion(self):
        """Test audio callback with stereo to mono conversion."""
        # Create a mock session
        session = Mock()
        session.is_active = True
        self.recorder.current_session = session
        
        # Create stereo input data
        stereo_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        
        # Call the callback
        self.recorder._audio_callback(stereo_data, 3, None, None)
        
        # Verify mono conversion and session call
        session.add_audio_chunk.assert_called_once()
        call_args = session.add_audio_chunk.call_args[0][0]
        expected_mono = np.array([0.15, 0.35, 0.55], dtype=np.float32)  # Average of stereo channels
        np.testing.assert_array_almost_equal(call_args, expected_mono)
    
    def test_audio_callback_already_mono(self):
        """Test audio callback with mono input data."""
        # Create a mock session
        session = Mock()
        session.is_active = True
        self.recorder.current_session = session
        
        # Create mono input data
        mono_data = np.array([[0.1], [0.3], [0.5]], dtype=np.float32)
        
        # Call the callback
        self.recorder._audio_callback(mono_data, 3, None, None)
        
        # Verify session call
        session.add_audio_chunk.assert_called_once()
        call_args = session.add_audio_chunk.call_args[0][0]
        expected_mono = np.array([0.1, 0.3, 0.5], dtype=np.float32)
        np.testing.assert_array_equal(call_args, expected_mono)
    
    def test_audio_callback_no_session(self):
        """Test audio callback when no session exists."""
        # This should not raise an exception
        stereo_data = np.array([[0.1, 0.2]], dtype=np.float32)
        self.recorder._audio_callback(stereo_data, 1, None, None)
        # No assertion needed - just verify no exception
    
    def test_cleanup(self):
        """Test recorder cleanup."""
        # Mock a recording session
        mock_session = Mock()
        self.recorder.current_session = mock_session
        
        # Mock a stream
        mock_stream = Mock()
        self.recorder._stream = mock_stream
        
        self.recorder.cleanup()
        
        assert self.recorder.current_session is None
        assert self.recorder._stream is None
        mock_stream.close.assert_called_once()
        mock_session.cleanup.assert_called_once()
    
    def test_context_manager(self):
        """Test using AudioRecorder as context manager."""
        with patch.object(AudioRecorder, 'cleanup') as mock_cleanup:
            with AudioRecorder() as recorder:
                assert isinstance(recorder, AudioRecorder)
            mock_cleanup.assert_called_once()
    
    def test_filename_wav_extension_added(self):
        """Test that .wav extension is added if missing."""
        with patch.object(self.recorder, 'check_blackhole_available', return_value=True):
            with patch.object(self.recorder, 'get_blackhole_device_id', return_value=0):
                with patch('sounddevice.InputStream'):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        filename = os.path.join(temp_dir, "test")  # No extension
                        
                        self.recorder.start_recording(filename)
                        
                        # Should have .wav extension added
                        expected_filename = os.path.join(temp_dir, "test.wav")
                        assert self.recorder.current_session.filename == expected_filename


@pytest.fixture
def sample_audio_file():
    """Create a sample WAV file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        # Create a simple sine wave
        sample_rate = 16000
        duration = 1.0  # 1 second
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to int16 and save as WAV
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        import wave
        with wave.open(f.name, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        yield f.name
        
        # Cleanup
        try:
            os.unlink(f.name)
        except OSError:
            pass


class TestIntegration:
    """Integration tests for AudioRecorder."""
    
    @pytest.mark.skipif(not os.getenv('RUN_INTEGRATION_TESTS'), 
                       reason="Integration tests require RUN_INTEGRATION_TESTS=1")
    def test_full_recording_cycle(self, sample_audio_file):
        """Test a complete recording cycle with real audio devices."""
        recorder = AudioRecorder()
        
        try:
            # Check if BlackHole is available (skip if not)
            if not recorder.check_blackhole_available():
                pytest.skip("BlackHole not available for integration test")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file = os.path.join(temp_dir, "integration_test.wav")
                
                # Start recording
                recorder.start_recording(output_file)
                assert recorder.is_recording()
                
                # Record for a short time
                time.sleep(0.5)
                
                # Stop recording
                result_file = recorder.stop_recording()
                
                # Verify results
                assert result_file == output_file
                assert os.path.exists(output_file)
                assert os.path.getsize(output_file) > 0
                assert not recorder.is_recording()
                
        finally:
            recorder.cleanup()