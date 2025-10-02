"""
Integration tests for the sequential processing pipeline.

Tests the complete workflow: audio recording → transcription → notes generation
with error handling, progress tracking, and intermediate result preservation.
"""

import asyncio
import os
import tempfile
import wave
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import pytest
import numpy as np

from ai_meeting_notes.processing_pipeline import ProcessingPipeline, ProcessingError
from ai_meeting_notes.models import MeetingSession, TranscriptResult, TranscriptSegment, MeetingNotes, MeetingInfo
from ai_meeting_notes.audio_recorder import AudioRecorder, AudioRecorderError
from ai_meeting_notes.transcription import BatchTranscriber
from ai_meeting_notes.notes_generator import NotesGenerator, OllamaError
from ai_meeting_notes.file_manager import FileManager
from ai_meeting_notes.config import AppConfig


class TestProcessingPipeline:
    """Test suite for the ProcessingPipeline class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_audio_recorder(self):
        """Create a mock audio recorder."""
        recorder = Mock(spec=AudioRecorder)
        recorder.check_blackhole_available.return_value = True
        recorder.is_recording.return_value = False
        recorder.get_recording_duration.return_value = 30.0
        recorder.start_recording = Mock()
        recorder.stop_recording.return_value = "/path/to/audio.wav"
        return recorder
    
    @pytest.fixture
    def mock_transcriber(self):
        """Create a mock batch transcriber."""
        transcriber = Mock(spec=BatchTranscriber)
        transcriber.estimate_processing_time.return_value = 5.0
        transcriber.get_progress.return_value = 0.5
        
        # Mock transcript result
        mock_segment = TranscriptSegment(
            text="This is a test meeting transcript.",
            start_time=0.0,
            end_time=5.0,
            confidence=0.95
        )
        mock_result = TranscriptResult(
            segments=[mock_segment],
            full_text="This is a test meeting transcript.",
            duration=30.0,
            language="en",
            processing_time=5.0,
            model_name="base",
            audio_file_path="/path/to/audio.wav"
        )
        transcriber.transcribe_file.return_value = mock_result
        return transcriber
    
    @pytest.fixture
    def mock_notes_generator(self):
        """Create a mock notes generator."""
        generator = Mock(spec=NotesGenerator)
        generator.check_ollama_available.return_value = True
        
        # Mock meeting notes
        mock_notes = MeetingNotes(
            meeting_info=MeetingInfo(title="Test Meeting"),
            summary="Test meeting summary",
            participants=["Alice", "Bob"],
            agenda_items=[],
            discussion_points=[],
            action_items=[],
            decisions=[]
        )
        generator.generate_notes.return_value = mock_notes
        return generator
    
    @pytest.fixture
    def mock_file_manager(self, temp_dir):
        """Create a mock file manager."""
        manager = Mock(spec=FileManager)
        manager.check_disk_space.return_value = (True, 10.0, "")
        manager.get_audio_file_path.return_value = str(temp_dir / "test_audio.wav")
        manager.cleanup_session_files.return_value = True
        manager.check_file_size_warning.return_value = None  # No size warning
        return manager
    
    @pytest.fixture
    def progress_callback(self):
        """Create a mock progress callback."""
        return Mock()
    
    @pytest.fixture
    def pipeline(self, mock_audio_recorder, mock_transcriber, mock_notes_generator, 
                 mock_file_manager, progress_callback):
        """Create a processing pipeline with mocked dependencies."""
        return ProcessingPipeline(
            audio_recorder=mock_audio_recorder,
            transcriber=mock_transcriber,
            notes_generator=mock_notes_generator,
            file_manager=mock_file_manager,
            progress_callback=progress_callback
        )
    
    def create_test_audio_file(self, filepath: str, duration: float = 1.0) -> str:
        """Create a test WAV audio file."""
        sample_rate = 16000
        frames = int(sample_rate * duration)
        
        # Generate sine wave test audio
        t = np.linspace(0, duration, frames)
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Write WAV file
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return filepath
    
    @pytest.mark.asyncio
    async def test_start_recording_success(self, pipeline, mock_file_manager):
        """Test successful recording start."""
        session_id = "test_session_123"
        
        session = await pipeline.start_recording(session_id)
        
        assert session is not None
        assert session.status == "recording"
        assert session.audio_file is not None
        
        # Verify audio recorder was called
        pipeline.audio_recorder.start_recording.assert_called_once()
        
        # Verify file manager was called
        mock_file_manager.get_audio_file_path.assert_called_once_with(session_id)
    
    @pytest.mark.asyncio
    async def test_start_recording_blackhole_unavailable(self, pipeline):
        """Test recording start failure when BlackHole is unavailable."""
        pipeline.audio_recorder.check_blackhole_available.return_value = False
        
        with pytest.raises(ProcessingError, match="BlackHole audio device not found"):
            await pipeline.start_recording("test_session")
    
    @pytest.mark.asyncio
    async def test_start_recording_insufficient_disk_space(self, pipeline, mock_file_manager):
        """Test recording start failure due to insufficient disk space."""
        mock_file_manager.check_disk_space.return_value = (False, 0.1, "Low disk space")
        
        with pytest.raises(ProcessingError, match="Insufficient disk space"):
            await pipeline.start_recording("test_session")
    
    @pytest.mark.asyncio
    async def test_start_recording_already_recording(self, pipeline):
        """Test recording start failure when already recording."""
        # Start first recording
        await pipeline.start_recording("session1")
        
        # Try to start second recording
        with pytest.raises(ProcessingError, match="Recording already in progress"):
            await pipeline.start_recording("session2")
    
    @pytest.mark.asyncio
    async def test_stop_recording_and_process_success(self, pipeline, temp_dir):
        """Test successful recording stop and processing initiation."""
        # Start recording first
        session_id = "test_session"
        await pipeline.start_recording(session_id)
        
        # Mock that recording is active
        pipeline.audio_recorder.is_recording.return_value = True
        
        # Create test audio file
        audio_file = str(temp_dir / "test_audio.wav")
        self.create_test_audio_file(audio_file)
        pipeline.audio_recorder.stop_recording.return_value = audio_file
        
        # Stop recording and process
        session = await pipeline.stop_recording_and_process()
        
        assert session is not None
        assert session.audio_file == audio_file
        
        # Verify audio recorder was called
        pipeline.audio_recorder.stop_recording.assert_called_once()
        
        # Give processing a moment to start
        await asyncio.sleep(0.1)
        
        # Should be processing
        assert pipeline.is_processing()
    
    @pytest.mark.asyncio
    async def test_stop_recording_no_active_recording(self, pipeline):
        """Test stop recording failure when no recording is active."""
        with pytest.raises(ProcessingError, match="No active recording to stop"):
            await pipeline.stop_recording_and_process()
    
    @pytest.mark.asyncio
    async def test_complete_processing_pipeline_success(self, pipeline, temp_dir, progress_callback):
        """Test complete processing pipeline from start to finish."""
        # Create test audio file
        audio_file = str(temp_dir / "test_audio.wav")
        self.create_test_audio_file(audio_file, duration=2.0)
        
        # Start recording
        session_id = "integration_test"
        session = await pipeline.start_recording(session_id)
        
        # Mock recording state
        pipeline.audio_recorder.is_recording.return_value = True
        pipeline.audio_recorder.stop_recording.return_value = audio_file
        
        # Stop recording and process
        session = await pipeline.stop_recording_and_process()
        
        # Wait for processing to complete
        max_wait = 10  # seconds
        wait_time = 0
        while pipeline.is_processing() and wait_time < max_wait:
            await asyncio.sleep(0.1)
            wait_time += 0.1
        
        # Verify final state
        final_session = pipeline.get_current_session()
        assert final_session is not None
        assert final_session.status == "completed"
        assert final_session.transcript is not None
        assert final_session.notes is not None
        assert final_session.processing_progress == 100.0
        
        # Verify progress callback was called
        assert progress_callback.call_count > 0
        
        # Verify all processing steps were called
        pipeline.transcriber.transcribe_file.assert_called_once_with(audio_file)
        pipeline.notes_generator.generate_notes.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transcription_failure_with_preservation(self, pipeline, temp_dir):
        """Test transcription failure with intermediate result preservation."""
        # Create test audio file
        audio_file = str(temp_dir / "test_audio.wav")
        self.create_test_audio_file(audio_file)
        
        # Start recording
        session = await pipeline.start_recording("test_session")
        
        # Mock recording state
        pipeline.audio_recorder.is_recording.return_value = True
        pipeline.audio_recorder.stop_recording.return_value = audio_file
        
        # Make transcription fail
        pipeline.transcriber.transcribe_file.side_effect = Exception("Transcription failed")
        
        # Stop recording and process
        session = await pipeline.stop_recording_and_process()
        
        # Wait for processing to complete
        max_wait = 5
        wait_time = 0
        while pipeline.is_processing() and wait_time < max_wait:
            await asyncio.sleep(0.1)
            wait_time += 0.1
        
        # Verify error state
        final_session = pipeline.get_current_session()
        assert final_session is not None
        assert final_session.status == "error"
        assert "Transcription failed" in final_session.error_message
        
        # Verify audio file is preserved
        assert final_session.audio_file == audio_file
        assert Path(audio_file).exists()
    
    @pytest.mark.asyncio
    async def test_notes_generation_failure_with_preservation(self, pipeline, temp_dir):
        """Test notes generation failure with transcript preservation."""
        # Create test audio file
        audio_file = str(temp_dir / "test_audio.wav")
        self.create_test_audio_file(audio_file)
        
        # Start recording
        session = await pipeline.start_recording("test_session")
        
        # Mock recording state
        pipeline.audio_recorder.is_recording.return_value = True
        pipeline.audio_recorder.stop_recording.return_value = audio_file
        
        # Make notes generation fail
        pipeline.notes_generator.generate_notes.side_effect = OllamaError("Notes generation failed")
        
        # Stop recording and process
        session = await pipeline.stop_recording_and_process()
        
        # Wait for processing to complete
        max_wait = 5
        wait_time = 0
        while pipeline.is_processing() and wait_time < max_wait:
            await asyncio.sleep(0.1)
            wait_time += 0.1
        
        # Verify error state
        final_session = pipeline.get_current_session()
        assert final_session is not None
        assert final_session.status == "error"
        assert "Notes generation failed" in final_session.error_message
        
        # Verify transcript is preserved
        assert final_session.transcript is not None
        assert final_session.audio_file == audio_file
    
    @pytest.mark.asyncio
    async def test_retry_transcription_success(self, pipeline, temp_dir):
        """Test successful transcription retry after failure."""
        # Create test audio file
        audio_file = str(temp_dir / "test_audio.wav")
        self.create_test_audio_file(audio_file)
        
        # Create a session with failed transcription
        session = MeetingSession(
            status="error",
            audio_file=audio_file,
            error_message="Transcription failed"
        )
        pipeline._current_session = session
        
        # Reset transcriber to succeed on retry
        pipeline.transcriber.transcribe_file.side_effect = None
        
        # Retry transcription
        await pipeline.retry_transcription(session)
        
        # Wait for processing
        max_wait = 5
        wait_time = 0
        while pipeline.is_processing() and wait_time < max_wait:
            await asyncio.sleep(0.1)
            wait_time += 0.1
        
        # Verify success
        assert session.status == "completed"
        assert session.transcript is not None
        assert session.notes is not None
    
    @pytest.mark.asyncio
    async def test_retry_notes_generation_success(self, pipeline):
        """Test successful notes generation retry after failure."""
        # Create a session with successful transcription but failed notes
        mock_transcript = TranscriptResult(
            segments=[],
            full_text="Test transcript",
            duration=30.0,
            language="en",
            processing_time=5.0,
            model_name="base",
            audio_file_path="/path/to/audio.wav"
        )
        
        session = MeetingSession(
            status="error",
            transcript=mock_transcript,
            error_message="Notes generation failed"
        )
        pipeline._current_session = session
        
        # Reset notes generator to succeed on retry
        pipeline.notes_generator.generate_notes.side_effect = None
        
        # Retry notes generation
        await pipeline.retry_notes_generation(session)
        
        # Wait for processing
        max_wait = 5
        wait_time = 0
        while pipeline.is_processing() and wait_time < max_wait:
            await asyncio.sleep(0.1)
            wait_time += 0.1
        
        # Verify success
        assert session.status == "completed"
        assert session.notes is not None
    
    def test_clear_session(self, pipeline):
        """Test session clearing functionality."""
        # Create a mock session
        session = MeetingSession(status="recording")
        pipeline._current_session = session
        pipeline._is_processing = True
        
        # Mock active recording
        pipeline.audio_recorder.is_recording.return_value = True
        
        # Clear session
        pipeline.clear_session()
        
        # Verify cleanup
        assert pipeline.get_current_session() is None
        assert not pipeline.is_processing()
        pipeline.audio_recorder.stop_recording.assert_called_once()
        pipeline.file_manager.cleanup_session_files.assert_called_once_with(session)
    
    def test_get_processing_status_no_session(self, pipeline):
        """Test processing status when no session exists."""
        status = pipeline.get_processing_status()
        
        assert not status["has_session"]
        assert status["status"] == "idle"
        assert status["message"] == "No active session"
    
    def test_get_processing_status_with_session(self, pipeline):
        """Test processing status with active session."""
        session = MeetingSession(
            status="transcribing",
            processing_progress=50.0
        )
        pipeline._current_session = session
        pipeline._is_processing = True
        
        status = pipeline.get_processing_status()
        
        assert status["has_session"]
        assert status["status"] == "transcribing"
        assert status["progress"] == 50.0
        assert status["is_processing"]
        assert status["has_audio"] is False  # No audio file set
        assert status["has_transcript"] is False
        assert status["has_notes"] is False


class TestProcessingPipelineIntegration:
    """Integration tests with real components (where possible)."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def real_file_manager(self, temp_dir):
        """Create a real file manager for integration testing."""
        return FileManager(
            base_dir=str(temp_dir),
            max_file_size_mb=100.0,
            auto_cleanup_days=1
        )
    
    def create_test_audio_file(self, filepath: str, duration: float = 1.0) -> str:
        """Create a test WAV audio file."""
        sample_rate = 16000
        frames = int(sample_rate * duration)
        
        # Generate sine wave test audio
        t = np.linspace(0, duration, frames)
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Write WAV file
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return filepath
    
    def test_file_manager_integration(self, real_file_manager, temp_dir):
        """Test file manager integration with real file operations."""
        # Test audio file path generation
        session_id = "integration_test_123"
        audio_path = real_file_manager.get_audio_file_path(session_id)
        
        assert session_id in audio_path
        assert audio_path.endswith('.wav')
        
        # Create test audio file
        self.create_test_audio_file(audio_path, duration=2.0)
        
        # Test file size checking
        size_mb = real_file_manager.get_file_size_mb(audio_path)
        assert size_mb > 0
        
        # Test disk space checking
        has_space, free_gb, warning = real_file_manager.check_disk_space()
        assert has_space  # Should have space in temp directory
        assert free_gb > 0
        
        # Test file cleanup
        session = MeetingSession(audio_file=audio_path)
        cleanup_success = real_file_manager.cleanup_session_files(session, preserve_on_error=False)
        assert cleanup_success
        assert not Path(audio_path).exists()
    
    def test_storage_info_integration(self, real_file_manager, temp_dir):
        """Test storage information gathering."""
        # Create some test files
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        test_files = []
        for i in range(3):
            audio_file = str(audio_dir / f"test_{i}.wav")
            self.create_test_audio_file(audio_file, duration=1.0)
            test_files.append(audio_file)
        
        # Get storage info
        storage_info = real_file_manager.get_storage_info()
        
        assert "has_sufficient_space" in storage_info
        assert "free_space_gb" in storage_info
        assert "audio_files_count" in storage_info
        assert "total_audio_size_mb" in storage_info
        
        # Should detect our test files
        assert storage_info["total_audio_size_mb"] > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])