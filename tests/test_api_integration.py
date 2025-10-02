"""
Integration tests for API endpoints with the processing pipeline.

Tests the complete API workflow with the sequential processing pipeline.
"""

import asyncio
import tempfile
import wave
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import pytest
import numpy as np
from fastapi.testclient import TestClient

from ai_meeting_notes.main import app
from ai_meeting_notes.processing_pipeline import ProcessingPipeline
from ai_meeting_notes.models import TranscriptResult, TranscriptSegment, MeetingNotes, MeetingInfo
from ai_meeting_notes.audio_recorder import AudioRecorder
from ai_meeting_notes.transcription import BatchTranscriber
from ai_meeting_notes.notes_generator import NotesGenerator
from ai_meeting_notes.file_manager import FileManager


class TestAPIIntegration:
    """Integration tests for API endpoints with processing pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_services(self, temp_dir):
        """Create mock services for testing."""
        # Mock audio recorder
        audio_recorder = Mock(spec=AudioRecorder)
        audio_recorder.check_blackhole_available.return_value = True
        audio_recorder.is_recording.return_value = False
        audio_recorder.get_recording_duration.return_value = 30.0
        audio_recorder.start_recording = Mock()
        audio_recorder.stop_recording.return_value = str(temp_dir / "test_audio.wav")
        
        # Mock transcriber
        transcriber = Mock(spec=BatchTranscriber)
        transcriber.estimate_processing_time.return_value = 5.0
        transcriber.get_progress.return_value = 1.0
        
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
            audio_file_path=str(temp_dir / "test_audio.wav")
        )
        transcriber.transcribe_file.return_value = mock_result
        
        # Mock notes generator
        notes_generator = Mock(spec=NotesGenerator)
        notes_generator.check_ollama_available.return_value = True
        
        mock_notes = MeetingNotes(
            meeting_info=MeetingInfo(title="Test Meeting"),
            summary="Test meeting summary",
            participants=["Alice", "Bob"],
            agenda_items=[],
            discussion_points=[],
            action_items=[],
            decisions=[]
        )
        notes_generator.generate_notes.return_value = mock_notes
        
        # Mock file manager
        file_manager = Mock(spec=FileManager)
        file_manager.check_disk_space.return_value = (True, 10.0, "")
        file_manager.get_audio_file_path.return_value = str(temp_dir / "test_audio.wav")
        file_manager.cleanup_session_files.return_value = True
        
        return {
            'audio_recorder': audio_recorder,
            'transcriber': transcriber,
            'notes_generator': notes_generator,
            'file_manager': file_manager
        }
    
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
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_health_check_endpoint(self, mock_pipeline, client, mock_services):
        """Test health check endpoint."""
        # Mock the pipeline and services
        mock_pipeline.audio_recorder = mock_services['audio_recorder']
        mock_pipeline.transcriber = mock_services['transcriber']
        mock_pipeline.notes_generator = mock_services['notes_generator']
        mock_pipeline.file_manager = mock_services['file_manager']
        
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] in ["healthy", "degraded"]
        assert "services" in data
        assert "message" in data
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_status_endpoint_no_session(self, mock_pipeline, client):
        """Test status endpoint when no session exists."""
        mock_pipeline.get_processing_status.return_value = {
            "has_session": False,
            "status": "idle",
            "message": "No active session"
        }
        
        response = client.get("/api/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "idle"
        assert data["message"] == "No active meeting session"
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_status_endpoint_with_session(self, mock_pipeline, client):
        """Test status endpoint with active session."""
        mock_pipeline.get_processing_status.return_value = {
            "has_session": True,
            "status": "recording",
            "progress": 0.0,
            "message": "Recording in progress...",
            "error_message": None,
            "recording_duration": 15.5
        }
        
        # Mock get_current_session to return a session object
        mock_session = Mock()
        mock_pipeline.get_current_session.return_value = mock_session
        
        response = client.get("/api/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "recording"
        assert data["recording_duration"] == 15.5
        assert data["processing_progress"] == 0.0
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_start_recording_success(self, mock_pipeline, client):
        """Test successful recording start via API."""
        # Mock successful recording start
        mock_session = Mock()
        mock_session.status = "recording"
        mock_session.audio_file = "/path/to/audio.wav"
        
        # Use AsyncMock for async method
        async def mock_start_recording(session_id):
            return mock_session
        
        mock_pipeline.start_recording = AsyncMock(side_effect=mock_start_recording)
        
        response = client.post("/api/start-recording")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "session_id" in data
        assert "Recording started successfully" in data["message"]
        
        # Verify pipeline was called
        mock_pipeline.start_recording.assert_called_once()
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_start_recording_failure(self, mock_pipeline, client):
        """Test recording start failure via API."""
        from ai_meeting_notes.processing_pipeline import ProcessingError
        
        # Mock recording failure
        mock_pipeline.start_recording.side_effect = ProcessingError("BlackHole not found")
        
        response = client.post("/api/start-recording")
        
        assert response.status_code == 400
        data = response.json()
        
        assert "BlackHole not found" in data["detail"]
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_stop_recording_success(self, mock_pipeline, client):
        """Test successful recording stop via API."""
        # Mock successful recording stop
        mock_session = Mock()
        mock_session.audio_file = "/path/to/audio.wav"
        
        # Use AsyncMock for async method
        async def mock_stop_recording():
            return mock_session
        
        mock_pipeline.stop_recording_and_process = AsyncMock(side_effect=mock_stop_recording)
        
        response = client.post("/api/stop-recording")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["audio_file"] == "/path/to/audio.wav"
        assert "processing started" in data["message"]
        
        # Verify pipeline was called
        mock_pipeline.stop_recording_and_process.assert_called_once()
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_stop_recording_failure(self, mock_pipeline, client):
        """Test recording stop failure via API."""
        from ai_meeting_notes.processing_pipeline import ProcessingError
        
        # Mock stop failure
        mock_pipeline.stop_recording_and_process.side_effect = ProcessingError("No active recording")
        
        response = client.post("/api/stop-recording")
        
        assert response.status_code == 400
        data = response.json()
        
        assert "No active recording" in data["detail"]
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_get_transcript_available(self, mock_pipeline, client):
        """Test getting available transcript via API."""
        # Mock session with transcript
        mock_transcript = TranscriptResult(
            segments=[],
            full_text="Test transcript text",
            duration=30.0,
            language="en",
            processing_time=5.0,
            model_name="base",
            audio_file_path="/path/to/audio.wav"
        )
        
        mock_session = Mock()
        mock_session.transcript = mock_transcript
        mock_pipeline.get_current_session.return_value = mock_session
        
        response = client.get("/api/transcript")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["available"] is True
        assert data["transcript"]["full_text"] == "Test transcript text"
        assert data["message"] == "Transcript available"
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_get_transcript_in_progress(self, mock_pipeline, client):
        """Test getting transcript while transcription is in progress."""
        # Mock session with transcription in progress
        mock_session = Mock()
        mock_session.transcript = None
        mock_session.status = "transcribing"
        mock_session.processing_progress = 45.0
        mock_pipeline.get_current_session.return_value = mock_session
        
        response = client.get("/api/transcript")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["available"] is False
        assert "45%" in data["message"]
        assert "in progress" in data["message"]
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_get_notes_available(self, mock_pipeline, client):
        """Test getting available notes via API."""
        # Mock session with notes
        mock_notes = MeetingNotes(
            meeting_info=MeetingInfo(title="Test Meeting"),
            summary="Test summary",
            participants=["Alice"],
            agenda_items=[],
            discussion_points=[],
            action_items=[],
            decisions=[]
        )
        
        mock_session = Mock()
        mock_session.notes = mock_notes
        mock_pipeline.get_current_session.return_value = mock_session
        
        response = client.get("/api/notes")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["available"] is True
        assert data["notes"]["summary"] == "Test summary"
        assert data["message"] == "Meeting notes available"
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_get_notes_generating(self, mock_pipeline, client):
        """Test getting notes while generation is in progress."""
        # Mock session with notes generation in progress
        mock_session = Mock()
        mock_session.notes = None
        mock_session.status = "generating_notes"
        mock_session.processing_progress = 75.0
        mock_session.transcript = Mock()  # Has transcript
        mock_pipeline.get_current_session.return_value = mock_session
        
        response = client.get("/api/notes")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["available"] is False
        assert "75%" in data["message"]
        assert "Generating notes" in data["message"]
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_clear_session_success(self, mock_pipeline, client):
        """Test successful session clearing via API."""
        mock_pipeline.clear_session.return_value = None
        
        response = client.post("/api/clear")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["message"] == "Session cleared"
        
        # Verify pipeline was called
        mock_pipeline.clear_session.assert_called_once()
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_retry_transcription_success(self, mock_pipeline, client):
        """Test successful transcription retry via API."""
        # Mock session in error state
        mock_session = Mock()
        mock_session.status = "error"
        mock_session.audio_file = "/path/to/audio.wav"
        mock_pipeline.get_current_session.return_value = mock_session
        
        # Mock successful retry with AsyncMock
        mock_pipeline.retry_transcription = AsyncMock(return_value=None)
        
        response = client.post("/api/retry-transcription")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "retry started" in data["message"]
        
        # Verify pipeline was called
        mock_pipeline.retry_transcription.assert_called_once_with(mock_session)
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_retry_transcription_no_session(self, mock_pipeline, client):
        """Test transcription retry with no active session."""
        mock_pipeline.get_current_session.return_value = None
        
        response = client.post("/api/retry-transcription")
        
        assert response.status_code == 400
        data = response.json()
        
        assert "No active session" in data["detail"]
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_retry_notes_success(self, mock_pipeline, client):
        """Test successful notes generation retry via API."""
        # Mock session with transcript but failed notes
        mock_session = Mock()
        mock_session.transcript = Mock()  # Has transcript
        mock_pipeline.get_current_session.return_value = mock_session
        
        # Mock successful retry with AsyncMock
        mock_pipeline.retry_notes_generation = AsyncMock(return_value=None)
        
        response = client.post("/api/retry-notes")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "retry started" in data["message"]
        
        # Verify pipeline was called
        mock_pipeline.retry_notes_generation.assert_called_once_with(mock_session)
    
    @patch('ai_meeting_notes.main.processing_pipeline')
    def test_retry_notes_no_transcript(self, mock_pipeline, client):
        """Test notes retry with no transcript available."""
        # Mock session without transcript
        mock_session = Mock()
        mock_session.transcript = None
        mock_pipeline.get_current_session.return_value = mock_session
        
        response = client.post("/api/retry-notes")
        
        assert response.status_code == 400
        data = response.json()
        
        assert "No transcript available" in data["detail"]
    
    def test_complete_workflow_simulation(self, client, temp_dir):
        """Test complete workflow simulation with mocked pipeline."""
        with patch('ai_meeting_notes.main.processing_pipeline') as mock_pipeline:
            # Setup mock pipeline responses
            mock_session = Mock()
            mock_session.status = "recording"
            mock_session.audio_file = str(temp_dir / "test.wav")
            
            # Create test audio file
            self.create_test_audio_file(mock_session.audio_file)
            
            # Mock start recording
            mock_pipeline.start_recording.return_value = mock_session
            mock_pipeline.get_processing_status.return_value = {
                "has_session": True,
                "status": "recording",
                "progress": 0.0,
                "message": "Recording in progress...",
                "error_message": None,
                "recording_duration": 0.0
            }
            mock_pipeline.get_current_session.return_value = mock_session
            
            # 1. Start recording
            response = client.post("/api/start-recording")
            assert response.status_code == 200
            
            # 2. Check status
            response = client.get("/api/status")
            assert response.status_code == 200
            assert response.json()["status"] == "recording"
            
            # 3. Stop recording and start processing
            mock_session.status = "transcribing"
            mock_pipeline.stop_recording_and_process.return_value = mock_session
            
            response = client.post("/api/stop-recording")
            assert response.status_code == 200
            
            # 4. Simulate transcription completion
            mock_transcript = TranscriptResult(
                segments=[],
                full_text="Test meeting transcript",
                duration=30.0,
                language="en",
                processing_time=5.0,
                model_name="base",
                audio_file_path=mock_session.audio_file
            )
            mock_session.transcript = mock_transcript
            mock_session.status = "generating_notes"
            
            response = client.get("/api/transcript")
            assert response.status_code == 200
            assert response.json()["available"] is True
            
            # 5. Simulate notes completion
            mock_notes = MeetingNotes(
                meeting_info=MeetingInfo(title="Test Meeting"),
                summary="Test summary",
                participants=["Alice"],
                agenda_items=[],
                discussion_points=[],
                action_items=[],
                decisions=[]
            )
            mock_session.notes = mock_notes
            mock_session.status = "completed"
            
            response = client.get("/api/notes")
            assert response.status_code == 200
            assert response.json()["available"] is True
            
            # 6. Clear session
            response = client.post("/api/clear")
            assert response.status_code == 200


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])