"""
Tests for FastAPI endpoints in AI Meeting Notes application.

Tests all API endpoints including meeting control, status monitoring,
and result retrieval with proper error handling scenarios.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime

from ai_meeting_notes.main import app
from ai_meeting_notes.models import (
    MeetingSession, TranscriptResult, TranscriptSegment, 
    MeetingNotes, MeetingInfo
)
from ai_meeting_notes.audio_recorder import AudioRecorderError
from ai_meeting_notes.notes_generator import OllamaError


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Mock all service dependencies."""
    with patch('ai_meeting_notes.main.audio_recorder') as mock_audio, \
         patch('ai_meeting_notes.main.transcriber') as mock_transcriber, \
         patch('ai_meeting_notes.main.notes_generator') as mock_notes_gen, \
         patch('ai_meeting_notes.main.file_manager') as mock_file_mgr:
        
        # Configure mocks with default behavior
        mock_audio.check_blackhole_available.return_value = True
        mock_audio.is_recording.return_value = False
        mock_audio.get_recording_duration.return_value = 0.0
        
        mock_transcriber.transcribe_file.return_value = create_mock_transcript()
        
        mock_notes_gen.check_ollama_available.return_value = True
        mock_notes_gen.generate_notes.return_value = create_mock_notes()
        
        mock_file_mgr.check_disk_space.return_value = True
        mock_file_mgr.get_audio_file_path.return_value = "/tmp/test_audio.wav"
        
        yield {
            'audio': mock_audio,
            'transcriber': mock_transcriber,
            'notes_generator': mock_notes_gen,
            'file_manager': mock_file_mgr
        }


def create_mock_transcript():
    """Create a mock transcript result for testing."""
    segments = [
        TranscriptSegment(
            text="Hello everyone, welcome to the meeting.",
            start_time=0.0,
            end_time=3.0,
            confidence=0.95
        ),
        TranscriptSegment(
            text="Let's discuss the project updates.",
            start_time=3.0,
            end_time=6.0,
            confidence=0.88
        )
    ]
    
    return TranscriptResult(
        segments=segments,
        full_text="Hello everyone, welcome to the meeting. Let's discuss the project updates.",
        duration=6.0,
        language="en",
        processing_time=2.5,
        model_name="base",
        audio_file_path="/tmp/test_audio.wav"
    )


def create_mock_notes():
    """Create mock meeting notes for testing."""
    return MeetingNotes(
        meeting_info=MeetingInfo(
            title="Test Meeting",
            duration=6.0
        ),
        participants=["Alice", "Bob"],
        summary="Brief meeting to discuss project updates.",
        agenda_items=[],
        discussion_points=[],
        action_items=[],
        decisions=[]
    )


class TestHealthEndpoint:
    """Test cases for health check endpoint."""
    
    def test_health_check_all_services_healthy(self, client, mock_services):
        """Test health check when all services are available."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["message"] == "All services operational"
        assert data["services"]["audio_recorder"] is True
        assert data["services"]["blackhole_available"] is True
        assert data["services"]["transcriber"] is True
        assert data["services"]["notes_generator"] is True
        assert data["services"]["ollama_available"] is True
        assert data["services"]["file_manager"] is True
        assert data["services"]["disk_space_ok"] is True
    
    def test_health_check_degraded_services(self, client, mock_services):
        """Test health check when some services are unavailable."""
        # Make BlackHole unavailable
        mock_services['audio'].check_blackhole_available.return_value = False
        mock_services['notes_generator'].check_ollama_available.return_value = False
        
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "degraded"
        assert data["message"] == "Some services unavailable"
        assert data["services"]["blackhole_available"] is False
        assert data["services"]["ollama_available"] is False


class TestStatusEndpoint:
    """Test cases for status endpoint."""
    
    def test_status_no_active_session(self, client, mock_services):
        """Test status endpoint when no session is active."""
        with patch('ai_meeting_notes.main.current_session', None):
            response = client.get("/api/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "idle"
            assert data["message"] == "No active meeting session"
            assert data["session_id"] is None
    
    def test_status_with_active_recording(self, client, mock_services):
        """Test status endpoint with active recording session."""
        mock_session = MeetingSession(status="recording")
        mock_services['audio'].is_recording.return_value = True
        mock_services['audio'].get_recording_duration.return_value = 45.5
        
        with patch('ai_meeting_notes.main.current_session', mock_session):
            response = client.get("/api/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "recording"
            assert data["recording_duration"] == 45.5
            assert data["session_id"] is not None
    
    def test_status_with_processing_session(self, client, mock_services):
        """Test status endpoint during processing phases."""
        mock_session = MeetingSession(
            status="transcribing",
            processing_progress=65.0
        )
        
        with patch('ai_meeting_notes.main.current_session', mock_session):
            response = client.get("/api/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "transcribing"
            assert data["processing_progress"] == 65.0


class TestRecordingEndpoints:
    """Test cases for recording control endpoints."""
    
    def test_start_recording_success(self, client, mock_services):
        """Test successful recording start."""
        with patch('ai_meeting_notes.main.current_session', None):
            response = client.post("/api/start-recording")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["message"] == "Recording started successfully"
            assert data["session_id"] is not None
            
            # Verify audio recorder was called
            mock_services['audio'].start_recording.assert_called_once()
    
    def test_start_recording_already_active(self, client, mock_services):
        """Test starting recording when already recording."""
        mock_session = MeetingSession(status="recording")
        
        with patch('ai_meeting_notes.main.current_session', mock_session):
            response = client.post("/api/start-recording")
            
            assert response.status_code == 400
            assert "already in progress" in response.json()["detail"]
    
    def test_start_recording_blackhole_unavailable(self, client, mock_services):
        """Test starting recording when BlackHole is not available."""
        mock_services['audio'].check_blackhole_available.return_value = False
        
        with patch('ai_meeting_notes.main.current_session', None):
            response = client.post("/api/start-recording")
            
            assert response.status_code == 400
            assert "BlackHole" in response.json()["detail"]
    
    def test_start_recording_insufficient_disk_space(self, client, mock_services):
        """Test starting recording with insufficient disk space."""
        mock_services['file_manager'].check_disk_space.return_value = False
        
        with patch('ai_meeting_notes.main.current_session', None):
            response = client.post("/api/start-recording")
            
            assert response.status_code == 400
            assert "disk space" in response.json()["detail"]
    
    def test_start_recording_audio_error(self, client, mock_services):
        """Test starting recording with audio recorder error."""
        mock_services['audio'].start_recording.side_effect = AudioRecorderError("Device busy")
        
        with patch('ai_meeting_notes.main.current_session', None):
            response = client.post("/api/start-recording")
            
            assert response.status_code == 400
            assert "Device busy" in response.json()["detail"]
    
    def test_stop_recording_success(self, client, mock_services):
        """Test successful recording stop."""
        mock_session = MeetingSession(status="recording", audio_file="/tmp/test.wav")
        mock_services['audio'].is_recording.return_value = True
        mock_services['audio'].stop_recording.return_value = "/tmp/test.wav"
        
        with patch('ai_meeting_notes.main.current_session', mock_session):
            response = client.post("/api/stop-recording")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["message"] == "Recording stopped, processing started"
            assert data["audio_file"] == "/tmp/test.wav"
            
            # Verify session status changed to transcribing initially
            # Note: In tests, background processing may complete immediately
            assert mock_session.status in ["transcribing", "generating_notes", "completed"]
    
    def test_stop_recording_no_active_recording(self, client, mock_services):
        """Test stopping recording when none is active."""
        with patch('ai_meeting_notes.main.current_session', None):
            response = client.post("/api/stop-recording")
            
            assert response.status_code == 400
            assert "No active recording" in response.json()["detail"]
    
    def test_stop_recording_audio_error(self, client, mock_services):
        """Test stopping recording with audio error."""
        mock_session = MeetingSession(status="recording")
        mock_services['audio'].is_recording.return_value = True
        mock_services['audio'].stop_recording.side_effect = Exception("Stop failed")
        
        with patch('ai_meeting_notes.main.current_session', mock_session):
            response = client.post("/api/stop-recording")
            
            assert response.status_code == 500
            assert "Stop failed" in response.json()["detail"]
            assert mock_session.status == "error"


class TestResultEndpoints:
    """Test cases for transcript and notes endpoints."""
    
    def test_get_transcript_available(self, client, mock_services):
        """Test getting transcript when available."""
        mock_transcript = create_mock_transcript()
        mock_session = MeetingSession(
            status="completed",
            transcript=mock_transcript
        )
        
        with patch('ai_meeting_notes.main.current_session', mock_session):
            response = client.get("/api/transcript")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["available"] is True
            assert data["message"] == "Transcript available"
            assert data["transcript"]["full_text"] == mock_transcript.full_text
    
    def test_get_transcript_in_progress(self, client, mock_services):
        """Test getting transcript while transcription is in progress."""
        mock_session = MeetingSession(
            status="transcribing",
            processing_progress=45.0
        )
        
        with patch('ai_meeting_notes.main.current_session', mock_session):
            response = client.get("/api/transcript")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["available"] is False
            assert "45%" in data["message"]
            assert data["transcript"] is None
    
    def test_get_transcript_no_session(self, client, mock_services):
        """Test getting transcript when no session exists."""
        with patch('ai_meeting_notes.main.current_session', None):
            response = client.get("/api/transcript")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["available"] is False
            assert data["message"] == "No active session"
    
    def test_get_transcript_error_state(self, client, mock_services):
        """Test getting transcript when session is in error state."""
        mock_session = MeetingSession(
            status="error",
            error_message="Transcription failed"
        )
        
        with patch('ai_meeting_notes.main.current_session', mock_session):
            response = client.get("/api/transcript")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["available"] is False
            assert "Transcription failed" in data["message"]
    
    def test_get_notes_available(self, client, mock_services):
        """Test getting notes when available."""
        mock_notes = create_mock_notes()
        mock_session = MeetingSession(
            status="completed",
            notes=mock_notes
        )
        
        with patch('ai_meeting_notes.main.current_session', mock_session):
            response = client.get("/api/notes")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["available"] is True
            assert data["message"] == "Meeting notes available"
            assert data["notes"]["summary"] == mock_notes.summary
    
    def test_get_notes_generating(self, client, mock_services):
        """Test getting notes while generation is in progress."""
        mock_session = MeetingSession(
            status="generating_notes",
            processing_progress=75.0
        )
        
        with patch('ai_meeting_notes.main.current_session', mock_session):
            response = client.get("/api/notes")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["available"] is False
            assert "75%" in data["message"]
    
    def test_get_notes_no_transcript(self, client, mock_services):
        """Test getting notes when transcript is not available."""
        mock_session = MeetingSession(status="transcribing")
        
        with patch('ai_meeting_notes.main.current_session', mock_session):
            response = client.get("/api/notes")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["available"] is False
            assert "Transcript not available" in data["message"]


class TestClearEndpoint:
    """Test cases for session clear endpoint."""
    
    def test_clear_session_success(self, client, mock_services):
        """Test successful session clearing."""
        mock_session = MeetingSession(status="completed")
        mock_services['audio'].is_recording.return_value = False
        
        with patch('ai_meeting_notes.main.current_session', mock_session) as mock_current:
            response = client.post("/api/clear")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["message"] == "Session cleared"
            
            # Verify cleanup was called
            mock_services['file_manager'].cleanup_session_files.assert_called_once_with(mock_session)
    
    def test_clear_session_with_active_recording(self, client, mock_services):
        """Test clearing session while recording is active."""
        mock_session = MeetingSession(status="recording")
        mock_services['audio'].is_recording.return_value = True
        
        with patch('ai_meeting_notes.main.current_session', mock_session):
            response = client.post("/api/clear")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            
            # Verify recording was stopped
            mock_services['audio'].stop_recording.assert_called_once()
    
    def test_clear_session_cleanup_error(self, client, mock_services):
        """Test clearing session when cleanup fails."""
        mock_session = MeetingSession(status="completed")
        mock_services['file_manager'].cleanup_session_files.side_effect = Exception("Cleanup failed")
        
        with patch('ai_meeting_notes.main.current_session', mock_session):
            response = client.post("/api/clear")
            
            # Should still succeed even if cleanup fails
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestErrorHandling:
    """Test cases for error handling scenarios."""
    
    def test_service_not_initialized_errors(self, client):
        """Test endpoints when services are not initialized."""
        with patch('ai_meeting_notes.main.audio_recorder', None), \
             patch('ai_meeting_notes.main.current_session', None):
            
            response = client.post("/api/start-recording")
            assert response.status_code == 500
            assert "not initialized" in response.json()["detail"]
    
    def test_concurrent_request_handling(self, client, mock_services):
        """Test handling of concurrent requests."""
        # This would require more complex async testing
        # For now, we verify that the endpoints handle state correctly
        mock_session = MeetingSession(status="recording")
        
        with patch('ai_meeting_notes.main.current_session', mock_session):
            # Try to start recording while already recording
            response1 = client.post("/api/start-recording")
            assert response1.status_code == 400
            
            # Status should still work
            response2 = client.get("/api/status")
            assert response2.status_code == 200


class TestBackgroundProcessing:
    """Test cases for background processing functionality."""
    
    @pytest.mark.asyncio
    async def test_process_meeting_audio_success(self, mock_services):
        """Test successful background processing of meeting audio."""
        from ai_meeting_notes.main import process_meeting_audio
        
        mock_session = MeetingSession(
            status="recording",
            audio_file="/tmp/test.wav"
        )
        
        mock_transcript = create_mock_transcript()
        mock_notes = create_mock_notes()
        
        mock_services['transcriber'].transcribe_file.return_value = mock_transcript
        mock_services['notes_generator'].generate_notes.return_value = mock_notes
        
        with patch('ai_meeting_notes.main.transcriber', mock_services['transcriber']), \
             patch('ai_meeting_notes.main.notes_generator', mock_services['notes_generator']), \
             patch('ai_meeting_notes.main.file_manager', mock_services['file_manager']):
            
            await process_meeting_audio(mock_session)
            
            assert mock_session.status == "completed"
            assert mock_session.transcript == mock_transcript
            assert mock_session.notes == mock_notes
    
    @pytest.mark.asyncio
    async def test_process_meeting_audio_transcription_error(self, mock_services):
        """Test background processing with transcription error."""
        from ai_meeting_notes.main import process_meeting_audio
        
        mock_session = MeetingSession(
            status="recording",
            audio_file="/tmp/test.wav"
        )
        
        mock_services['transcriber'].transcribe_file.side_effect = Exception("Transcription failed")
        
        with patch('ai_meeting_notes.main.transcriber', mock_services['transcriber']):
            await process_meeting_audio(mock_session)
            
            assert mock_session.status == "error"
            assert "Transcription failed" in mock_session.error_message
    
    @pytest.mark.asyncio
    async def test_process_meeting_audio_notes_error(self, mock_services):
        """Test background processing with notes generation error."""
        from ai_meeting_notes.main import process_meeting_audio
        
        mock_session = MeetingSession(
            status="recording",
            audio_file="/tmp/test.wav"
        )
        
        mock_transcript = create_mock_transcript()
        mock_services['transcriber'].transcribe_file.return_value = mock_transcript
        mock_services['notes_generator'].generate_notes.side_effect = OllamaError("Notes generation failed")
        
        with patch('ai_meeting_notes.main.transcriber', mock_services['transcriber']), \
             patch('ai_meeting_notes.main.notes_generator', mock_services['notes_generator']):
            
            await process_meeting_audio(mock_session)
            
            assert mock_session.status == "error"
            assert "Notes generation failed" in mock_session.error_message
            # Transcript should still be available
            assert mock_session.transcript == mock_transcript


if __name__ == "__main__":
    pytest.main([__file__, "-v"])