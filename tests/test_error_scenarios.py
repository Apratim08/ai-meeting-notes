"""
Tests for specific error scenarios and failure modes.

This module tests common real-world failure scenarios that users might encounter,
ensuring proper error handling, user-friendly messages, and recovery options.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from ai_meeting_notes.processing_pipeline import ProcessingPipeline
from ai_meeting_notes.models import MeetingSession, TranscriptResult, TranscriptSegment
from ai_meeting_notes.audio_recorder import AudioRecorderError
from ai_meeting_notes.notes_generator import OllamaError
from ai_meeting_notes.error_handling import ProcessingError, ErrorCategory


class TestAudioDeviceFailures:
    """Test audio device related failure scenarios."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return {
            'audio_recorder': Mock(),
            'transcriber': Mock(),
            'notes_generator': Mock(),
            'file_manager': Mock()
        }
    
    @pytest.fixture
    def pipeline(self, mock_services):
        """Create ProcessingPipeline with mocked services."""
        return ProcessingPipeline(
            audio_recorder=mock_services['audio_recorder'],
            transcriber=mock_services['transcriber'],
            notes_generator=mock_services['notes_generator'],
            file_manager=mock_services['file_manager']
        )
    
    @pytest.mark.asyncio
    async def test_blackhole_not_installed(self, pipeline, mock_services):
        """Test scenario where BlackHole is not installed."""
        mock_services['audio_recorder'].check_blackhole_available.return_value = False
        mock_services['file_manager'].check_disk_space.return_value = (True, 10.0, "")
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline.start_recording("test_session")
        
        error = exc_info.value
        assert error.category == ErrorCategory.AUDIO_DEVICE
        assert "blackhole" in error.user_message.lower()
        assert len(error.recovery_actions) > 0
        
        # Check that recovery actions include installation guidance
        action_descriptions = [action.description for action in error.recovery_actions]
        assert any("install" in desc.lower() for desc in action_descriptions)
    
    @pytest.mark.asyncio
    async def test_audio_device_permission_denied(self, pipeline, mock_services):
        """Test scenario where audio device access is denied."""
        mock_services['audio_recorder'].check_blackhole_available.return_value = True
        mock_services['file_manager'].check_disk_space.return_value = (True, 10.0, "")
        mock_services['audio_recorder'].start_recording.side_effect = AudioRecorderError(
            "Permission denied: microphone access not granted"
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline.start_recording("test_session")
        
        error = exc_info.value
        assert "permission" in error.user_message.lower() or "access" in error.user_message.lower()
    
    @pytest.mark.asyncio
    async def test_audio_device_busy(self, pipeline, mock_services):
        """Test scenario where audio device is busy/in use."""
        mock_services['audio_recorder'].check_blackhole_available.return_value = True
        mock_services['file_manager'].check_disk_space.return_value = (True, 10.0, "")
        mock_services['audio_recorder'].start_recording.side_effect = AudioRecorderError(
            "Device busy: audio device is already in use"
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline.start_recording("test_session")
        
        error = exc_info.value
        assert "busy" in error.user_message.lower() or "in use" in error.user_message.lower()


class TestFileSystemFailures:
    """Test file system related failure scenarios."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return {
            'audio_recorder': Mock(),
            'transcriber': Mock(),
            'notes_generator': Mock(),
            'file_manager': Mock()
        }
    
    @pytest.fixture
    def pipeline(self, mock_services):
        """Create ProcessingPipeline with mocked services."""
        return ProcessingPipeline(
            audio_recorder=mock_services['audio_recorder'],
            transcriber=mock_services['transcriber'],
            notes_generator=mock_services['notes_generator'],
            file_manager=mock_services['file_manager']
        )
    
    @pytest.mark.asyncio
    async def test_insufficient_disk_space(self, pipeline, mock_services):
        """Test scenario with insufficient disk space."""
        mock_services['audio_recorder'].check_blackhole_available.return_value = True
        mock_services['file_manager'].check_disk_space.return_value = (
            False, 0.1, "Only 0.1GB available, need at least 1.0GB"
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline.start_recording("test_session")
        
        error = exc_info.value
        assert error.category == ErrorCategory.SYSTEM_RESOURCE
        assert "disk space" in error.user_message.lower()
    
    @pytest.mark.asyncio
    async def test_file_permission_error(self, pipeline, mock_services):
        """Test scenario where file cannot be created due to permissions."""
        mock_services['audio_recorder'].check_blackhole_available.return_value = True
        mock_services['file_manager'].check_disk_space.return_value = (True, 10.0, "")
        mock_services['file_manager'].get_audio_file_path.return_value = "/readonly/path/audio.wav"
        mock_services['audio_recorder'].start_recording.side_effect = PermissionError(
            "Permission denied: cannot write to /readonly/path/audio.wav"
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline.start_recording("test_session")
        
        error = exc_info.value
        assert error.category == ErrorCategory.FILE_SYSTEM
        assert "permission" in error.user_message.lower()
    
    @pytest.mark.asyncio
    async def test_audio_file_corrupted(self, pipeline, mock_services):
        """Test scenario where audio file becomes corrupted."""
        session = MeetingSession(audio_file="corrupted.wav")
        
        with patch('pathlib.Path.exists', return_value=True):
            mock_services['transcriber'].transcribe_file.side_effect = Exception(
                "Audio file is corrupted or unreadable"
            )
            
            # Mock the transcribe_with_progress_monitoring method
            async def mock_transcribe(audio_file):
                raise Exception("Audio file is corrupted or unreadable")
            
            pipeline._transcribe_with_progress_monitoring = mock_transcribe
            
            with pytest.raises(ProcessingError) as exc_info:
                await pipeline._transcription_phase(session)
            
            error = exc_info.value
            assert "corrupted" in error.user_message.lower() or "unreadable" in error.user_message.lower()
    
    @pytest.mark.asyncio
    async def test_audio_file_deleted_during_processing(self, pipeline, mock_services):
        """Test scenario where audio file is deleted during processing."""
        session = MeetingSession(audio_file="deleted.wav")
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline._transcription_phase(session)
        
        error = exc_info.value
        assert error.category == ErrorCategory.FILE_SYSTEM
        assert "missing" in error.user_message.lower() or "not found" in error.user_message.lower()


class TestTranscriptionFailures:
    """Test transcription related failure scenarios."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return {
            'audio_recorder': Mock(),
            'transcriber': Mock(),
            'notes_generator': Mock(),
            'file_manager': Mock()
        }
    
    @pytest.fixture
    def pipeline(self, mock_services):
        """Create ProcessingPipeline with mocked services."""
        return ProcessingPipeline(
            audio_recorder=mock_services['audio_recorder'],
            transcriber=mock_services['transcriber'],
            notes_generator=mock_services['notes_generator'],
            file_manager=mock_services['file_manager']
        )
    
    @pytest.mark.asyncio
    async def test_whisper_model_not_available(self, pipeline, mock_services):
        """Test scenario where Whisper model is not available."""
        session = MeetingSession(audio_file="test.wav")
        
        with patch('pathlib.Path.exists', return_value=True):
            async def mock_transcribe(audio_file):
                raise Exception("WhisperModel not found: base model not available")
            
            pipeline._transcribe_with_progress_monitoring = mock_transcribe
            
            with pytest.raises(ProcessingError) as exc_info:
                await pipeline._transcription_phase(session)
            
            error = exc_info.value
            assert error.category == ErrorCategory.TRANSCRIPTION
            assert "model" in error.user_message.lower()
    
    @pytest.mark.asyncio
    async def test_transcription_timeout(self, pipeline, mock_services):
        """Test scenario where transcription times out."""
        session = MeetingSession(audio_file="long_audio.wav")
        
        with patch('pathlib.Path.exists', return_value=True):
            async def mock_transcribe(audio_file):
                raise asyncio.TimeoutError("Transcription timed out after 300 seconds")
            
            pipeline._transcribe_with_progress_monitoring = mock_transcribe
            
            with pytest.raises(ProcessingError) as exc_info:
                await pipeline._transcription_phase(session)
            
            error = exc_info.value
            assert "timeout" in error.user_message.lower() or "time" in error.user_message.lower()
    
    @pytest.mark.asyncio
    async def test_silent_audio_transcription(self, pipeline, mock_services):
        """Test scenario where audio contains no speech."""
        session = MeetingSession(audio_file="silent.wav")
        
        with patch('pathlib.Path.exists', return_value=True):
            # Mock empty transcription result
            empty_result = TranscriptResult(
                segments=[],
                full_text="",
                duration=60.0,
                language="en",
                processing_time=5.0,
                model_name="base",
                audio_file_path="silent.wav"
            )
            
            async def mock_transcribe(audio_file):
                return empty_result
            
            pipeline._transcribe_with_progress_monitoring = mock_transcribe
            
            with pytest.raises(ProcessingError) as exc_info:
                await pipeline._transcription_phase(session)
            
            error = exc_info.value
            assert error.category == ErrorCategory.TRANSCRIPTION
            assert "no speech" in error.user_message.lower()
    
    @pytest.mark.asyncio
    async def test_low_quality_audio_transcription(self, pipeline, mock_services):
        """Test scenario with very low quality audio transcription."""
        session = MeetingSession(audio_file="low_quality.wav")
        
        with patch('pathlib.Path.exists', return_value=True):
            # Mock low confidence transcription result
            low_confidence_result = TranscriptResult(
                segments=[
                    TranscriptSegment(
                        text="unintelligible mumbling",
                        start_time=0.0,
                        end_time=5.0,
                        confidence=0.2,
                        is_uncertain=True
                    )
                ],
                full_text="unintelligible mumbling",
                duration=60.0,
                language="en",
                processing_time=5.0,
                model_name="base",
                audio_file_path="low_quality.wav"
            )
            
            async def mock_transcribe(audio_file):
                return low_confidence_result
            
            pipeline._transcribe_with_progress_monitoring = mock_transcribe
            
            # This should succeed but with warnings
            await pipeline._transcription_phase(session)
            
            assert session.transcript is not None
            assert session.transcript.average_confidence < 0.5


class TestNotesGenerationFailures:
    """Test notes generation related failure scenarios."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return {
            'audio_recorder': Mock(),
            'transcriber': Mock(),
            'notes_generator': Mock(),
            'file_manager': Mock()
        }
    
    @pytest.fixture
    def pipeline(self, mock_services):
        """Create ProcessingPipeline with mocked services."""
        return ProcessingPipeline(
            audio_recorder=mock_services['audio_recorder'],
            transcriber=mock_services['transcriber'],
            notes_generator=mock_services['notes_generator'],
            file_manager=mock_services['file_manager']
        )
    
    @pytest.fixture
    def session_with_transcript(self):
        """Create a session with a valid transcript."""
        transcript = TranscriptResult(
            segments=[
                TranscriptSegment(
                    text="Welcome to our team meeting. Let's discuss the project status.",
                    start_time=0.0,
                    end_time=5.0,
                    confidence=0.9
                ),
                TranscriptSegment(
                    text="We need to complete the documentation by Friday.",
                    start_time=5.0,
                    end_time=10.0,
                    confidence=0.85
                )
            ],
            full_text="Welcome to our team meeting. Let's discuss the project status. We need to complete the documentation by Friday.",
            duration=300.0,  # 5 minutes
            language="en",
            processing_time=15.0,
            model_name="base",
            audio_file_path="meeting.wav"
        )
        return MeetingSession(transcript=transcript, audio_file="meeting.wav")
    
    @pytest.mark.asyncio
    async def test_ollama_not_running(self, pipeline, mock_services, session_with_transcript):
        """Test scenario where Ollama service is not running."""
        mock_services['notes_generator'].check_ollama_available.return_value = False
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline._notes_generation_phase(session_with_transcript)
        
        error = exc_info.value
        assert error.category == ErrorCategory.NOTES_GENERATION
        assert "ollama" in error.user_message.lower()
        
        # Check recovery actions include starting Ollama
        action_descriptions = [action.description for action in error.recovery_actions]
        assert any("ollama" in desc.lower() for desc in action_descriptions)
    
    @pytest.mark.asyncio
    async def test_ollama_model_not_available(self, pipeline, mock_services, session_with_transcript):
        """Test scenario where required Ollama model is not available."""
        mock_services['notes_generator'].check_ollama_available.return_value = False
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline._notes_generation_phase(session_with_transcript)
        
        error = exc_info.value
        assert error.category == ErrorCategory.NOTES_GENERATION
        assert "model" in error.user_message.lower() or "ollama" in error.user_message.lower()
    
    @pytest.mark.asyncio
    async def test_ollama_connection_timeout(self, pipeline, mock_services, session_with_transcript):
        """Test scenario where Ollama connection times out."""
        mock_services['notes_generator'].check_ollama_available.return_value = True
        
        async def mock_generate_notes(transcript, duration):
            raise OllamaError("Ollama API request timed out")
        
        pipeline._generate_notes_with_progress_monitoring = mock_generate_notes
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline._notes_generation_phase(session_with_transcript)
        
        error = exc_info.value
        assert error.category == ErrorCategory.NOTES_GENERATION
        assert "timeout" in error.user_message.lower() or "connection" in error.user_message.lower()
    
    @pytest.mark.asyncio
    async def test_transcript_too_short_for_notes(self, pipeline, mock_services):
        """Test scenario where transcript is too short for meaningful notes."""
        # Create session with very short transcript
        short_transcript = TranscriptResult(
            segments=[
                TranscriptSegment(
                    text="Hi",
                    start_time=0.0,
                    end_time=1.0,
                    confidence=0.9
                )
            ],
            full_text="Hi",
            duration=5.0,
            language="en",
            processing_time=1.0,
            model_name="base",
            audio_file_path="short.wav"
        )
        session = MeetingSession(transcript=short_transcript)
        
        mock_services['notes_generator'].check_ollama_available.return_value = True
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline._notes_generation_phase(session)
        
        error = exc_info.value
        assert error.category == ErrorCategory.TRANSCRIPTION
        assert "too short" in error.user_message.lower()


class TestNetworkFailures:
    """Test network related failure scenarios."""
    
    @pytest.mark.asyncio
    async def test_ollama_network_connection_lost(self):
        """Test scenario where network connection to Ollama is lost."""
        from ai_meeting_notes.notes_generator import NotesGenerator
        from unittest.mock import patch
        import requests
        
        notes_generator = NotesGenerator()
        
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection lost")
            
            with pytest.raises(OllamaError) as exc_info:
                notes_generator._call_ollama_api("test prompt")
            
            assert "connection" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_ollama_server_error(self):
        """Test scenario where Ollama server returns an error."""
        from ai_meeting_notes.notes_generator import NotesGenerator
        from unittest.mock import patch, Mock
        
        notes_generator = NotesGenerator()
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal server error"
            mock_post.return_value = mock_response
            
            with pytest.raises(OllamaError) as exc_info:
                notes_generator._call_ollama_api("test prompt")
            
            assert "500" in str(exc_info.value)


class TestRecoveryScenarios:
    """Test error recovery scenarios."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return {
            'audio_recorder': Mock(),
            'transcriber': Mock(),
            'notes_generator': Mock(),
            'file_manager': Mock()
        }
    
    @pytest.fixture
    def pipeline(self, mock_services):
        """Create ProcessingPipeline with mocked services."""
        return ProcessingPipeline(
            audio_recorder=mock_services['audio_recorder'],
            transcriber=mock_services['transcriber'],
            notes_generator=mock_services['notes_generator'],
            file_manager=mock_services['file_manager']
        )
    
    @pytest.mark.asyncio
    async def test_successful_transcription_retry(self, pipeline, mock_services):
        """Test successful retry after transcription failure."""
        session = MeetingSession(audio_file="test.wav", status="error")
        
        # Mock successful transcription on retry
        transcript_result = TranscriptResult(
            segments=[],
            full_text="Successful transcription",
            duration=60.0,
            language="en",
            processing_time=5.0,
            model_name="base",
            audio_file_path="test.wav"
        )
        
        with patch('pathlib.Path.exists', return_value=True):
            async def mock_transcribe(audio_file):
                return transcript_result
            
            pipeline._transcribe_with_progress_monitoring = mock_transcribe
            
            # Mock successful notes generation
            from ai_meeting_notes.models import MeetingNotes, MeetingInfo
            mock_notes = MeetingNotes(
                meeting_info=MeetingInfo(),
                summary="Test meeting summary"
            )
            
            async def mock_generate_notes(transcript, duration):
                return mock_notes
            
            pipeline._generate_notes_with_progress_monitoring = mock_generate_notes
            
            # Retry should succeed
            await pipeline.retry_transcription(session)
            
            assert session.status == "completed"
            assert session.transcript is not None
            assert session.notes is not None
    
    @pytest.mark.asyncio
    async def test_successful_notes_generation_retry(self, pipeline, mock_services):
        """Test successful retry after notes generation failure."""
        # Create session with transcript but no notes
        transcript = TranscriptResult(
            segments=[],
            full_text="Test meeting transcript",
            duration=60.0,
            language="en",
            processing_time=5.0,
            model_name="base",
            audio_file_path="test.wav"
        )
        session = MeetingSession(transcript=transcript, status="error")
        
        # Mock successful notes generation on retry
        from ai_meeting_notes.models import MeetingNotes, MeetingInfo
        mock_notes = MeetingNotes(
            meeting_info=MeetingInfo(),
            summary="Test meeting summary"
        )
        
        async def mock_generate_notes(transcript, duration):
            return mock_notes
        
        pipeline._generate_notes_with_progress_monitoring = mock_generate_notes
        
        # Retry should succeed
        await pipeline.retry_notes_generation(session)
        
        assert session.status == "completed"
        assert session.notes is not None


if __name__ == "__main__":
    pytest.main([__file__])