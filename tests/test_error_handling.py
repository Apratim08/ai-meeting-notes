"""
Comprehensive tests for error handling and recovery mechanisms.

Tests cover common failure modes, retry logic, error categorization,
user-friendly error messages, and graceful degradation scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

from ai_meeting_notes.error_handling import (
    ProcessingError, ErrorCategory, ErrorSeverity, ErrorContext,
    ErrorRecoveryManager, RetryConfig, handle_processing_error,
    preserve_intermediate_results, error_recovery_manager
)
from ai_meeting_notes.models import MeetingSession, TranscriptResult, TranscriptSegment
from ai_meeting_notes.processing_pipeline import ProcessingPipeline
from ai_meeting_notes.audio_recorder import AudioRecorderError
from ai_meeting_notes.notes_generator import OllamaError


class TestProcessingError:
    """Test ProcessingError class functionality."""
    
    def test_processing_error_creation(self):
        """Test creating a ProcessingError with all parameters."""
        context = ErrorContext(
            timestamp=datetime.now(),
            component="test_component",
            operation="test_operation"
        )
        
        error = ProcessingError(
            message="Test error",
            category=ErrorCategory.TRANSCRIPTION,
            severity=ErrorSeverity.HIGH,
            context=context,
            user_message="User-friendly error message",
            technical_details="Technical details here"
        )
        
        assert error.message == "Test error"
        assert error.category == ErrorCategory.TRANSCRIPTION
        assert error.severity == ErrorSeverity.HIGH
        assert error.user_message == "User-friendly error message"
        assert error.technical_details == "Technical details here"
        assert error.context == context
    
    def test_processing_error_to_dict(self):
        """Test converting ProcessingError to dictionary."""
        error = ProcessingError(
            message="Test error",
            category=ErrorCategory.AUDIO_DEVICE,
            severity=ErrorSeverity.MEDIUM
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["message"] == "Test error"
        assert error_dict["category"] == "audio_device"
        assert error_dict["severity"] == "medium"
        assert "timestamp" in error_dict
        assert "recovery_actions" in error_dict


class TestRetryConfig:
    """Test RetryConfig functionality."""
    
    def test_retry_config_defaults(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_backoff is True
        assert config.jitter is True
    
    def test_get_delay_exponential_backoff(self):
        """Test delay calculation with exponential backoff."""
        config = RetryConfig(base_delay=2.0, exponential_backoff=True, jitter=False)
        
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0
        assert config.get_delay(3) == 8.0
    
    def test_get_delay_linear_backoff(self):
        """Test delay calculation with linear backoff."""
        config = RetryConfig(base_delay=2.0, exponential_backoff=False, jitter=False)
        
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 2.0
        assert config.get_delay(3) == 2.0
    
    def test_get_delay_max_delay_limit(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(base_delay=10.0, max_delay=15.0, exponential_backoff=True, jitter=False)
        
        assert config.get_delay(1) == 10.0
        assert config.get_delay(2) == 15.0  # Capped at max_delay
        assert config.get_delay(3) == 15.0  # Still capped


class TestErrorRecoveryManager:
    """Test ErrorRecoveryManager functionality."""
    
    @pytest.fixture
    def recovery_manager(self):
        """Create a fresh ErrorRecoveryManager for testing."""
        manager = ErrorRecoveryManager()
        manager.clear_error_history()
        return manager
    
    def test_handle_generic_exception(self, recovery_manager):
        """Test handling a generic exception."""
        exception = ValueError("Test value error")
        context = ErrorContext(
            timestamp=datetime.now(),
            component="test",
            operation="test_op"
        )
        
        processing_error = recovery_manager.handle_error(exception, context)
        
        assert isinstance(processing_error, ProcessingError)
        assert processing_error.message == "Test value error"
        assert processing_error.original_exception == exception
        assert len(recovery_manager.error_history) == 1
    
    def test_handle_processing_error(self, recovery_manager):
        """Test handling an existing ProcessingError."""
        original_error = ProcessingError(
            message="Original error",
            category=ErrorCategory.TRANSCRIPTION
        )
        
        result = recovery_manager.handle_error(original_error)
        
        assert result == original_error
        assert len(recovery_manager.error_history) == 1
    
    def test_categorize_audio_device_error(self, recovery_manager):
        """Test categorization of audio device errors."""
        error = Exception("BlackHole device not found")
        
        processing_error = recovery_manager.handle_error(error)
        
        assert processing_error.category == ErrorCategory.AUDIO_DEVICE
        assert processing_error.severity == ErrorSeverity.HIGH
    
    def test_categorize_transcription_error(self, recovery_manager):
        """Test categorization of transcription errors."""
        error = Exception("Whisper model failed to load")
        
        processing_error = recovery_manager.handle_error(error)
        
        assert processing_error.category == ErrorCategory.TRANSCRIPTION
        assert processing_error.severity == ErrorSeverity.MEDIUM
    
    def test_categorize_ollama_error(self, recovery_manager):
        """Test categorization of Ollama errors."""
        error = OllamaError("Ollama connection failed")
        
        processing_error = recovery_manager.handle_error(error)
        
        assert processing_error.category == ErrorCategory.NOTES_GENERATION
        assert processing_error.severity == ErrorSeverity.MEDIUM
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self, recovery_manager):
        """Test successful retry operation."""
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"
        
        config = RetryConfig(max_attempts=3, base_delay=0.1, jitter=False)
        result = await recovery_manager.retry_with_backoff(
            failing_operation,
            "test_operation",
            config
        )
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_all_failures(self, recovery_manager):
        """Test retry operation that fails all attempts."""
        call_count = 0
        
        async def always_failing_operation():
            nonlocal call_count
            call_count += 1
            raise Exception(f"Failure {call_count}")
        
        config = RetryConfig(max_attempts=2, base_delay=0.1)
        
        with pytest.raises(ProcessingError):
            await recovery_manager.retry_with_backoff(
                always_failing_operation,
                "test_operation",
                config
            )
        
        assert call_count == 2
    
    def test_get_error_summary(self, recovery_manager):
        """Test getting error summary."""
        # Add some errors
        for i in range(5):
            error = Exception(f"Error {i}")
            recovery_manager.handle_error(error)
        
        summary = recovery_manager.get_error_summary()
        
        assert summary["total_errors"] == 5
        assert summary["recent_errors"] == 5
        assert "category_breakdown" in summary
        assert "severity_breakdown" in summary
        assert "last_error" in summary


class TestErrorHandlingIntegration:
    """Test error handling integration with processing pipeline."""
    
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
    async def test_start_recording_blackhole_error(self, pipeline, mock_services):
        """Test start recording with BlackHole not available."""
        mock_services['audio_recorder'].check_blackhole_available.return_value = False
        mock_services['file_manager'].check_disk_space.return_value = (True, 10.0, "")
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline.start_recording("test_session")
        
        error = exc_info.value
        assert error.category == ErrorCategory.AUDIO_DEVICE
        assert error.severity == ErrorSeverity.HIGH
        assert "BlackHole" in error.user_message
    
    @pytest.mark.asyncio
    async def test_start_recording_disk_space_error(self, pipeline, mock_services):
        """Test start recording with insufficient disk space."""
        mock_services['audio_recorder'].check_blackhole_available.return_value = True
        mock_services['file_manager'].check_disk_space.return_value = (False, 0.1, "Low disk space")
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline.start_recording("test_session")
        
        error = exc_info.value
        assert error.category == ErrorCategory.SYSTEM_RESOURCE
        assert error.severity == ErrorSeverity.HIGH
        assert "disk space" in error.user_message.lower()
    
    @pytest.mark.asyncio
    async def test_transcription_phase_missing_file(self, pipeline, mock_services):
        """Test transcription phase with missing audio file."""
        session = MeetingSession(audio_file="/nonexistent/file.wav")
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline._transcription_phase(session)
        
        error = exc_info.value
        assert error.category == ErrorCategory.FILE_SYSTEM
        assert error.severity == ErrorSeverity.HIGH
        assert "missing" in error.user_message.lower()
    
    @pytest.mark.asyncio
    async def test_transcription_phase_empty_result(self, pipeline, mock_services):
        """Test transcription phase with empty transcription result."""
        session = MeetingSession(audio_file="test.wav")
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            # Mock empty transcription result
            empty_result = TranscriptResult(
                segments=[],
                full_text="",
                duration=60.0,
                language="en",
                processing_time=5.0,
                model_name="base",
                audio_file_path="test.wav"
            )
            
            async def mock_transcribe():
                return empty_result
            
            pipeline._transcribe_with_progress_monitoring = mock_transcribe
            
            with pytest.raises(ProcessingError) as exc_info:
                await pipeline._transcription_phase(session)
            
            error = exc_info.value
            assert error.category == ErrorCategory.TRANSCRIPTION
            assert "no speech" in error.user_message.lower()
    
    @pytest.mark.asyncio
    async def test_notes_generation_phase_no_transcript(self, pipeline):
        """Test notes generation phase with no transcript."""
        session = MeetingSession()
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline._notes_generation_phase(session)
        
        error = exc_info.value
        assert error.category == ErrorCategory.TRANSCRIPTION
        assert error.severity == ErrorSeverity.HIGH
        assert "transcript" in error.user_message.lower()
    
    @pytest.mark.asyncio
    async def test_notes_generation_phase_ollama_unavailable(self, pipeline, mock_services):
        """Test notes generation phase with Ollama unavailable."""
        # Create session with transcript
        transcript = TranscriptResult(
            segments=[TranscriptSegment(
                text="This is a test meeting",
                start_time=0.0,
                end_time=5.0,
                confidence=0.9
            )],
            full_text="This is a test meeting",
            duration=60.0,
            language="en",
            processing_time=5.0,
            model_name="base",
            audio_file_path="test.wav"
        )
        session = MeetingSession(transcript=transcript)
        
        mock_services['notes_generator'].check_ollama_available.return_value = False
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline._notes_generation_phase(session)
        
        error = exc_info.value
        assert error.category == ErrorCategory.NOTES_GENERATION
        assert error.severity == ErrorSeverity.HIGH
        assert "ollama" in error.user_message.lower()
    
    @pytest.mark.asyncio
    async def test_notes_generation_phase_short_transcript(self, pipeline, mock_services):
        """Test notes generation phase with very short transcript."""
        # Create session with very short transcript
        transcript = TranscriptResult(
            segments=[TranscriptSegment(
                text="Hi",
                start_time=0.0,
                end_time=1.0,
                confidence=0.9
            )],
            full_text="Hi",
            duration=5.0,
            language="en",
            processing_time=1.0,
            model_name="base",
            audio_file_path="test.wav"
        )
        session = MeetingSession(transcript=transcript)
        
        mock_services['notes_generator'].check_ollama_available.return_value = True
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline._notes_generation_phase(session)
        
        error = exc_info.value
        assert error.category == ErrorCategory.TRANSCRIPTION
        assert error.severity == ErrorSeverity.MEDIUM
        assert "too short" in error.user_message.lower()


class TestPreserveIntermediateResults:
    """Test intermediate result preservation functionality."""
    
    def test_preserve_results_with_audio_file(self):
        """Test preserving results when audio file exists."""
        session = MeetingSession(audio_file="test.wav")
        error = ProcessingError("Test error")
        
        with patch('pathlib.Path.exists', return_value=True):
            preserve_intermediate_results(session, error)
        
        assert session.status == "error"
        assert session.error_message == error.user_message
        assert error.context.additional_data is not None
        assert "preserved_results" in error.context.additional_data
    
    def test_preserve_results_with_transcript(self):
        """Test preserving results when transcript exists."""
        transcript = TranscriptResult(
            segments=[],
            full_text="Test transcript content",
            duration=60.0,
            language="en",
            processing_time=5.0,
            model_name="base",
            audio_file_path="test.wav"
        )
        session = MeetingSession(transcript=transcript)
        error = ProcessingError("Test error")
        
        preserve_intermediate_results(session, error)
        
        assert session.status == "error"
        assert "Transcript" in str(error.context.additional_data["preserved_results"])


class TestRecoveryActions:
    """Test recovery action functionality."""
    
    @pytest.fixture
    def pipeline_with_session(self, mock_services):
        """Create pipeline with a session in error state."""
        pipeline = ProcessingPipeline(
            audio_recorder=mock_services['audio_recorder'],
            transcriber=mock_services['transcriber'],
            notes_generator=mock_services['notes_generator'],
            file_manager=mock_services['file_manager']
        )
        
        # Create session in error state
        session = MeetingSession(
            status="error",
            audio_file="test.wav",
            error_message="Test error"
        )
        pipeline._current_session = session
        
        return pipeline, session
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return {
            'audio_recorder': Mock(),
            'transcriber': Mock(),
            'notes_generator': Mock(),
            'file_manager': Mock()
        }
    
    def test_get_recovery_options_audio_only(self, pipeline_with_session):
        """Test recovery options when only audio file exists."""
        pipeline, session = pipeline_with_session
        
        with patch('pathlib.Path.exists', return_value=True):
            options = pipeline._get_recovery_options(session)
        
        option_actions = [opt["action"] for opt in options]
        assert "clear_session" in option_actions
        assert "retry_transcription" in option_actions
        assert "retry_full_pipeline" in option_actions
    
    def test_get_recovery_options_with_transcript(self, pipeline_with_session):
        """Test recovery options when transcript exists."""
        pipeline, session = pipeline_with_session
        
        # Add transcript to session
        session.transcript = TranscriptResult(
            segments=[],
            full_text="Test transcript",
            duration=60.0,
            language="en",
            processing_time=5.0,
            model_name="base",
            audio_file_path="test.wav"
        )
        
        with patch('pathlib.Path.exists', return_value=True):
            options = pipeline._get_recovery_options(session)
        
        option_actions = [opt["action"] for opt in options]
        assert "retry_notes_generation" in option_actions
    
    @pytest.mark.asyncio
    async def test_execute_recovery_clear_session(self, pipeline_with_session):
        """Test executing clear session recovery action."""
        pipeline, session = pipeline_with_session
        
        result = await pipeline.execute_recovery_action("clear_session")
        
        assert result["success"] is True
        assert "cleared" in result["message"].lower()
        assert pipeline._current_session is None
    
    @pytest.mark.asyncio
    async def test_execute_recovery_unknown_action(self, pipeline_with_session):
        """Test executing unknown recovery action."""
        pipeline, session = pipeline_with_session
        
        result = await pipeline.execute_recovery_action("unknown_action")
        
        assert result["success"] is False
        assert "unknown" in result["message"].lower()


class TestGracefulDegradation:
    """Test graceful degradation scenarios."""
    
    @pytest.mark.asyncio
    async def test_partial_transcription_success(self):
        """Test handling partial transcription success."""
        # This would test scenarios where transcription partially succeeds
        # but with low confidence or missing segments
        pass
    
    @pytest.mark.asyncio
    async def test_notes_generation_minimal_content(self):
        """Test handling notes generation with minimal content."""
        # This would test scenarios where notes generation succeeds
        # but produces minimal or low-quality output
        pass
    
    def test_service_unavailable_fallback(self):
        """Test fallback behavior when services are unavailable."""
        # This would test fallback mechanisms when external services
        # like Ollama are temporarily unavailable
        pass


if __name__ == "__main__":
    pytest.main([__file__])