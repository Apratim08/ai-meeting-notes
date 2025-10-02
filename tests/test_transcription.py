import os
import pytest
import tempfile
import wave
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from ai_meeting_notes.transcription import BatchTranscriber
from ai_meeting_notes.models import TranscriptResult, TranscriptSegment


class TestBatchTranscriber:
    """Test suite for BatchTranscriber class."""
    
    @pytest.fixture
    def transcriber(self):
        """Create a BatchTranscriber instance for testing."""
        return BatchTranscriber(model_size="tiny", device="cpu")
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create a temporary WAV file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Create a simple 1-second audio file
            sample_rate = 16000
            duration = 1.0
            samples = int(sample_rate * duration)
            
            # Generate a simple sine wave
            frequency = 440  # A4 note
            t = np.linspace(0, duration, samples, False)
            audio_data = np.sin(2 * np.pi * frequency * t)
            
            # Convert to 16-bit integers
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            yield tmp_file.name
            
            # Cleanup
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass
    
    def test_init(self):
        """Test BatchTranscriber initialization."""
        transcriber = BatchTranscriber(model_size="base", device="cpu")
        assert transcriber.model_size == "base"
        assert transcriber.device == "cpu"
        assert transcriber.model is None
        assert transcriber._current_progress == 0.0
        assert not transcriber._is_processing
    
    def test_get_audio_duration(self, transcriber, sample_audio_file):
        """Test audio duration calculation."""
        duration = transcriber._get_audio_duration(sample_audio_file)
        assert abs(duration - 1.0) < 0.1  # Should be approximately 1 second
    
    def test_get_audio_duration_invalid_file(self, transcriber):
        """Test audio duration with invalid file."""
        duration = transcriber._get_audio_duration("nonexistent.wav")
        assert duration == 0.0
    
    def test_estimate_processing_time(self, transcriber):
        """Test processing time estimation."""
        # Test with different model sizes
        transcriber.model_size = "tiny"
        time_tiny = transcriber.estimate_processing_time(60.0)  # 1 minute
        assert time_tiny == 3.0  # 60 * 0.05
        
        transcriber.model_size = "base"
        time_base = transcriber.estimate_processing_time(60.0)
        assert time_base == 6.0  # 60 * 0.1
        
        transcriber.model_size = "large"
        time_large = transcriber.estimate_processing_time(60.0)
        assert time_large == 24.0  # 60 * 0.4
    
    def test_get_progress(self, transcriber):
        """Test progress tracking."""
        assert transcriber.get_progress() == 0.0
        
        transcriber._current_progress = 0.5
        assert transcriber.get_progress() == 0.5
        
        transcriber._current_progress = 1.0
        assert transcriber.get_progress() == 1.0
    
    def test_is_processing(self, transcriber):
        """Test processing status."""
        assert not transcriber.is_processing()
        
        transcriber._is_processing = True
        assert transcriber.is_processing()
        
        transcriber._is_processing = False
        assert not transcriber.is_processing()
    
    def test_transcribe_file_not_found(self, transcriber):
        """Test transcription with non-existent file."""
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe_file("nonexistent.wav")
    
    @patch('ai_meeting_notes.transcription.WhisperModel')
    def test_load_model_success(self, mock_whisper_model, transcriber):
        """Test successful model loading."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        result = transcriber._load_model("base")
        
        mock_whisper_model.assert_called_once_with("base", device="cpu")
        assert result == mock_model
    
    @patch('ai_meeting_notes.transcription.WhisperModel')
    def test_load_model_failure(self, mock_whisper_model, transcriber):
        """Test model loading failure."""
        mock_whisper_model.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            transcriber._load_model("base")
    
    @patch('ai_meeting_notes.transcription.WhisperModel')
    def test_transcribe_with_model_success(self, mock_whisper_model, transcriber, sample_audio_file):
        """Test successful transcription with mocked model."""
        # Mock the Whisper model
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        # Mock transcription segments
        mock_segment1 = Mock()
        mock_segment1.text = "Hello world"
        mock_segment1.start = 0.0
        mock_segment1.end = 1.0
        mock_segment1.avg_logprob = -0.5
        
        mock_segment2 = Mock()
        mock_segment2.text = "This is a test"
        mock_segment2.start = 1.0
        mock_segment2.end = 2.5
        mock_segment2.avg_logprob = -1.0
        
        mock_segments = [mock_segment1, mock_segment2]
        
        # Mock transcription info
        mock_info = Mock()
        mock_info.language = "en"
        
        mock_model.transcribe.return_value = (mock_segments, mock_info)
        
        # Perform transcription
        result = transcriber._transcribe_with_model(sample_audio_file, "tiny")
        
        # Verify result
        assert isinstance(result, TranscriptResult)
        assert len(result.segments) == 2
        assert result.full_text == "Hello world This is a test"
        assert result.language == "en"
        assert result.model_name == "tiny"
        assert result.audio_file_path == sample_audio_file
        
        # Check segments
        seg1 = result.segments[0]
        assert seg1.text == "Hello world"
        assert seg1.start_time == 0.0
        assert seg1.end_time == 1.0
        assert seg1.confidence > 0.0
        
        seg2 = result.segments[1]
        assert seg2.text == "This is a test"
        assert seg2.start_time == 1.0
        assert seg2.end_time == 2.5
        assert seg2.confidence > 0.0
    
    @patch('ai_meeting_notes.transcription.WhisperModel')
    def test_transcribe_with_low_confidence(self, mock_whisper_model, transcriber, sample_audio_file):
        """Test transcription with low confidence segments."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        # Mock segment with low confidence
        mock_segment = Mock()
        mock_segment.text = "Uncertain text"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.avg_logprob = -5.0  # Very low confidence
        
        mock_info = Mock()
        mock_info.language = "en"
        
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        result = transcriber._transcribe_with_model(sample_audio_file, "tiny")
        
        # Check that low confidence is detected
        assert len(result.segments) == 1
        segment = result.segments[0]
        assert segment.is_uncertain  # Should be marked as uncertain
        assert segment.confidence < 0.7
    
    @patch('ai_meeting_notes.transcription.WhisperModel')
    def test_transcribe_file_with_fallback(self, mock_whisper_model, transcriber, sample_audio_file):
        """Test transcription with fallback to different model."""
        # First model fails
        mock_whisper_model.side_effect = [
            Exception("Primary model failed"),  # First call fails
            Mock()  # Second call succeeds
        ]
        
        # Mock successful transcription on retry
        mock_model = Mock()
        mock_segment = Mock()
        mock_segment.text = "Fallback success"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.avg_logprob = -0.5
        
        mock_info = Mock()
        mock_info.language = "en"
        
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_model.side_effect = [Exception("Primary failed"), mock_model]
        
        # Set up transcriber with fallback models
        transcriber.model_size = "medium"  # This will fail
        transcriber.fallback_models = ["base", "tiny"]
        
        with patch.object(transcriber, '_transcribe_with_model') as mock_transcribe:
            # First call fails, second succeeds
            mock_transcribe.side_effect = [
                Exception("Primary failed"),
                TranscriptResult(
                    segments=[TranscriptSegment(
                        text="Fallback success",
                        start_time=0.0,
                        end_time=1.0,
                        confidence=0.8
                    )],
                    full_text="Fallback success",
                    duration=1.0,
                    language="en",
                    processing_time=1.0,
                    model_name="base",
                    audio_file_path=sample_audio_file
                )
            ]
            
            result = transcriber.transcribe_file(sample_audio_file)
            
            # Should have tried primary model first, then fallback
            assert mock_transcribe.call_count == 2
            assert result.full_text == "Fallback success"
    
    @patch('ai_meeting_notes.transcription.WhisperModel')
    def test_transcribe_file_all_models_fail(self, mock_whisper_model, transcriber, sample_audio_file):
        """Test transcription when all models fail."""
        with patch.object(transcriber, '_transcribe_with_model') as mock_transcribe:
            mock_transcribe.side_effect = Exception("All models failed")
            
            with pytest.raises(Exception, match="All transcription models failed"):
                transcriber.transcribe_file(sample_audio_file)
    
    @patch('ai_meeting_notes.transcription.WhisperModel')
    def test_retry_transcription(self, mock_whisper_model, transcriber, sample_audio_file):
        """Test retry transcription functionality."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        mock_segment = Mock()
        mock_segment.text = "Retry success"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.avg_logprob = -0.5
        
        mock_info = Mock()
        mock_info.language = "en"
        
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        # Set original model
        transcriber.model_size = "medium"
        transcriber.fallback_models = ["base", "tiny"]
        
        with patch.object(transcriber, 'transcribe_file') as mock_transcribe:
            mock_result = TranscriptResult(
                segments=[TranscriptSegment(
                    text="Retry success",
                    start_time=0.0,
                    end_time=1.0,
                    confidence=0.8
                )],
                full_text="Retry success",
                duration=1.0,
                language="en",
                processing_time=1.0,
                model_name="base",
                audio_file_path=sample_audio_file
            )
            mock_transcribe.return_value = mock_result
            
            result = transcriber.retry_transcription(sample_audio_file)
            
            # Should reset model and try transcription
            assert transcriber.model is None
            assert result.full_text == "Retry success"
    
    def test_transcript_result_properties(self):
        """Test TranscriptResult computed properties."""
        segments = [
            TranscriptSegment(text="High confidence", start_time=0.0, end_time=1.0, confidence=0.9),
            TranscriptSegment(text="Low confidence", start_time=1.0, end_time=2.0, confidence=0.5, is_uncertain=True),
            TranscriptSegment(text="Medium confidence", start_time=2.0, end_time=3.0, confidence=0.7)
        ]
        
        result = TranscriptResult(
            segments=segments,
            full_text="High confidence Low confidence Medium confidence",
            duration=3.0,
            language="en",
            processing_time=1.0,
            model_name="base",
            audio_file_path="test.wav"
        )
        
        # Test average confidence
        expected_avg = (0.9 + 0.5 + 0.7) / 3
        assert abs(result.average_confidence - expected_avg) < 0.01
        
        # Test uncertain segments count
        assert result.uncertain_segments_count == 1
    
    def test_transcript_result_empty_segments(self):
        """Test TranscriptResult with empty segments."""
        result = TranscriptResult(
            segments=[],
            full_text="",
            duration=0.0,
            language="en",
            processing_time=0.0,
            model_name="base",
            audio_file_path="test.wav"
        )
        
        assert result.average_confidence == 0.0
        assert result.uncertain_segments_count == 0
    
    def test_reset_progress(self, transcriber):
        """Test progress reset functionality."""
        # Set some progress
        transcriber._current_progress = 0.5
        transcriber._is_processing = True
        
        # Reset progress
        transcriber.reset_progress()
        
        assert transcriber.get_progress() == 0.0
        assert not transcriber.is_processing()
    
    @patch('ai_meeting_notes.transcription.WhisperModel')
    def test_check_model_availability_success(self, mock_whisper_model, transcriber):
        """Test successful model availability check."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        result = transcriber.check_model_availability()
        
        assert result is True
        mock_whisper_model.assert_called_once_with("tiny", device="cpu")
    
    @patch('ai_meeting_notes.transcription.WhisperModel')
    def test_check_model_availability_failure(self, mock_whisper_model, transcriber):
        """Test model availability check failure."""
        mock_whisper_model.side_effect = Exception("Model not available")
        
        result = transcriber.check_model_availability()
        
        assert result is False
    
    @patch('ai_meeting_notes.transcription.WhisperModel')
    def test_get_available_models(self, mock_whisper_model, transcriber):
        """Test getting available models."""
        # Mock successful loading for tiny and base, failure for others
        def mock_model_side_effect(model_size, device):
            if model_size in ["tiny", "base"]:
                return Mock()
            else:
                raise Exception("Model not available")
        
        mock_whisper_model.side_effect = mock_model_side_effect
        
        available = transcriber.get_available_models()
        
        assert "tiny" in available
        assert "base" in available
        assert "small" not in available
        assert "medium" not in available
        assert "large" not in available