"""Integration tests for transcription service with real audio files."""

import os
import pytest
import tempfile
import wave
import numpy as np
from pathlib import Path

from ai_meeting_notes.transcription import BatchTranscriber
from ai_meeting_notes.models import TranscriptResult


class TestTranscriptionIntegration:
    """Integration tests for BatchTranscriber with real audio processing."""
    
    @pytest.fixture
    def speech_audio_file(self):
        """Create a more realistic audio file with speech-like patterns."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sample_rate = 16000
            duration = 3.0  # 3 seconds
            samples = int(sample_rate * duration)
            
            # Create a more complex waveform that resembles speech
            t = np.linspace(0, duration, samples, False)
            
            # Combine multiple frequencies to simulate speech formants
            f1, f2, f3 = 200, 800, 2400  # Typical formant frequencies
            audio_data = (
                0.3 * np.sin(2 * np.pi * f1 * t) +
                0.2 * np.sin(2 * np.pi * f2 * t) +
                0.1 * np.sin(2 * np.pi * f3 * t)
            )
            
            # Add some amplitude modulation to simulate speech rhythm
            envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))  # 2 Hz modulation
            audio_data *= envelope
            
            # Add some noise for realism
            noise = 0.05 * np.random.normal(0, 1, samples)
            audio_data += noise
            
            # Normalize and convert to 16-bit integers
            audio_data = audio_data / np.max(np.abs(audio_data))
            audio_data = (audio_data * 32767 * 0.8).astype(np.int16)
            
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
    
    @pytest.fixture
    def silent_audio_file(self):
        """Create an audio file with mostly silence."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sample_rate = 16000
            duration = 2.0
            samples = int(sample_rate * duration)
            
            # Create mostly silent audio with very low amplitude noise
            audio_data = 0.001 * np.random.normal(0, 1, samples)
            audio_data = (audio_data * 32767).astype(np.int16)
            
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            yield tmp_file.name
            
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass
    
    def test_transcribe_speech_like_audio(self, speech_audio_file):
        """Test transcription with speech-like audio patterns."""
        transcriber = BatchTranscriber(model_size="tiny", device="cpu")
        
        # This test may not produce meaningful text since it's synthetic audio,
        # but it should complete without errors
        try:
            result = transcriber.transcribe_file(speech_audio_file)
            
            # Verify result structure
            assert isinstance(result, TranscriptResult)
            assert result.audio_file_path == speech_audio_file
            assert result.duration > 0
            assert result.processing_time > 0
            assert result.model_name == "tiny"
            assert isinstance(result.segments, list)
            assert isinstance(result.full_text, str)
            
            # Progress should be complete
            assert transcriber.get_progress() == 1.0
            assert not transcriber.is_processing()
            
            print(f"Transcription result: '{result.full_text}'")
            print(f"Segments: {len(result.segments)}")
            print(f"Average confidence: {result.average_confidence:.2f}")
            
        except Exception as e:
            # If faster-whisper is not available, skip the test
            if "faster_whisper" in str(e) or "WhisperModel" in str(e):
                pytest.skip(f"faster-whisper not available: {e}")
            else:
                raise
    
    def test_transcribe_silent_audio(self, silent_audio_file):
        """Test transcription with silent audio."""
        transcriber = BatchTranscriber(model_size="tiny", device="cpu")
        
        try:
            result = transcriber.transcribe_file(silent_audio_file)
            
            # Silent audio should produce minimal or no segments
            assert isinstance(result, TranscriptResult)
            assert result.audio_file_path == silent_audio_file
            assert result.duration > 0
            
            # May have empty or very short text for silent audio
            assert isinstance(result.full_text, str)
            
            print(f"Silent audio result: '{result.full_text}'")
            print(f"Segments: {len(result.segments)}")
            
        except Exception as e:
            if "faster_whisper" in str(e) or "WhisperModel" in str(e):
                pytest.skip(f"faster-whisper not available: {e}")
            else:
                raise
    
    def test_progress_tracking_during_transcription(self, speech_audio_file):
        """Test that progress tracking works during transcription."""
        transcriber = BatchTranscriber(model_size="tiny", device="cpu")
        
        try:
            # Initial progress should be 0
            assert transcriber.get_progress() == 0.0
            assert not transcriber.is_processing()
            
            # Start transcription (this will complete quickly with tiny model)
            result = transcriber.transcribe_file(speech_audio_file)
            
            # After completion, progress should be 1.0 and not processing
            assert transcriber.get_progress() == 1.0
            assert not transcriber.is_processing()
            
        except Exception as e:
            if "faster_whisper" in str(e) or "WhisperModel" in str(e):
                pytest.skip(f"faster-whisper not available: {e}")
            else:
                raise
    
    def test_processing_time_estimation(self, speech_audio_file):
        """Test processing time estimation accuracy."""
        transcriber = BatchTranscriber(model_size="tiny", device="cpu")
        
        # Get actual audio duration
        duration = transcriber._get_audio_duration(speech_audio_file)
        assert duration > 0
        
        # Get estimated processing time
        estimated_time = transcriber.estimate_processing_time(duration)
        assert estimated_time > 0
        
        # For tiny model, should be much faster than real-time
        assert estimated_time < duration
        
        print(f"Audio duration: {duration:.1f}s")
        print(f"Estimated processing time: {estimated_time:.1f}s")
    
    def test_model_fallback_behavior(self, speech_audio_file):
        """Test fallback behavior when primary model fails."""
        # Create transcriber with non-existent model that should fail
        transcriber = BatchTranscriber(model_size="nonexistent_model", device="cpu")
        
        try:
            # This should fall back to base/tiny models
            result = transcriber.transcribe_file(speech_audio_file)
            
            # Should succeed with fallback model
            assert isinstance(result, TranscriptResult)
            # Model name should be one of the fallback models
            assert result.model_name in ["base", "tiny"]
            
        except Exception as e:
            # Expected if no models are available
            if "faster_whisper" in str(e) or "WhisperModel" in str(e):
                pytest.skip(f"faster-whisper not available: {e}")
            elif "All transcription models failed" in str(e):
                # This is expected behavior when all models fail
                pass
            else:
                raise
    
    def test_retry_transcription_functionality(self, speech_audio_file):
        """Test retry transcription functionality."""
        transcriber = BatchTranscriber(model_size="tiny", device="cpu")
        
        try:
            # First transcription
            result1 = transcriber.transcribe_file(speech_audio_file)
            
            # Retry transcription
            result2 = transcriber.retry_transcription(speech_audio_file)
            
            # Both should succeed and have similar structure
            assert isinstance(result1, TranscriptResult)
            assert isinstance(result2, TranscriptResult)
            assert result1.audio_file_path == result2.audio_file_path
            
        except Exception as e:
            if "faster_whisper" in str(e) or "WhisperModel" in str(e):
                pytest.skip(f"faster-whisper not available: {e}")
            else:
                raise


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])