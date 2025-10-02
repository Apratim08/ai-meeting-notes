import os
import time
import logging
from pathlib import Path
from typing import Optional, List
import wave

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from .models import TranscriptResult, TranscriptSegment


logger = logging.getLogger(__name__)


class BatchTranscriber:
    """Batch transcription service using faster-whisper."""
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Initialize the BatchTranscriber.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device to use ("cpu", "cuda", "auto")
        """
        self.model_size = model_size
        self.device = device
        self.model: Optional[WhisperModel] = None
        self.fallback_models = ["base", "tiny"]  # Fallback models if primary fails
        self._current_progress = 0.0
        self._is_processing = False
        
    def _load_model(self, model_size: str) -> WhisperModel:
        """Load a Whisper model with error handling."""
        try:
            logger.info(f"Loading Whisper model: {model_size}")
            model = WhisperModel(model_size, device=self.device)
            logger.info(f"Successfully loaded model: {model_size}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_size}: {e}")
            raise
    
    def _get_audio_duration(self, audio_file: str) -> float:
        """Get audio file duration in seconds."""
        try:
            with wave.open(audio_file, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
                return duration
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
            return 0.0
    

    
    def estimate_processing_time(self, audio_duration: float) -> float:
        """
        Estimate processing time based on audio duration.
        
        Args:
            audio_duration: Duration of audio in seconds
            
        Returns:
            Estimated processing time in seconds
        """
        # Based on faster-whisper performance: roughly 0.1x real-time for base model
        base_ratio = 0.1
        model_multipliers = {
            "tiny": 0.05,
            "base": 0.1,
            "small": 0.15,
            "medium": 0.25,
            "large": 0.4
        }
        
        multiplier = model_multipliers.get(self.model_size, base_ratio)
        return audio_duration * multiplier
    
    def get_progress(self) -> float:
        """Get current transcription progress (0.0 to 1.0)."""
        return self._current_progress
    
    def is_processing(self) -> bool:
        """Check if transcription is currently in progress."""
        return self._is_processing
    
    def reset_progress(self) -> None:
        """Reset progress tracking for a new transcription."""
        self._current_progress = 0.0
        self._is_processing = False
    
    def check_model_availability(self) -> bool:
        """
        Check if faster-whisper models are available.
        
        Returns:
            True if models can be loaded, False otherwise
        """
        try:
            # Try to load the smallest model to check availability
            test_model = WhisperModel("tiny", device=self.device)
            del test_model  # Clean up
            return True
        except Exception as e:
            logger.error(f"Model availability check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Whisper model sizes.
        
        Returns:
            List of available model names
        """
        available_models = []
        model_sizes = ["tiny", "base", "small", "medium", "large"]
        
        for model_size in model_sizes:
            try:
                # Quick check - just try to initialize
                test_model = WhisperModel(model_size, device=self.device)
                available_models.append(model_size)
                del test_model
            except Exception:
                # Model not available, skip
                continue
        
        return available_models
    
    def transcribe_file(self, audio_file: str) -> TranscriptResult:
        """
        Transcribe an audio file to text with comprehensive error handling.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            TranscriptResult with segments and metadata
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If transcription fails with all models
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Validate file format and accessibility
        try:
            file_size = os.path.getsize(audio_file)
            if file_size == 0:
                raise ValueError(f"Audio file is empty: {audio_file}")
            
            # Check if file is readable
            with open(audio_file, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
                
        except (OSError, IOError) as e:
            raise Exception(f"Cannot access audio file {audio_file}: {e}")
        
        self._is_processing = True
        self._current_progress = 0.0
        
        try:
            # Try primary model first
            return self._transcribe_with_model(audio_file, self.model_size)
        except Exception as e:
            logger.error(f"Primary model {self.model_size} failed: {e}")
            
            # Check if it's a model availability issue
            if "model" in str(e).lower() and "not found" in str(e).lower():
                logger.warning(f"Model {self.model_size} not available, trying fallbacks")
            
            # Try fallback models
            for fallback_model in self.fallback_models:
                if fallback_model != self.model_size:
                    try:
                        logger.info(f"Trying fallback model: {fallback_model}")
                        return self._transcribe_with_model(audio_file, fallback_model)
                    except Exception as fallback_error:
                        logger.error(f"Fallback model {fallback_model} failed: {fallback_error}")
                        continue
            
            # If all models fail, provide detailed error message
            error_msg = f"All transcription models failed. Primary error: {e}"
            if "faster_whisper" in str(e):
                error_msg += " (faster-whisper library may not be properly installed)"
            elif "cuda" in str(e).lower():
                error_msg += " (GPU/CUDA issues detected, try CPU mode)"
            elif "memory" in str(e).lower():
                error_msg += " (insufficient memory, try a smaller model)"
            
            raise Exception(error_msg)
        finally:
            self._is_processing = False
            # Don't reset progress to 0.0 - keep it at final value for UI feedback
    
    def _transcribe_with_model(self, audio_file: str, model_size: str) -> TranscriptResult:
        """Transcribe using a specific model."""
        start_time = time.time()
        
        # Load model if not already loaded or if different model needed
        if self.model is None or model_size != self.model_size:
            self.model = self._load_model(model_size)
            self.model_size = model_size
        
        self._current_progress = 0.1  # Model loaded
        
        # Get audio duration for progress tracking
        audio_duration = self._get_audio_duration(audio_file)
        
        logger.info(f"Starting transcription of {audio_file} (duration: {audio_duration:.1f}s)")
        
        # Transcribe with progress tracking
        segments_list = []
        full_text_parts = []
        
        try:
            # Use faster-whisper transcribe method
            segments, info = self.model.transcribe(
                audio_file,
                beam_size=5,
                language=None,  # Auto-detect language
                condition_on_previous_text=False,
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            self._current_progress = 0.2  # Transcription started
            
            # Process segments
            total_segments = 0
            processed_segments = 0
            
            # First pass: count segments for progress tracking
            segments_iter = list(segments)
            total_segments = len(segments_iter)
            
            for segment in segments_iter:
                # Create transcript segment
                confidence = getattr(segment, 'avg_logprob', 0.0)
                # Convert log probability to confidence (approximate)
                confidence_score = max(0.0, min(1.0, (confidence + 5.0) / 5.0))
                
                is_uncertain = confidence_score < 0.7
                
                transcript_segment = TranscriptSegment(
                    text=segment.text.strip(),
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=confidence_score,
                    is_uncertain=is_uncertain
                )
                
                segments_list.append(transcript_segment)
                full_text_parts.append(segment.text.strip())
                
                processed_segments += 1
                # Update progress (20% to 90% for processing segments)
                self._current_progress = 0.2 + (processed_segments / total_segments) * 0.7
            
            self._current_progress = 0.95  # Processing complete
            
            # Create final result
            processing_time = time.time() - start_time
            full_text = " ".join(full_text_parts)
            
            result = TranscriptResult(
                segments=segments_list,
                full_text=full_text,
                duration=audio_duration,
                language=info.language,
                processing_time=processing_time,
                model_name=model_size,
                audio_file_path=audio_file
            )
            
            self._current_progress = 1.0  # Complete
            
            logger.info(f"Transcription completed in {processing_time:.1f}s. "
                       f"Language: {info.language}, "
                       f"Segments: {len(segments_list)}, "
                       f"Avg confidence: {result.average_confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def retry_transcription(self, audio_file: str) -> TranscriptResult:
        """
        Retry transcription with a different model or settings.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            TranscriptResult from retry attempt
        """
        logger.info(f"Retrying transcription for {audio_file}")
        
        # Reset model to force reload
        self.model = None
        
        # Try with a different model size if available
        original_model = self.model_size
        
        # Use fallback models for retry
        for fallback_model in self.fallback_models:
            if fallback_model != original_model:
                try:
                    logger.info(f"Retry with model: {fallback_model}")
                    self.model_size = fallback_model
                    return self.transcribe_file(audio_file)
                except Exception as e:
                    logger.error(f"Retry with {fallback_model} failed: {e}")
                    continue
        
        # If all retries fail, try original model one more time
        self.model_size = original_model
        return self.transcribe_file(audio_file)