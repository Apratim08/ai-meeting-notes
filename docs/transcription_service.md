# Batch Transcription Service

The `BatchTranscriber` class provides audio-to-text transcription capabilities using the faster-whisper library. It's designed for batch processing of audio files with progress tracking, error handling, and retry mechanisms.

## Features

- **Batch Processing**: Transcribes complete audio files (not real-time)
- **Progress Tracking**: Monitor transcription progress for UI updates
- **Model Fallback**: Automatically tries fallback models if primary model fails
- **Retry Mechanism**: Retry failed transcriptions with different models
- **Confidence Scoring**: Marks uncertain text segments (confidence < 70%)
- **Error Preservation**: Preserves original audio files on transcription failure

## Usage

### Basic Usage

```python
from ai_meeting_notes.transcription import BatchTranscriber

# Initialize transcriber
transcriber = BatchTranscriber(model_size="base", device="cpu")

# Transcribe an audio file
result = transcriber.transcribe_file("meeting.wav")

# Access results
print(f"Transcript: {result.full_text}")
print(f"Language: {result.language}")
print(f"Confidence: {result.average_confidence:.2f}")
```

### Progress Monitoring

```python
# Check if transcription is in progress
if transcriber.is_processing():
    progress = transcriber.get_progress()  # 0.0 to 1.0
    print(f"Progress: {progress * 100:.0f}%")
```

### Error Handling and Retry

```python
try:
    result = transcriber.transcribe_file("audio.wav")
except Exception as e:
    print(f"Transcription failed: {e}")
    
    # Retry with different model
    try:
        result = transcriber.retry_transcription("audio.wav")
        print("Retry successful!")
    except Exception as retry_error:
        print(f"Retry also failed: {retry_error}")
```

### Model Management

```python
# Check if models are available
if transcriber.check_model_availability():
    print("Models are ready")

# Get list of available models
available = transcriber.get_available_models()
print(f"Available models: {available}")

# Estimate processing time
duration = transcriber._get_audio_duration("audio.wav")
estimated_time = transcriber.estimate_processing_time(duration)
print(f"Estimated processing time: {estimated_time:.1f}s")
```

## Configuration

### Model Sizes

- `tiny`: Fastest, lowest accuracy (~39 MB)
- `base`: Good balance of speed and accuracy (~74 MB)  
- `small`: Better accuracy, slower (~244 MB)
- `medium`: High accuracy (~769 MB)
- `large`: Best accuracy, slowest (~1550 MB)

### Device Options

- `cpu`: Use CPU for processing (default)
- `cuda`: Use GPU if available (requires CUDA)
- `auto`: Automatically detect best device

## Data Models

### TranscriptResult

The main result object containing:

```python
class TranscriptResult:
    segments: List[TranscriptSegment]  # Individual text segments
    full_text: str                     # Complete transcript
    duration: float                    # Audio duration in seconds
    language: str                      # Detected language
    processing_time: float             # Time taken to process
    model_name: str                    # Model used for transcription
    audio_file_path: str              # Path to original audio file
    
    # Computed properties
    average_confidence: float          # Average confidence score
    uncertain_segments_count: int      # Number of low-confidence segments
```

### TranscriptSegment

Individual text segments with timing and confidence:

```python
class TranscriptSegment:
    text: str                 # Transcribed text
    start_time: float         # Start time in seconds
    end_time: float           # End time in seconds
    confidence: float         # Confidence score (0.0 to 1.0)
    speaker_id: Optional[str] # Speaker identifier (if available)
    is_uncertain: bool        # True if confidence < 0.7
```

## Requirements Compliance

This implementation satisfies the following requirements:

- **Requirement 2.1**: Converts speech to text using faster-whisper (open-source)
- **Requirement 2.2**: Provides timestamps for each text segment
- **Requirement 2.4**: Marks uncertain text with confidence indicators
- **Requirement 5.2**: Implements retry mechanism with fallback models and preserves original audio

## Performance

Expected processing times (approximate):

- **tiny model**: ~5% of real-time (3 minutes audio → 9 seconds processing)
- **base model**: ~10% of real-time (3 minutes audio → 18 seconds processing)
- **small model**: ~15% of real-time (3 minutes audio → 27 seconds processing)

Actual performance depends on:
- Hardware capabilities (CPU/GPU)
- Audio complexity (number of speakers, background noise)
- Audio duration and quality

## Error Handling

The service handles various error scenarios:

1. **File not found**: Raises `FileNotFoundError`
2. **Model loading failure**: Tries fallback models automatically
3. **Transcription failure**: Preserves audio file and allows retry
4. **All models fail**: Raises exception with detailed error message

## Testing

Run the test suite:

```bash
# Unit tests
python -m pytest tests/test_transcription.py -v

# Integration tests
python -m pytest tests/test_transcription_integration.py -v

# Demo script
python examples/transcription_demo.py
```

## Dependencies

- `faster-whisper>=0.10.0`: Core transcription engine
- `numpy>=1.24.0`: Audio processing
- `pydantic>=2.5.0`: Data validation
- `wave`: Audio file handling (built-in)