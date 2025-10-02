#!/usr/bin/env python3
"""
Demo script for the BatchTranscriber service.

This script demonstrates how to use the BatchTranscriber to transcribe audio files.
"""

import os
import sys
import tempfile
import wave
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_meeting_notes.transcription import BatchTranscriber
from ai_meeting_notes.models import TranscriptResult


def create_sample_audio(filename: str, duration: float = 5.0) -> str:
    """Create a sample audio file for testing."""
    sample_rate = 16000
    samples = int(sample_rate * duration)
    
    # Create a more complex waveform
    t = np.linspace(0, duration, samples, False)
    
    # Simulate speech-like patterns with multiple frequencies
    f1, f2, f3 = 200, 800, 2400  # Formant frequencies
    audio_data = (
        0.3 * np.sin(2 * np.pi * f1 * t) +
        0.2 * np.sin(2 * np.pi * f2 * t) +
        0.1 * np.sin(2 * np.pi * f3 * t)
    )
    
    # Add amplitude modulation
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 1.5 * t))
    audio_data *= envelope
    
    # Add some noise
    noise = 0.02 * np.random.normal(0, 1, samples)
    audio_data += noise
    
    # Normalize and convert to 16-bit
    audio_data = audio_data / np.max(np.abs(audio_data))
    audio_data = (audio_data * 32767 * 0.8).astype(np.int16)
    
    # Write WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return filename


def demo_transcription():
    """Demonstrate the transcription service."""
    print("🎤 AI Meeting Notes - Transcription Service Demo")
    print("=" * 50)
    
    # Create a temporary audio file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        audio_file = tmp_file.name
    
    try:
        print(f"📁 Creating sample audio file: {audio_file}")
        create_sample_audio(audio_file, duration=3.0)
        
        # Initialize transcriber
        print("🤖 Initializing BatchTranscriber...")
        transcriber = BatchTranscriber(model_size="tiny", device="cpu")
        
        # Check model availability
        print("🔍 Checking model availability...")
        if transcriber.check_model_availability():
            print("✅ Models are available")
        else:
            print("❌ Models are not available - this demo may not work")
            return
        
        # Get available models
        available_models = transcriber.get_available_models()
        print(f"📋 Available models: {available_models}")
        
        # Get audio duration and estimate processing time
        duration = transcriber._get_audio_duration(audio_file)
        estimated_time = transcriber.estimate_processing_time(duration)
        print(f"⏱️  Audio duration: {duration:.1f}s")
        print(f"⏱️  Estimated processing time: {estimated_time:.1f}s")
        
        # Perform transcription
        print("🔄 Starting transcription...")
        print("Progress: ", end="", flush=True)
        
        # Start transcription in a way that we can monitor progress
        # (In a real application, you'd do this in a separate thread)
        result = transcriber.transcribe_file(audio_file)
        
        print(f" {transcriber.get_progress()*100:.0f}%")
        print("✅ Transcription completed!")
        
        # Display results
        print("\n📊 Transcription Results:")
        print("-" * 30)
        print(f"Language: {result.language}")
        print(f"Model used: {result.model_name}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Number of segments: {len(result.segments)}")
        print(f"Average confidence: {result.average_confidence:.2f}")
        print(f"Uncertain segments: {result.uncertain_segments_count}")
        
        print(f"\n📝 Full transcript:")
        print(f"'{result.full_text}'")
        
        if result.segments:
            print(f"\n🔍 Segment details:")
            for i, segment in enumerate(result.segments):
                confidence_indicator = "❓" if segment.is_uncertain else "✅"
                print(f"  {i+1}. [{segment.start_time:.1f}s-{segment.end_time:.1f}s] "
                      f"{confidence_indicator} ({segment.confidence:.2f}) '{segment.text}'")
        
        # Demonstrate retry functionality
        print(f"\n🔄 Testing retry functionality...")
        retry_result = transcriber.retry_transcription(audio_file)
        print(f"✅ Retry completed with model: {retry_result.model_name}")
        
        print(f"\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            os.unlink(audio_file)
            print(f"🧹 Cleaned up temporary file")
        except OSError:
            pass


if __name__ == "__main__":
    demo_transcription()