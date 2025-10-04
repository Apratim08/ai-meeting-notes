#!/usr/bin/env python3
"""Test script for dual-stream audio recording (BlackHole + Microphone)."""

import time
from ai_meeting_notes.audio_recorder import AudioRecorder
from ai_meeting_notes.config import config

def test_dual_recording():
    """Test dual-stream recording with microphone enabled."""

    print("=" * 60)
    print("Dual-Stream Recording Test")
    print("=" * 60)

    # Enable microphone recording
    config.audio.enable_microphone = True
    config.audio.mic_gain = 0.5  # Reduced gain to minimize echo (was 1.0)

    print(f"\nConfiguration:")
    print(f"  Enable Microphone: {config.audio.enable_microphone}")
    print(f"  Microphone Gain: {config.audio.mic_gain}")
    print(f"  Sample Rate: {config.audio.sample_rate} Hz")
    print(f"  Chunk Size: {config.audio.chunk_size}")

    # Create recorder
    recorder = AudioRecorder()

    print(f"\nDevice Status:")
    print(f"  BlackHole Available: {recorder.check_blackhole_available()}")
    print(f"  Microphone Available: {recorder.check_microphone_available()}")

    # List available devices
    print(f"\nAvailable Devices:")
    devices = recorder.get_available_devices()
    for device in devices:
        device_type = '🎤 Mic' if device['is_microphone'] else '🔊 BlackHole'
        print(f"  [{device['id']}] {device['name']} - {device_type}")

    # Start recording
    print("\n" + "-" * 60)
    print("Starting dual-stream recording...")
    print("-" * 60)

    output_file = "temp/test_dual_recording.wav"
    recorder.start_recording(output_file)

    # Record for 10 seconds
    print("\n🎙️  Recording for 10 seconds...")
    print("   👉 Please speak into your microphone")
    print("   👉 Play some audio through your speakers/meeting app")

    for i in range(10, 0, -1):
        print(f"   ⏱️  {i} seconds remaining...", end='\r')
        time.sleep(1)

    print("\n")

    # Stop recording
    print("-" * 60)
    print("Stopping recording...")
    print("-" * 60)

    saved_file = recorder.stop_recording()

    print(f"\n✅ Recording complete!")
    print(f"📁 Saved to: {saved_file}")

    # Get recording info
    import wave
    with wave.open(saved_file, 'rb') as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        duration = frames / sample_rate

        print(f"\nRecording Details:")
        print(f"  Channels: {channels}")
        print(f"  Sample Rate: {sample_rate} Hz")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Total Frames: {frames}")

    print(f"\n🎧 Play the recording to verify:")
    print(f"   You should hear BOTH:")
    print(f"   ✅ Your voice (from microphone)")
    print(f"   ✅ System/meeting audio (from BlackHole)")

    print("\n" + "=" * 60)


def test_blackhole_only():
    """Test BlackHole-only recording (microphone disabled)."""

    print("\n" + "=" * 60)
    print("BlackHole-Only Recording Test (Legacy Mode)")
    print("=" * 60)

    # Disable microphone
    config.audio.enable_microphone = False

    print(f"\nConfiguration:")
    print(f"  Enable Microphone: {config.audio.enable_microphone}")

    recorder = AudioRecorder()

    print("\n" + "-" * 60)
    print("Starting BlackHole-only recording...")
    print("-" * 60)

    output_file = "temp/test_blackhole_only.wav"
    recorder.start_recording(output_file)

    print("\n🔊 Recording for 5 seconds...")
    print("   👉 Play some audio through your speakers")

    for i in range(5, 0, -1):
        print(f"   ⏱️  {i} seconds remaining...", end='\r')
        time.sleep(1)

    print("\n")

    saved_file = recorder.stop_recording()

    print(f"\n✅ Recording complete!")
    print(f"📁 Saved to: {saved_file}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test dual-stream recording
    test_dual_recording()

    # Optionally test BlackHole-only mode
    print("\n\nPress Enter to test BlackHole-only mode (or Ctrl+C to skip)...")
    try:
        input()
        test_blackhole_only()
    except KeyboardInterrupt:
        print("\n\nSkipped BlackHole-only test.")

    print("\n✨ All tests complete!")
