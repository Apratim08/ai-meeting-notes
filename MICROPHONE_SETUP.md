# Microphone Input Setup Guide

## Overview

The AI Meeting Notes application now supports **dual-stream recording** to capture both:
- 🔊 **System audio** (others in the meeting) via BlackHole
- 🎤 **Your voice** via microphone

Both streams are mixed in real-time to create a complete meeting recording.

## Configuration

### Enable Microphone Recording

Add to your environment variables or config:

```bash
# Enable microphone input
export AUDIO_ENABLE_MICROPHONE=true

# Optional: Specify microphone device name (auto-detected if not set)
export AUDIO_MICROPHONE_DEVICE_NAME="Built-in Microphone"

# Optional: Adjust microphone gain (0.0 - 2.0, default 1.0)
export AUDIO_MIC_GAIN=1.2
```

Or in Python:

```python
from ai_meeting_notes.config import config

config.audio.enable_microphone = True
config.audio.mic_gain = 1.2  # Boost mic volume by 20%
```

## Recording Modes

The system automatically selects the best recording mode based on available devices:

### 1. **Dual-Stream Mode** (Recommended)
- **When**: `enable_microphone=True` AND both BlackHole and mic are available
- **Captures**: System audio + your voice
- **Output**: Mixed audio with both sources

### 2. **BlackHole Only Mode** (Current behavior)
- **When**: BlackHole available, microphone disabled or unavailable
- **Captures**: System audio only
- **Note**: Your voice will NOT be recorded (Google Meet doesn't echo you back)

### 3. **Microphone Only Mode** (Fallback)
- **When**: BlackHole unavailable, microphone available
- **Captures**: Your voice only
- **Note**: Others' audio will be degraded/echoed through speakers

## How Real-Time Mixing Works

```
BlackHole Stream → [Callback] → Latest BlackHole Chunk
                                          ↓
                                    [Mix & Average] → Recording
                                          ↑
Microphone Stream → [Callback] → Latest Mic Chunk (× gain)
```

**No timestamps needed!** Both streams run synchronously at the same sample rate (16kHz).

Each callback:
1. Updates its latest audio chunk
2. Mixes with the other stream's latest chunk
3. Stores the mixed result

## Audio Quality Settings

### Microphone Gain
Adjust if your voice is too quiet or too loud:

```python
config.audio.mic_gain = 0.8  # Reduce mic by 20%
config.audio.mic_gain = 1.5  # Boost mic by 50%
```

### Mixing Formula
```python
mixed_audio = (blackhole_chunk + mic_chunk * mic_gain) / 2
```

## Testing Your Setup

### 1. Check Available Devices

```python
from ai_meeting_notes.audio_recorder import AudioRecorder

recorder = AudioRecorder()
devices = recorder.get_available_devices()

for device in devices:
    print(f"{device['name']}: {'🎤 Mic' if device['is_microphone'] else '🔊 BlackHole'}")
```

### 2. Test Recording

```python
# Enable microphone
from ai_meeting_notes.config import config
config.audio.enable_microphone = True

# Start recording
recorder = AudioRecorder()
recorder.start_recording("test_recording.wav")

# You should see:
# 🎙️  Recording mode: Dual-stream (BlackHole + Microphone)
# Using BlackHole: BlackHole 2ch
# Using Microphone: Built-in Microphone

# Speak and play audio, then stop
recorder.stop_recording()
```

### 3. Verify Output

Listen to `test_recording.wav` - you should hear:
- ✅ Other participants from the meeting (via BlackHole)
- ✅ Your own voice (via microphone)

## Troubleshooting

### "Microphone enabled but not found"
- Check microphone permissions in System Preferences → Security & Privacy → Microphone
- Verify microphone is connected and working
- Try specifying device name explicitly: `config.audio.microphone_device_name = "Your Mic Name"`

### "Your voice is too loud/quiet"
- Adjust mic gain: `config.audio.mic_gain = 1.5` (or lower)

### "Echo or duplicate audio"
- This is expected if your mic picks up speaker output
- Use headphones to prevent feedback
- Or lower mic gain: `config.audio.mic_gain = 0.7`

### "Only hearing one source"
- Check that both devices are detected: `recorder.get_available_devices()`
- Ensure `enable_microphone = True`
- Verify BlackHole Multi-Output Device is properly configured

## API Changes

### New Config Options
- `audio.enable_microphone: bool` - Enable microphone input (default: `False`)
- `audio.microphone_device_name: Optional[str]` - Specific mic name (default: auto-detect)
- `audio.mic_gain: float` - Mic volume multiplier 0.0-2.0 (default: `1.0`)

### New Methods
- `AudioRecorder.get_microphone_device_id() -> Optional[int]`
- `AudioRecorder.check_microphone_available() -> bool`

### Updated Methods
- `AudioRecorder.get_available_devices()` - Now includes `is_microphone` flag
- `AudioRecorder.start_recording()` - Automatically detects and uses dual-stream mode

## Backward Compatibility

✅ **100% Backward Compatible**

- Default config: `enable_microphone = False` (current behavior)
- Existing code works without changes
- No breaking API changes
