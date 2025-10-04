# AI Meeting Notes

An AI-powered meeting notes application that captures audio from system speakers and microphone, transcribes it to text, and exports to ChatGPT/Claude for structured meeting notes generation.

## Features

- Dual-stream audio capture from system speakers (via BlackHole) and microphone
- Real-time audio mixing with configurable gain
- High-quality transcription using faster-whisper
- One-click export to ChatGPT/Claude for AI-generated notes
- Simple web interface for monitoring and control
- Privacy-focused: transcription happens locally, you control what goes to cloud LLMs

## Setup Requirements

### macOS Audio Setup

1. **Install BlackHole** (virtual audio driver):
   - Download from: https://github.com/ExistentialAudio/BlackHole
   - Install BlackHole 2ch

2. **Create Multi-Output Device**:
   - Open Audio MIDI Setup app
   - Click "+" and select "Create Multi-Output Device"
   - Check both your speakers/headphones AND BlackHole 2ch
   - Set this as your system output during meetings

### Software Requirements

1. **Install Python 3.9+**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Set your system audio output to the Multi-Output Device
2. Start the application:
   ```bash
   python -m ai_meeting_notes.main
   ```
3. Open http://localhost:8000 in your browser
4. Click "Start Recording" when your meeting begins
5. Click "Stop Recording" when the meeting ends
6. Wait for transcription to complete
7. Click "📋 Export for ChatGPT/Claude" to copy the prompt + transcript
8. Paste into ChatGPT or Claude to generate structured meeting notes

## Configuration

Copy `.env.example` to `.env` and modify settings as needed:

```bash
# Audio settings
AUDIO_SAMPLE_RATE=16000
MIC_GAIN=0.5  # Microphone gain (0.0-2.0)

# Transcription settings
WHISPER_MODEL=base  # tiny, base, small, medium, large
TRANSCRIPTION_LANGUAGE=auto
```

## Project Structure

```
ai_meeting_notes/
├── __init__.py
├── config.py              # Configuration management
├── audio_recorder.py      # Dual-stream audio recording
├── transcription.py       # Batch transcription service
├── notes_generator.py     # ChatGPT/Claude prompt generation
├── models.py             # Data models
├── file_manager.py       # File management and cleanup
├── main.py              # FastAPI server
├── api/                 # API endpoints
├── templates/           # HTML templates
└── static/             # CSS/JS assets

tests/                   # Unit and integration tests
```

## Development

Run tests:
```bash
pytest
```

Start development server with auto-reload:
```bash
uvicorn ai_meeting_notes.main:app --reload
```

## Why ChatGPT/Claude instead of local LLM?

- **Higher quality**: ChatGPT and Claude produce better structured notes
- **No GPU required**: No need for expensive local compute
- **Faster processing**: Cloud LLMs are optimized and fast
- **You control privacy**: Review the transcript before sending to cloud
- **Cost effective**: Pay only for what you use
