# AI Meeting Notes

An AI-powered meeting notes application that captures audio from system speakers, transcribes it to text, and uses an open-source LLM to generate structured meeting notes.

## Features

- Manual audio capture from system speakers (via BlackHole on macOS)
- Batch transcription using faster-whisper
- AI-generated structured meeting notes using Ollama
- Simple web interface for monitoring and control
- Privacy-focused: all processing happens locally

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
2. **Install Ollama**:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the required model
   ollama pull llama3.1:8b
   ```

3. **Install Python dependencies**:
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
6. Wait for transcription and notes generation to complete

## Configuration

Copy `.env.example` to `.env` and modify settings as needed.

## Project Structure

```
ai_meeting_notes/
├── __init__.py
├── config.py              # Configuration management
├── audio_recorder.py      # Audio recording with BlackHole
├── transcription.py       # Batch transcription service
├── notes_generator.py     # AI notes generation
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

Start development server:
```bash
uvicorn ai_meeting_notes.main:app --reload
```