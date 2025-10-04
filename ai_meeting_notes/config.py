"""Configuration management for AI Meeting Notes application."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    """Audio recording configuration."""
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels (mono)")
    chunk_size: int = Field(default=1024, description="Audio buffer chunk size")
    device_id: Optional[int] = Field(default=None, description="Audio device ID (auto-detect if None)")
    blackhole_device_name: str = Field(default="BlackHole 2ch", description="BlackHole device name to detect")
    max_recording_hours: float = Field(default=4.0, description="Maximum recording duration in hours")
    enable_microphone: bool = Field(default=True, description="Enable microphone input for capturing your voice")
    microphone_device_name: Optional[str] = Field(default=None, description="Specific microphone device name (auto-detect if None)")
    mic_gain: float = Field(default=0.5, description="Microphone audio gain multiplier (0.0-2.0)")


class TranscriptionConfig(BaseModel):
    """Transcription service configuration."""
    model_name: str = Field(default="base", description="Whisper model size (tiny, base, small, medium, large)")
    language: str = Field(default="auto", description="Language code or 'auto' for detection")
    enable_speaker_diarization: bool = Field(default=False, description="Enable speaker identification")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for transcription")
    device: str = Field(default="auto", description="Device for inference (auto, cpu, cuda)")


class LLMConfig(BaseModel):
    """LLM processing configuration."""
    model_name: str = Field(default="qwen2.5:14b", description="Ollama model name")
    temperature: float = Field(default=0.2, description="LLM temperature for generation")
    max_tokens: int = Field(default=16000, description="Maximum tokens for notes generation")
    timeout_seconds: int = Field(default=1200, description="Timeout for LLM requests")
    max_retries: int = Field(default=3, description="Number of retry attempts on failure")
    retry_delay: float = Field(default=2.0, description="Delay between retries in seconds")
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama server URL")


class ServerConfig(BaseModel):
    """FastAPI server configuration."""
    host: str = Field(default="127.0.0.1", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")


class FileConfig(BaseModel):
    """File management configuration."""
    temp_dir: Path = Field(default=Path("temp"), description="Temporary files directory")
    max_file_size_mb: float = Field(default=500.0, description="Maximum audio file size in MB")
    cleanup_after_success: bool = Field(default=False, description="Auto-cleanup files after successful processing")
    retention_days: int = Field(default=7, description="Days to retain files before cleanup")


class AppConfig(BaseModel):
    """Main application configuration."""
    audio: AudioConfig = Field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    files: FileConfig = Field(default_factory=FileConfig)
    
    @classmethod
    def load_from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        config = cls()
        
        # Audio settings
        if os.getenv("AUDIO_SAMPLE_RATE"):
            config.audio.sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE"))
        if os.getenv("AUDIO_DEVICE_ID"):
            config.audio.device_id = int(os.getenv("AUDIO_DEVICE_ID"))
        
        # Transcription settings
        if os.getenv("WHISPER_MODEL"):
            config.transcription.model_name = os.getenv("WHISPER_MODEL")
        if os.getenv("TRANSCRIPTION_LANGUAGE"):
            config.transcription.language = os.getenv("TRANSCRIPTION_LANGUAGE")
        
        # LLM settings
        if os.getenv("OLLAMA_MODEL"):
            config.llm.model_name = os.getenv("OLLAMA_MODEL")
        if os.getenv("LLM_TEMPERATURE"):
            config.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))
        
        # Server settings
        if os.getenv("SERVER_HOST"):
            config.server.host = os.getenv("SERVER_HOST")
        if os.getenv("SERVER_PORT"):
            config.server.port = int(os.getenv("SERVER_PORT"))
        if os.getenv("DEBUG"):
            config.server.debug = os.getenv("DEBUG").lower() == "true"
        
        return config
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.files.temp_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = AppConfig.load_from_env()