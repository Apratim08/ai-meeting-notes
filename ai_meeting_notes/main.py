"""
FastAPI server for AI Meeting Notes application.

Provides REST API endpoints for meeting control, status monitoring,
and retrieving transcription and notes results.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .audio_recorder import AudioRecorder, AudioRecorderError
from .transcription import BatchTranscriber
from .notes_generator import NotesGenerator, OllamaError
from .file_manager import FileManager
from .processing_pipeline import ProcessingPipeline, ProcessingError
from .models import MeetingSession, TranscriptResult, MeetingNotes
from .config import config
from .setup_validator import SetupValidator, get_system_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
processing_pipeline: Optional[ProcessingPipeline] = None
audio_recorder: Optional[AudioRecorder] = None
transcriber: Optional[BatchTranscriber] = None
notes_generator: Optional[NotesGenerator] = None
file_manager: Optional[FileManager] = None
setup_validator: Optional[SetupValidator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    global processing_pipeline, audio_recorder, transcriber, notes_generator, file_manager, setup_validator
    
    logger.info("Starting AI Meeting Notes server...")
    
    # Initialize services
    try:
        audio_recorder = AudioRecorder()
        transcriber = BatchTranscriber(
            model_size=config.transcription.model_name,
            device=config.transcription.device
        )
        notes_generator = NotesGenerator(config)
        file_manager = FileManager(
            base_dir=str(config.files.temp_dir),
            max_file_size_mb=config.files.max_file_size_mb,
            auto_cleanup_days=config.files.retention_days
        )
        setup_validator = SetupValidator()
        
        # Initialize processing pipeline
        processing_pipeline = ProcessingPipeline(
            audio_recorder=audio_recorder,
            transcriber=transcriber,
            notes_generator=notes_generator,
            file_manager=file_manager,
            progress_callback=None  # Could add WebSocket progress updates later
        )
        
        # Ensure directories exist
        config.ensure_directories()
        
        logger.info("All services and processing pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Meeting Notes server...")
    
    # Clear any active session and stop recording
    if processing_pipeline:
        try:
            processing_pipeline.clear_session()
            logger.info("Cleared active session during shutdown")
        except Exception as e:
            logger.error(f"Error clearing session during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="AI Meeting Notes API",
    description="REST API for AI-powered meeting transcription and notes generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))


# Response models
class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    status: str
    message: str
    session_id: Optional[str] = None
    recording_duration: Optional[float] = None
    processing_progress: Optional[float] = None
    error_message: Optional[str] = None


class StartRecordingResponse(BaseModel):
    """Response model for start recording endpoint."""
    success: bool
    message: str
    session_id: str


class StopRecordingResponse(BaseModel):
    """Response model for stop recording endpoint."""
    success: bool
    message: str
    audio_file: Optional[str] = None


class TranscriptResponse(BaseModel):
    """Response model for transcript endpoint."""
    transcript: Optional[TranscriptResult] = None
    available: bool
    message: str


class NotesResponse(BaseModel):
    """Response model for notes endpoint."""
    notes: Optional[MeetingNotes] = None
    available: bool
    message: str


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    services: Dict[str, bool]
    message: str


class SetupValidationResponse(BaseModel):
    """Response model for setup validation endpoint."""
    overall_status: str
    results: List[Dict[str, Any]]
    setup_complete: bool
    next_steps: List[str]


class SetupWizardResponse(BaseModel):
    """Response model for setup wizard endpoint."""
    steps: List[Dict[str, Any]]


class SystemInfoResponse(BaseModel):
    """Response model for system info endpoint."""
    system_info: Dict[str, Any]


# Web Routes

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main web interface."""
    if templates_dir.exists():
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse("""
        <html>
            <head><title>AI Meeting Notes</title></head>
            <body>
                <h1>AI Meeting Notes</h1>
                <p>Template directory not found. Please ensure templates are properly installed.</p>
            </body>
        </html>
        """)


@app.get("/setup", response_class=HTMLResponse)
async def setup_wizard(request: Request):
    """Serve the setup wizard interface."""
    if templates_dir.exists():
        return templates.TemplateResponse("setup.html", {"request": request})
    else:
        return HTMLResponse("""
        <html>
            <head><title>Setup Wizard - AI Meeting Notes</title></head>
            <body>
                <h1>Setup Wizard</h1>
                <p>Template directory not found. Please ensure templates are properly installed.</p>
                <p><a href="/">← Back to Main App</a></p>
            </body>
        </html>
        """)


# API Routes

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify all services are available."""
    services = {}
    
    # Check audio recorder
    try:
        services["audio_recorder"] = audio_recorder is not None
        if audio_recorder:
            services["blackhole_available"] = audio_recorder.check_blackhole_available()
    except Exception:
        services["audio_recorder"] = False
        services["blackhole_available"] = False
    
    # Check transcriber
    try:
        services["transcriber"] = transcriber is not None
    except Exception:
        services["transcriber"] = False
    
    # Check notes generator
    try:
        services["notes_generator"] = notes_generator is not None
        if notes_generator:
            services["ollama_available"] = notes_generator.check_ollama_available()
    except Exception:
        services["notes_generator"] = False
        services["ollama_available"] = False
    
    # Check file manager
    try:
        services["file_manager"] = file_manager is not None
        if file_manager:
            disk_check_result = file_manager.check_disk_space()
            # check_disk_space returns a tuple (bool, float, str)
            services["disk_space_ok"] = disk_check_result[0] if isinstance(disk_check_result, tuple) else disk_check_result
    except Exception:
        services["file_manager"] = False
        services["disk_space_ok"] = False
    
    all_healthy = all(services.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        services=services,
        message="All services operational" if all_healthy else "Some services unavailable"
    )


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current meeting session status."""
    global processing_pipeline
    
    if not processing_pipeline:
        raise HTTPException(status_code=500, detail="Processing pipeline not initialized")
    
    status_info = processing_pipeline.get_processing_status()
    
    if not status_info["has_session"]:
        return StatusResponse(
            status="idle",
            message="No active meeting session"
        )
    
    return StatusResponse(
        status=status_info["status"],
        message=status_info["message"],
        session_id=str(id(processing_pipeline.get_current_session())),
        recording_duration=status_info.get("recording_duration"),
        processing_progress=status_info["progress"],
        error_message=status_info.get("error_message")
    )


@app.post("/api/start-recording", response_model=StartRecordingResponse)
async def start_recording():
    """Start audio recording for a new meeting session."""
    global processing_pipeline
    
    if not processing_pipeline:
        raise HTTPException(status_code=500, detail="Processing pipeline not initialized")
    
    try:
        # Generate unique session ID
        session_id = f"session_{int(datetime.now().timestamp())}"
        
        # Start recording through pipeline
        session = await processing_pipeline.start_recording(session_id)
        
        logger.info(f"Started recording session {session_id}")
        
        return StartRecordingResponse(
            success=True,
            message="Recording started successfully",
            session_id=session_id
        )
        
    except ProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error starting recording: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/stop-recording", response_model=StopRecordingResponse)
async def stop_recording():
    """Stop audio recording and begin processing."""
    global processing_pipeline
    
    if not processing_pipeline:
        raise HTTPException(status_code=500, detail="Processing pipeline not initialized")
    
    try:
        # Stop recording and start processing through pipeline
        session = await processing_pipeline.stop_recording_and_process()
        
        logger.info(f"Stopped recording, processing started: {session.audio_file}")
        
        return StopRecordingResponse(
            success=True,
            message="Recording stopped, processing started",
            audio_file=session.audio_file
        )
        
    except ProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop recording: {str(e)}")


@app.get("/api/transcript", response_model=TranscriptResponse)
async def get_transcript():
    """Get transcription result for current session."""
    global processing_pipeline
    
    if not processing_pipeline:
        raise HTTPException(status_code=500, detail="Processing pipeline not initialized")
    
    current_session = processing_pipeline.get_current_session()
    
    if not current_session:
        return TranscriptResponse(
            transcript=None,
            available=False,
            message="No active session"
        )
    
    if current_session.transcript:
        return TranscriptResponse(
            transcript=current_session.transcript,
            available=True,
            message="Transcript available"
        )
    
    if current_session.status == "transcribing":
        return TranscriptResponse(
            transcript=None,
            available=False,
            message=f"Transcription in progress ({current_session.processing_progress:.0f}%)"
        )
    
    if current_session.status == "error":
        return TranscriptResponse(
            transcript=None,
            available=False,
            message=f"Transcription failed: {current_session.error_message}"
        )
    
    return TranscriptResponse(
        transcript=None,
        available=False,
        message="Transcript not yet available"
    )


@app.get("/api/notes", response_model=NotesResponse)
async def get_notes():
    """Get meeting notes for current session."""
    global processing_pipeline
    
    if not processing_pipeline:
        raise HTTPException(status_code=500, detail="Processing pipeline not initialized")
    
    current_session = processing_pipeline.get_current_session()
    
    if not current_session:
        return NotesResponse(
            notes=None,
            available=False,
            message="No active session"
        )
    
    if current_session.notes:
        return NotesResponse(
            notes=current_session.notes,
            available=True,
            message="Meeting notes available"
        )
    
    if current_session.status == "generating_notes":
        return NotesResponse(
            notes=None,
            available=False,
            message=f"Generating notes ({current_session.processing_progress:.0f}%)"
        )
    
    if current_session.status == "error":
        return NotesResponse(
            notes=None,
            available=False,
            message=f"Notes generation failed: {current_session.error_message}"
        )
    
    if not current_session.transcript:
        return NotesResponse(
            notes=None,
            available=False,
            message="Transcript not available yet"
        )
    
    return NotesResponse(
        notes=None,
        available=False,
        message="Notes not yet generated"
    )


@app.post("/api/clear")
async def clear_session():
    """Clear current session and cleanup files."""
    global processing_pipeline
    
    if not processing_pipeline:
        raise HTTPException(status_code=500, detail="Processing pipeline not initialized")
    
    try:
        processing_pipeline.clear_session()
        return {"success": True, "message": "Session cleared"}
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return {"success": False, "message": f"Error clearing session: {e}"}


# Retry endpoints for failed processing steps

@app.post("/api/retry-transcription")
async def retry_transcription():
    """Retry transcription for the current session."""
    global processing_pipeline
    
    if not processing_pipeline:
        raise HTTPException(status_code=500, detail="Processing pipeline not initialized")
    
    current_session = processing_pipeline.get_current_session()
    
    if not current_session:
        raise HTTPException(status_code=400, detail="No active session to retry")
    
    if current_session.status != "error":
        raise HTTPException(status_code=400, detail="Session is not in error state")
    
    if not current_session.audio_file:
        raise HTTPException(status_code=400, detail="No audio file available for retry")
    
    try:
        await processing_pipeline.retry_transcription(current_session)
        return {"success": True, "message": "Transcription retry started"}
    except ProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrying transcription: {e}")
        raise HTTPException(status_code=500, detail="Failed to retry transcription")


@app.post("/api/retry-notes")
async def retry_notes_generation():
    """Retry notes generation for the current session."""
    global processing_pipeline

    if not processing_pipeline:
        raise HTTPException(status_code=500, detail="Processing pipeline not initialized")

    current_session = processing_pipeline.get_current_session()

    if not current_session:
        raise HTTPException(status_code=400, detail="No active session to retry")

    if not current_session.transcript:
        raise HTTPException(status_code=400, detail="No transcript available for notes generation")

    try:
        await processing_pipeline.retry_notes_generation(current_session)
        return {"success": True, "message": "Notes generation retry started"}
    except ProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrying notes generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to retry notes generation")


@app.get("/api/export-prompt")
async def get_export_prompt():
    """Get the ChatGPT/Claude export prompt with transcript."""
    global processing_pipeline, notes_generator

    if not processing_pipeline:
        raise HTTPException(status_code=500, detail="Processing pipeline not initialized")

    if not notes_generator:
        raise HTTPException(status_code=500, detail="Notes generator not initialized")

    current_session = processing_pipeline.get_current_session()

    if not current_session:
        raise HTTPException(status_code=400, detail="No active session")

    if not current_session.transcript:
        raise HTTPException(status_code=400, detail="No transcript available yet")

    try:
        # Calculate meeting duration from transcript
        meeting_duration = current_session.transcript.duration / 60.0  # Convert to minutes

        # Get the export prompt
        export_prompt = notes_generator.get_export_prompt(
            current_session.transcript.full_text,
            meeting_duration
        )

        return {
            "success": True,
            "prompt": export_prompt,
            "transcript_duration": meeting_duration,
            "transcript_language": current_session.transcript.language
        }
    except Exception as e:
        logger.error(f"Error generating export prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate export prompt: {str(e)}")


# Setup and Validation Routes

@app.get("/api/setup/validate", response_model=SetupValidationResponse)
async def validate_setup():
    """Run complete setup validation and return detailed report."""
    global setup_validator
    
    if not setup_validator:
        raise HTTPException(status_code=500, detail="Setup validator not initialized")
    
    try:
        report = setup_validator.run_full_validation()
        return SetupValidationResponse(**report.to_dict())
    except Exception as e:
        logger.error(f"Error running setup validation: {e}")
        raise HTTPException(status_code=500, detail="Failed to run setup validation")


@app.get("/api/setup/validate/audio", response_model=SetupValidationResponse)
async def validate_audio_setup():
    """Validate only audio-related setup components."""
    global setup_validator
    
    if not setup_validator:
        raise HTTPException(status_code=500, detail="Setup validator not initialized")
    
    try:
        report = setup_validator.validate_audio_setup_only()
        return SetupValidationResponse(**report.to_dict())
    except Exception as e:
        logger.error(f"Error validating audio setup: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate audio setup")


@app.get("/api/setup/validate/llm", response_model=SetupValidationResponse)
async def validate_llm_setup():
    """Validate only LLM-related setup components."""
    global setup_validator
    
    if not setup_validator:
        raise HTTPException(status_code=500, detail="Setup validator not initialized")
    
    try:
        report = setup_validator.validate_llm_setup_only()
        return SetupValidationResponse(**report.to_dict())
    except Exception as e:
        logger.error(f"Error validating LLM setup: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate LLM setup")


@app.get("/api/setup/wizard", response_model=SetupWizardResponse)
async def get_setup_wizard():
    """Get step-by-step setup wizard instructions."""
    global setup_validator
    
    if not setup_validator:
        raise HTTPException(status_code=500, detail="Setup validator not initialized")
    
    try:
        steps = setup_validator.get_setup_wizard_steps()
        return SetupWizardResponse(steps=steps)
    except Exception as e:
        logger.error(f"Error getting setup wizard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get setup wizard")


@app.get("/api/setup/instructions/{component}")
async def get_setup_instructions(component: str):
    """Get detailed setup instructions for a specific component."""
    global setup_validator
    
    if not setup_validator:
        raise HTTPException(status_code=500, detail="Setup validator not initialized")
    
    instruction_methods = {
        "blackhole": setup_validator._get_blackhole_installation_instructions,
        "multi-output": setup_validator._get_multi_output_setup_instructions,
        "audio-permissions": setup_validator._get_audio_permissions_instructions,
        "ollama": setup_validator._get_ollama_installation_instructions,
        "model": setup_validator._get_model_download_instructions,
        "integration-test": setup_validator._get_integration_test_instructions
    }
    
    if component not in instruction_methods:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown component: {component}. Available: {list(instruction_methods.keys())}"
        )
    
    try:
        instructions = instruction_methods[component]()
        return {"component": component, "instructions": instructions}
    except Exception as e:
        logger.error(f"Error getting instructions for {component}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get instructions for {component}")


@app.get("/api/setup/system-info", response_model=SystemInfoResponse)
async def get_system_info_endpoint():
    """Get comprehensive system information for troubleshooting."""
    try:
        system_info = get_system_info()
        return SystemInfoResponse(system_info=system_info)
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system information")


@app.get("/api/setup/audio-devices")
async def get_audio_devices():
    """Get list of available audio devices for troubleshooting."""
    global audio_recorder
    
    if not audio_recorder:
        raise HTTPException(status_code=500, detail="Audio recorder not initialized")
    
    try:
        devices = audio_recorder.get_available_devices()
        blackhole_available = audio_recorder.check_blackhole_available()
        multi_output_available = audio_recorder.check_multi_output_setup()
        
        return {
            "devices": devices,
            "blackhole_available": blackhole_available,
            "multi_output_available": multi_output_available,
            "setup_instructions": audio_recorder.get_setup_instructions()
        }
    except Exception as e:
        logger.error(f"Error getting audio devices: {e}")
        raise HTTPException(status_code=500, detail="Failed to get audio devices")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "ai_meeting_notes.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.debug,
        log_level="info"
    )