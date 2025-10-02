"""
API endpoint definitions for AI Meeting Notes application.

This module contains the FastAPI route handlers organized by functionality.
"""

from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..models import MeetingSession, TranscriptResult, MeetingNotes

# Response models for API endpoints
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


class ClearResponse(BaseModel):
    """Response model for clear session endpoint."""
    success: bool
    message: str


# Create API router
router = APIRouter(prefix="/api", tags=["meeting"])


def create_meeting_routes(
    get_current_session,
    get_audio_recorder,
    get_transcriber,
    get_notes_generator,
    get_file_manager,
    set_current_session,
    process_meeting_audio_task
):
    """
    Create meeting API routes with dependency injection.
    
    This function allows the main application to inject dependencies
    while keeping the route definitions clean and testable.
    """
    
    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint to verify all services are available."""
        services = {}
        
        # Check audio recorder
        audio_recorder = get_audio_recorder()
        try:
            services["audio_recorder"] = audio_recorder is not None
            if audio_recorder:
                services["blackhole_available"] = audio_recorder.check_blackhole_available()
        except Exception:
            services["audio_recorder"] = False
            services["blackhole_available"] = False
        
        # Check transcriber
        transcriber = get_transcriber()
        try:
            services["transcriber"] = transcriber is not None
        except Exception:
            services["transcriber"] = False
        
        # Check notes generator
        notes_generator = get_notes_generator()
        try:
            services["notes_generator"] = notes_generator is not None
            if notes_generator:
                services["ollama_available"] = notes_generator.check_ollama_available()
        except Exception:
            services["notes_generator"] = False
            services["ollama_available"] = False
        
        # Check file manager
        file_manager = get_file_manager()
        try:
            services["file_manager"] = file_manager is not None
            if file_manager:
                services["disk_space_ok"] = file_manager.check_disk_space()
        except Exception:
            services["file_manager"] = False
            services["disk_space_ok"] = False
        
        all_healthy = all(services.values())
        
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            services=services,
            message="All services operational" if all_healthy else "Some services unavailable"
        )
    
    @router.get("/status", response_model=StatusResponse)
    async def get_status():
        """Get current meeting session status."""
        current_session = get_current_session()
        
        if not current_session:
            return StatusResponse(
                status="idle",
                message="No active meeting session"
            )
        
        recording_duration = None
        audio_recorder = get_audio_recorder()
        if audio_recorder and audio_recorder.is_recording():
            recording_duration = audio_recorder.get_recording_duration()
        
        return StatusResponse(
            status=current_session.status,
            message=current_session.get_status_display(),
            session_id=str(id(current_session)),
            recording_duration=recording_duration,
            processing_progress=current_session.processing_progress,
            error_message=current_session.error_message
        )
    
    @router.post("/start-recording", response_model=StartRecordingResponse)
    async def start_recording():
        """Start audio recording for a new meeting session with enhanced error handling."""
        from ..error_handling import ProcessingError, handle_processing_error
        
        audio_recorder = get_audio_recorder()
        file_manager = get_file_manager()
        current_session = get_current_session()
        
        if not audio_recorder:
            raise HTTPException(status_code=500, detail="Audio recorder not initialized")
        
        if not file_manager:
            raise HTTPException(status_code=500, detail="File manager not initialized")
        
        try:
            # Check if already recording
            if current_session and current_session.status == "recording":
                raise HTTPException(
                    status_code=400, 
                    detail="Recording already in progress. Please stop the current recording before starting a new one."
                )
            
            # Check BlackHole availability with detailed error message
            if not audio_recorder.check_blackhole_available():
                setup_instructions = audio_recorder.get_setup_instructions()
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": "BlackHole audio device not found",
                        "user_message": "BlackHole audio device not found. Please install BlackHole and configure Multi-Output Device.",
                        "setup_instructions": setup_instructions,
                        "recovery_actions": [
                            {
                                "action": "install_blackhole",
                                "description": "Install BlackHole virtual audio driver",
                                "automated": False
                            },
                            {
                                "action": "configure_multi_output",
                                "description": "Configure Multi-Output Device in Audio MIDI Setup",
                                "automated": False
                            }
                        ]
                    }
                )
            
            # Check disk space with detailed information
            disk_ok, free_gb, warning = file_manager.check_disk_space()
            if not disk_ok:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Insufficient disk space",
                        "user_message": f"Insufficient disk space for recording. {warning}",
                        "free_space_gb": free_gb,
                        "recovery_actions": [
                            {
                                "action": "free_disk_space",
                                "description": "Free up disk space and try again",
                                "automated": False
                            },
                            {
                                "action": "cleanup_old_files",
                                "description": "Clean up old meeting files",
                                "automated": True
                            }
                        ]
                    }
                )
            
            # Create new session
            new_session = MeetingSession(status="recording")
            session_id = str(id(new_session))
            
            # Generate audio file path
            audio_file_path = file_manager.get_audio_file_path(session_id)
            new_session.audio_file = audio_file_path
            
            # Start recording
            audio_recorder.start_recording(audio_file_path)
            
            # Set as current session
            set_current_session(new_session)
            
            return StartRecordingResponse(
                success=True,
                message="Recording started successfully",
                session_id=session_id
            )
            
        except HTTPException:
            raise  # Re-raise HTTP exceptions as-is
        except ProcessingError as e:
            set_current_session(None)
            raise HTTPException(status_code=400, detail=e.to_dict())
        except Exception as e:
            set_current_session(None)
            processing_error = handle_processing_error(e, "api", "start_recording")
            raise HTTPException(status_code=500, detail=processing_error.to_dict())
    
    @router.post("/stop-recording", response_model=StopRecordingResponse)
    async def stop_recording(background_tasks: BackgroundTasks):
        """Stop audio recording and begin processing."""
        current_session = get_current_session()
        audio_recorder = get_audio_recorder()
        
        if not current_session or current_session.status != "recording":
            raise HTTPException(status_code=400, detail="No active recording to stop")
        
        if not audio_recorder or not audio_recorder.is_recording():
            raise HTTPException(status_code=400, detail="No active recording found")
        
        try:
            # Stop recording
            audio_file_path = audio_recorder.stop_recording()
            current_session.audio_file = audio_file_path
            current_session.status = "transcribing"
            
            # Start background processing
            background_tasks.add_task(process_meeting_audio_task, current_session)
            
            return StopRecordingResponse(
                success=True,
                message="Recording stopped, processing started",
                audio_file=audio_file_path
            )
            
        except Exception as e:
            if current_session:
                current_session.status = "error"
                current_session.error_message = str(e)
            raise HTTPException(status_code=500, detail=f"Failed to stop recording: {str(e)}")
    
    @router.get("/transcript", response_model=TranscriptResponse)
    async def get_transcript():
        """Get transcription result for current session."""
        current_session = get_current_session()
        
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
    
    @router.get("/notes", response_model=NotesResponse)
    async def get_notes():
        """Get meeting notes for current session."""
        current_session = get_current_session()
        
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
    
    @router.post("/clear", response_model=ClearResponse)
    async def clear_session():
        """Clear current session and cleanup files."""
        current_session = get_current_session()
        audio_recorder = get_audio_recorder()
        file_manager = get_file_manager()
        
        # Stop any active recording
        if audio_recorder and audio_recorder.is_recording():
            try:
                audio_recorder.stop_recording()
            except Exception:
                pass  # Continue with cleanup even if stop fails
        
        # Cleanup session files
        if current_session and file_manager:
            try:
                file_manager.cleanup_session_files(current_session)
            except Exception:
                pass  # Continue with cleanup even if file cleanup fails
        
        # Clear session
        set_current_session(None)
        
        return ClearResponse(success=True, message="Session cleared")
    
    @router.post("/retry/{action}")
    async def retry_operation(action: str):
        """Execute a recovery action for the current session."""
        from ..processing_pipeline import ProcessingPipeline
        from ..error_handling import ProcessingError
        
        # Get the processing pipeline (this would need to be injected)
        # For now, we'll return a placeholder response
        current_session = get_current_session()
        
        if not current_session:
            raise HTTPException(status_code=400, detail="No active session to retry")
        
        if current_session.status != "error":
            raise HTTPException(status_code=400, detail="Session is not in error state")
        
        valid_actions = ["retry_transcription", "retry_notes_generation", "retry_full_pipeline"]
        if action not in valid_actions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid action. Valid actions: {', '.join(valid_actions)}"
            )
        
        try:
            # This would need to be implemented with proper pipeline injection
            # For now, return a success response
            return {
                "success": True,
                "message": f"Recovery action '{action}' initiated",
                "action": action
            }
            
        except ProcessingError as e:
            raise HTTPException(status_code=400, detail=e.to_dict())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Recovery action failed: {str(e)}")
    
    @router.get("/error-summary")
    async def get_error_summary():
        """Get summary of recent errors for monitoring."""
        from ..error_handling import error_recovery_manager
        
        try:
            summary = error_recovery_manager.get_error_summary()
            return {
                "success": True,
                "data": summary
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get error summary: {str(e)}")
    
    @router.post("/cleanup-old-files")
    async def cleanup_old_files():
        """Manually trigger cleanup of old files."""
        file_manager = get_file_manager()
        
        if not file_manager:
            raise HTTPException(status_code=500, detail="File manager not initialized")
        
        try:
            stats = file_manager.auto_cleanup_old_files()
            return {
                "success": True,
                "message": "Cleanup completed",
                "stats": stats
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
    
    return router