"""
Sequential processing pipeline for AI Meeting Notes.

Handles the complete workflow: audio recording → transcription → notes generation
with comprehensive error handling, retry mechanisms, progress tracking, and 
intermediate result preservation.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

from .models import MeetingSession, TranscriptResult, MeetingNotes
from .audio_recorder import AudioRecorder, AudioRecorderError
from .transcription import BatchTranscriber
from .notes_generator import NotesGenerator, OllamaError
from .file_manager import FileManager
from .config import config
from .error_handling import (
    ProcessingError, ErrorCategory, ErrorSeverity, ErrorContext,
    error_recovery_manager, handle_processing_error, preserve_intermediate_results,
    RetryConfig
)

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """
    Sequential processing pipeline that coordinates audio recording, transcription,
    and notes generation with comprehensive error handling and progress tracking.
    """
    
    def __init__(
        self,
        audio_recorder: AudioRecorder,
        transcriber: BatchTranscriber,
        notes_generator: NotesGenerator,
        file_manager: FileManager,
        progress_callback: Optional[Callable[[str, float, str], None]] = None
    ):
        """
        Initialize the processing pipeline.
        
        Args:
            audio_recorder: Audio recording service
            transcriber: Batch transcription service
            notes_generator: Notes generation service
            file_manager: File management service
            progress_callback: Optional callback for progress updates (status, progress, message)
        """
        self.audio_recorder = audio_recorder
        self.transcriber = transcriber
        self.notes_generator = notes_generator
        self.file_manager = file_manager
        self.progress_callback = progress_callback
        
        self._current_session: Optional[MeetingSession] = None
        self._is_processing = False
        
    def get_current_session(self) -> Optional[MeetingSession]:
        """Get the current processing session."""
        return self._current_session
    
    def is_processing(self) -> bool:
        """Check if pipeline is currently processing."""
        return self._is_processing
    
    def _update_progress(self, status: str, progress: float, message: str) -> None:
        """Update session progress and notify callback."""
        if self._current_session:
            self._current_session.status = status
            self._current_session.processing_progress = progress
            
        if self.progress_callback:
            self.progress_callback(status, progress, message)
            
        logger.info(f"Pipeline progress: {status} ({progress:.1f}%) - {message}")
    
    def _preserve_intermediate_results(self, session: MeetingSession, error: Exception) -> None:
        """Preserve intermediate results when an error occurs."""
        try:
            # Convert to ProcessingError if needed
            if isinstance(error, ProcessingError):
                processing_error = error
            else:
                processing_error = handle_processing_error(
                    error,
                    component="processing_pipeline",
                    operation="preserve_results",
                    session_id=str(id(session)) if session else None,
                    audio_file=session.audio_file if session else None
                )
            
            # Use centralized preservation logic
            preserve_intermediate_results(session, processing_error, self.file_manager)
            
        except Exception as preserve_error:
            logger.error(f"Failed to preserve intermediate results: {preserve_error}")
    
    async def start_recording(self, session_id: str) -> MeetingSession:
        """
        Start a new recording session with comprehensive error handling.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            MeetingSession: The created session
            
        Raises:
            ProcessingError: If recording cannot be started
        """
        context = ErrorContext(
            timestamp=datetime.now(),
            component="processing_pipeline",
            operation="start_recording",
            session_id=session_id
        )
        
        try:
            if self._current_session and self._current_session.status == "recording":
                raise ProcessingError(
                    "Recording already in progress",
                    category=ErrorCategory.USER_INPUT,
                    severity=ErrorSeverity.LOW,
                    context=context,
                    user_message="A recording is already in progress. Please stop the current recording before starting a new one."
                )
            
            # Pre-flight checks with specific error handling
            if not self.audio_recorder.check_blackhole_available():
                from .error_handling import RecoveryAction
                raise ProcessingError(
                    "BlackHole audio device not found",
                    category=ErrorCategory.AUDIO_DEVICE,
                    severity=ErrorSeverity.HIGH,
                    context=context,
                    user_message="BlackHole audio device not found. Please install BlackHole and configure Multi-Output Device.",
                    recovery_actions=[
                        RecoveryAction(
                            "manual", 
                            "Install BlackHole and configure Multi-Output Device", 
                            False
                        ),
                        RecoveryAction(
                            "retry", 
                            "Retry after fixing audio setup", 
                            False
                        )
                    ]
                )
            
            disk_ok, free_gb, warning = self.file_manager.check_disk_space()
            if not disk_ok:
                from .error_handling import RecoveryAction
                raise ProcessingError(
                    f"Insufficient disk space: {warning}",
                    category=ErrorCategory.SYSTEM_RESOURCE,
                    severity=ErrorSeverity.HIGH,
                    context=context,
                    user_message=f"Insufficient disk space for recording. {warning}",
                    recovery_actions=[
                        RecoveryAction(
                            "manual", 
                            "Free up disk space and try again", 
                            False
                        ),
                        RecoveryAction(
                            "cleanup", 
                            "Clean up old meeting files", 
                            True
                        )
                    ]
                )
            
            # Create new session
            session = MeetingSession(status="recording")
            audio_file_path = self.file_manager.get_audio_file_path(session_id)
            session.audio_file = audio_file_path
            context.audio_file = audio_file_path
            
            # Start recording with retry logic
            async def start_recording_operation():
                return self.audio_recorder.start_recording(audio_file_path)
            
            await error_recovery_manager.retry_with_backoff(
                start_recording_operation,
                "audio_recording",
                RetryConfig(max_attempts=2, base_delay=1.0),
                context
            )
            
            self._current_session = session
            self._update_progress("recording", 0.0, "Recording started")
            
            logger.info(f"Started recording session {session_id} to {audio_file_path}")
            return session
            
        except ProcessingError:
            raise  # Re-raise ProcessingError as-is
        except AudioRecorderError as e:
            raise handle_processing_error(e, "processing_pipeline", "start_recording", session_id)
        except Exception as e:
            logger.error(f"Unexpected error starting recording: {e}")
            raise handle_processing_error(e, "processing_pipeline", "start_recording", session_id)
    
    async def stop_recording_and_process(self) -> MeetingSession:
        """
        Stop recording and begin sequential processing.
        
        Returns:
            MeetingSession: The session being processed
            
        Raises:
            ProcessingError: If recording cannot be stopped or processing fails
        """
        if not self._current_session or self._current_session.status != "recording":
            raise ProcessingError("No active recording to stop")
        
        if not self.audio_recorder.is_recording():
            raise ProcessingError("No active recording found")
        
        try:
            # Stop recording
            audio_file_path = self.audio_recorder.stop_recording()
            self._current_session.audio_file = audio_file_path
            
            recording_duration = self.audio_recorder.get_recording_duration()
            logger.info(f"Recording stopped. Duration: {recording_duration:.1f}s, File: {audio_file_path}")
            
            # Check file size and warn if large
            size_warning = self.file_manager.check_file_size_warning(audio_file_path)
            if size_warning:
                logger.warning(size_warning)
            
            # Start background processing - use asyncio.ensure_future for Python 3.9 compatibility
            asyncio.ensure_future(self._process_audio_pipeline(self._current_session))
            
            return self._current_session
            
        except Exception as e:
            if self._current_session:
                self._preserve_intermediate_results(self._current_session, e)
            logger.error(f"Error stopping recording: {e}")
            raise ProcessingError(f"Failed to stop recording: {e}")
    
    async def _process_audio_pipeline(self, session: MeetingSession) -> None:
        """
        Execute the complete processing pipeline: transcription → notes generation.
        
        Args:
            session: The session to process
        """
        self._is_processing = True
        
        try:
            # Phase 1: Transcription
            await self._transcription_phase(session)
            
            # Phase 2: Notes Generation
            await self._notes_generation_phase(session)
            
            # Phase 3: Completion and Cleanup
            await self._completion_phase(session)
            
        except Exception as e:
            logger.error(f"Processing pipeline failed: {e}")
            self._preserve_intermediate_results(session, e)
        finally:
            self._is_processing = False
    
    async def _transcription_phase(self, session: MeetingSession) -> None:
        """Execute the transcription phase with comprehensive error handling and retry logic."""
        context = ErrorContext(
            timestamp=datetime.now(),
            component="processing_pipeline",
            operation="transcription_phase",
            session_id=str(id(session)),
            audio_file=session.audio_file
        )
        
        try:
            self._update_progress("transcribing", 0.0, "Starting transcription...")
            
            if not session.audio_file or not Path(session.audio_file).exists():
                raise ProcessingError(
                    "Audio file not found for transcription",
                    category=ErrorCategory.FILE_SYSTEM,
                    severity=ErrorSeverity.HIGH,
                    context=context,
                    user_message="Audio file is missing. Please try recording again."
                )
            
            # Estimate processing time for user feedback
            audio_duration = self._get_audio_duration(session.audio_file)
            estimated_time = self.transcriber.estimate_processing_time(audio_duration)
            
            self._update_progress(
                "transcribing", 
                5.0, 
                f"Transcribing audio... (estimated {estimated_time/60:.1f} minutes)"
            )
            
            # Perform transcription with retry logic
            async def transcription_operation():
                return await self._transcribe_with_progress_monitoring(session.audio_file)
            
            transcript_result = await error_recovery_manager.retry_with_backoff(
                transcription_operation,
                "transcription",
                RetryConfig(max_attempts=3, base_delay=2.0),
                context
            )
            
            # Validate transcription result
            if not transcript_result or not transcript_result.full_text.strip():
                raise ProcessingError(
                    "Transcription produced no usable text",
                    category=ErrorCategory.TRANSCRIPTION,
                    severity=ErrorSeverity.MEDIUM,
                    context=context,
                    user_message="No speech was detected in the audio. Please check that the recording contains clear speech."
                )
            
            # Store result
            session.transcript = transcript_result

            # Save transcript to file for debugging
            try:
                session_id = str(id(session))
                transcript_path = self.file_manager.save_transcript(transcript_result, session_id)
                logger.info(f"Transcript saved to: {transcript_path}")
            except Exception as e:
                logger.warning(f"Failed to save transcript file: {e}")

            self._update_progress("transcribing", 100.0, "Transcription completed")

            logger.info(
                f"Transcription completed: {len(transcript_result.segments)} segments, "
                f"avg confidence: {transcript_result.average_confidence:.2f}, "
                f"language: {transcript_result.language}"
            )
            
        except ProcessingError:
            raise  # Re-raise ProcessingError as-is
        except Exception as e:
            logger.error(f"Transcription phase failed: {e}")
            raise handle_processing_error(e, "processing_pipeline", "transcription_phase", 
                                        str(id(session)), session.audio_file)
    
    async def _transcribe_with_progress_monitoring(self, audio_file: str) -> TranscriptResult:
        """Transcribe audio with progress monitoring."""
        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Start transcription
        transcription_task = loop.run_in_executor(
            None, 
            self.transcriber.transcribe_file, 
            audio_file
        )
        
        # Monitor progress
        while not transcription_task.done():
            progress = self.transcriber.get_progress()
            # Map transcriber progress (0-1) to our phase progress (5-95)
            phase_progress = 5.0 + (progress * 90.0)
            
            self._update_progress(
                "transcribing", 
                phase_progress, 
                f"Processing audio... ({progress*100:.0f}%)"
            )
            
            await asyncio.sleep(1.0)  # Update every second
        
        # Get result
        return await transcription_task
    
    async def _notes_generation_phase(self, session: MeetingSession) -> None:
        """Execute the notes generation phase with comprehensive error handling and retry logic."""
        context = ErrorContext(
            timestamp=datetime.now(),
            component="processing_pipeline",
            operation="notes_generation_phase",
            session_id=str(id(session)),
            audio_file=session.audio_file
        )
        
        try:
            if not session.transcript:
                raise ProcessingError(
                    "No transcript available for notes generation",
                    category=ErrorCategory.TRANSCRIPTION,
                    severity=ErrorSeverity.HIGH,
                    context=context,
                    user_message="Transcript is not available. Please retry transcription first."
                )
            
            self._update_progress("generating_notes", 0.0, "Starting notes generation...")
            
            # Check Ollama availability with detailed error handling
            if not self.notes_generator.check_ollama_available():
                raise ProcessingError(
                    "Ollama not available",
                    category=ErrorCategory.NOTES_GENERATION,
                    severity=ErrorSeverity.HIGH,
                    context=context,
                    user_message="Ollama AI service is not available. Please ensure Ollama is running and the required model is installed."
                )
            
            # Validate transcript content
            if len(session.transcript.full_text.strip()) < 50:
                raise ProcessingError(
                    "Transcript too short for meaningful notes generation",
                    category=ErrorCategory.TRANSCRIPTION,
                    severity=ErrorSeverity.MEDIUM,
                    context=context,
                    user_message="The transcript is too short to generate meaningful meeting notes. Please ensure the recording contains sufficient speech content."
                )
            
            # Calculate meeting duration from transcript
            meeting_duration = session.transcript.duration / 60.0  # Convert to minutes
            
            self._update_progress("generating_notes", 10.0, "Generating structured notes...")
            
            # Generate notes with retry logic
            async def notes_generation_operation():
                return await self._generate_notes_with_progress_monitoring(
                    session.transcript.full_text, 
                    meeting_duration
                )
            
            meeting_notes = await error_recovery_manager.retry_with_backoff(
                notes_generation_operation,
                "notes_generation",
                RetryConfig(max_attempts=2, base_delay=5.0),
                context
            )
            
            # Validate notes result
            if not meeting_notes or not meeting_notes.summary.strip():
                logger.warning("Notes generation produced minimal content")
                # Don't fail completely, but log the issue
            
            # Store result
            session.notes = meeting_notes
            self._update_progress("generating_notes", 100.0, "Notes generation completed")
            
            logger.info(
                f"Notes generation completed: {len(meeting_notes.agenda_items)} agenda items, "
                f"{len(meeting_notes.action_items)} action items, "
                f"{len(meeting_notes.participants)} participants"
            )
            
        except ProcessingError:
            raise  # Re-raise ProcessingError as-is
        except Exception as e:
            logger.error(f"Notes generation phase failed: {e}")
            raise handle_processing_error(e, "processing_pipeline", "notes_generation_phase", 
                                        str(id(session)), session.audio_file)
    
    async def _generate_notes_with_progress_monitoring(
        self, 
        transcript: str, 
        meeting_duration: float
    ) -> MeetingNotes:
        """Generate notes with progress monitoring."""
        # Run notes generation in thread pool
        loop = asyncio.get_event_loop()
        
        # Start notes generation
        notes_task = loop.run_in_executor(
            None,
            self.notes_generator.generate_notes,
            transcript,
            meeting_duration
        )
        
        # Monitor progress (notes generation doesn't have fine-grained progress)
        start_time = time.time()
        while not notes_task.done():
            elapsed = time.time() - start_time
            # Estimate progress based on elapsed time (rough estimate: 2-3 minutes total)
            estimated_total = 150  # 2.5 minutes
            progress = min(90.0, (elapsed / estimated_total) * 90.0)
            
            self._update_progress(
                "generating_notes",
                10.0 + progress,
                f"Processing transcript with AI... ({elapsed:.0f}s elapsed)"
            )
            
            await asyncio.sleep(2.0)  # Update every 2 seconds
        
        # Get result
        return await notes_task
    
    async def _completion_phase(self, session: MeetingSession) -> None:
        """Execute the completion phase of the pipeline."""
        try:
            session.status = "completed"
            session.processing_progress = 100.0
            
            self._update_progress("completed", 100.0, "Meeting notes ready")
            
            # Optional cleanup if configured
            if config.files.cleanup_after_success:
                try:
                    # Give user a moment to see completion before cleanup
                    await asyncio.sleep(2.0)
                    
                    cleanup_success = self.file_manager.cleanup_session_files(
                        session, 
                        preserve_on_error=False
                    )
                    
                    if cleanup_success:
                        logger.info("Session files cleaned up after successful processing")
                    else:
                        logger.warning("File cleanup failed, but processing completed successfully")
                        
                except Exception as cleanup_error:
                    logger.error(f"Cleanup error (processing still successful): {cleanup_error}")
            
            logger.info("Processing pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Completion phase error: {e}")
            # Don't raise here - processing was successful, just cleanup failed
    
    def _get_audio_duration(self, audio_file: str) -> float:
        """Get audio file duration in seconds."""
        try:
            import wave
            with wave.open(audio_file, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                return frames / float(sample_rate)
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
            return 0.0
    
    async def retry_transcription(self, session: MeetingSession) -> None:
        """
        Retry transcription for a failed session with enhanced error handling.
        
        Args:
            session: Session with failed transcription
        """
        context = ErrorContext(
            timestamp=datetime.now(),
            component="processing_pipeline",
            operation="retry_transcription",
            session_id=str(id(session)),
            audio_file=session.audio_file
        )
        
        try:
            if not session.audio_file or not Path(session.audio_file).exists():
                raise ProcessingError(
                    "Audio file not available for retry",
                    category=ErrorCategory.FILE_SYSTEM,
                    severity=ErrorSeverity.HIGH,
                    context=context,
                    user_message="The audio file is no longer available. Please record the meeting again."
                )
            
            logger.info(f"Retrying transcription for session: {session.audio_file}")
            
            # Reset session state for retry
            session.status = "transcribing"
            session.error_message = None
            session.processing_progress = 0.0
            
            self._current_session = session
            self._is_processing = True
            
            try:
                await self._transcription_phase(session)
                
                # If transcription succeeds, continue with notes generation
                if session.transcript:
                    await self._notes_generation_phase(session)
                    await self._completion_phase(session)
                    
            finally:
                self._is_processing = False
                
        except ProcessingError:
            self._preserve_intermediate_results(session, ProcessingError)
            raise
        except Exception as e:
            processing_error = handle_processing_error(e, "processing_pipeline", "retry_transcription", 
                                                     str(id(session)), session.audio_file)
            self._preserve_intermediate_results(session, processing_error)
            raise processing_error
    
    async def retry_notes_generation(self, session: MeetingSession) -> None:
        """
        Retry notes generation for a session with successful transcription.
        
        Args:
            session: Session with successful transcription but failed notes generation
        """
        context = ErrorContext(
            timestamp=datetime.now(),
            component="processing_pipeline",
            operation="retry_notes_generation",
            session_id=str(id(session)),
            audio_file=session.audio_file
        )
        
        try:
            if not session.transcript:
                raise ProcessingError(
                    "No transcript available for notes generation retry",
                    category=ErrorCategory.TRANSCRIPTION,
                    severity=ErrorSeverity.HIGH,
                    context=context,
                    user_message="Transcript is not available. Please retry transcription first."
                )
            
            logger.info("Retrying notes generation for session")
            
            # Reset session state for retry
            session.status = "generating_notes"
            session.error_message = None
            session.processing_progress = 0.0
            
            self._current_session = session
            self._is_processing = True
            
            try:
                await self._notes_generation_phase(session)
                await self._completion_phase(session)
            finally:
                self._is_processing = False
                
        except ProcessingError:
            self._preserve_intermediate_results(session, ProcessingError)
            raise
        except Exception as e:
            processing_error = handle_processing_error(e, "processing_pipeline", "retry_notes_generation", 
                                                     str(id(session)), session.audio_file)
            self._preserve_intermediate_results(session, processing_error)
            raise processing_error
    
    def clear_session(self) -> None:
        """Clear the current session and stop any active recording."""
        try:
            # Stop recording if active
            if self.audio_recorder.is_recording():
                self.audio_recorder.stop_recording()
                logger.info("Stopped active recording during session clear")
        except Exception as e:
            logger.error(f"Error stopping recording during clear: {e}")
        
        # Cleanup session files if exists
        if self._current_session:
            try:
                self.file_manager.cleanup_session_files(self._current_session)
                logger.info("Cleaned up session files")
            except Exception as e:
                logger.error(f"Error cleaning up session files: {e}")
        
        # Clear session
        self._current_session = None
        self._is_processing = False
        
        logger.info("Session cleared")
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get comprehensive processing status information with error details."""
        if not self._current_session:
            return {
                "has_session": False,
                "status": "idle",
                "message": "No active session"
            }
        
        session = self._current_session
        
        status_info = {
            "has_session": True,
            "status": session.status,
            "progress": session.processing_progress,
            "message": session.get_status_display(),
            "error_message": session.error_message,
            "has_audio": session.audio_file is not None,
            "has_transcript": session.transcript is not None,
            "has_notes": session.notes is not None,
            "is_processing": self._is_processing,
            "start_time": session.start_time.isoformat(),
            "recording_duration": (
                self.audio_recorder.get_recording_duration() 
                if self.audio_recorder.is_recording() 
                else None
            )
        }
        
        # Add error recovery information if in error state
        if session.status == "error":
            status_info["recovery_options"] = self._get_recovery_options(session)
            status_info["error_summary"] = error_recovery_manager.get_error_summary()
        
        return status_info
    
    def _get_recovery_options(self, session: MeetingSession) -> List[Dict[str, Any]]:
        """Get available recovery options for a failed session."""
        options = []
        
        # Always allow session clearing
        options.append({
            "action": "clear_session",
            "label": "Clear Session",
            "description": "Clear the current session and start over",
            "automated": False
        })
        
        # Transcription retry if we have audio but no transcript
        if session.audio_file and Path(session.audio_file).exists() and not session.transcript:
            options.append({
                "action": "retry_transcription",
                "label": "Retry Transcription",
                "description": "Retry converting the audio to text",
                "automated": True
            })
        
        # Notes generation retry if we have transcript but no notes
        if session.transcript and not session.notes:
            options.append({
                "action": "retry_notes_generation",
                "label": "Retry Notes Generation",
                "description": "Retry generating meeting notes from the transcript",
                "automated": True
            })
        
        # Full pipeline retry if we have audio
        if session.audio_file and Path(session.audio_file).exists():
            options.append({
                "action": "retry_full_pipeline",
                "label": "Retry Full Processing",
                "description": "Retry the complete processing pipeline from the beginning",
                "automated": True
            })
        
        return options
    
    async def execute_recovery_action(self, action: str) -> Dict[str, Any]:
        """
        Execute a recovery action for the current session.
        
        Args:
            action: The recovery action to execute
            
        Returns:
            Dictionary with execution result
        """
        if not self._current_session:
            return {"success": False, "message": "No active session to recover"}
        
        session = self._current_session
        
        try:
            if action == "clear_session":
                self.clear_session()
                return {"success": True, "message": "Session cleared successfully"}
            
            elif action == "retry_transcription":
                await self.retry_transcription(session)
                return {"success": True, "message": "Transcription retry completed"}
            
            elif action == "retry_notes_generation":
                await self.retry_notes_generation(session)
                return {"success": True, "message": "Notes generation retry completed"}
            
            elif action == "retry_full_pipeline":
                # Reset session and retry from transcription
                session.transcript = None
                session.notes = None
                await self.retry_transcription(session)
                return {"success": True, "message": "Full pipeline retry completed"}
            
            else:
                return {"success": False, "message": f"Unknown recovery action: {action}"}
                
        except ProcessingError as e:
            return {
                "success": False, 
                "message": e.user_message,
                "technical_details": e.technical_details,
                "recovery_actions": [action.to_dict() for action in e.recovery_actions]
            }
        except Exception as e:
            return {"success": False, "message": f"Recovery action failed: {str(e)}"}