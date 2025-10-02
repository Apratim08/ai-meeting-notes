"""
Comprehensive error handling and recovery mechanisms for AI Meeting Notes.

This module provides centralized error handling, retry logic, user-friendly error messages,
and graceful degradation capabilities for all processing components.
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for categorizing failures."""
    LOW = "low"           # Minor issues, system can continue
    MEDIUM = "medium"     # Significant issues, some functionality affected
    HIGH = "high"         # Major issues, core functionality affected
    CRITICAL = "critical" # System cannot function


class ErrorCategory(Enum):
    """Categories of errors for better handling and user guidance."""
    AUDIO_DEVICE = "audio_device"
    AUDIO_RECORDING = "audio_recording"
    TRANSCRIPTION = "transcription"
    NOTES_GENERATION = "notes_generation"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    SYSTEM_RESOURCE = "system_resource"
    USER_INPUT = "user_input"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for an error occurrence."""
    timestamp: datetime
    component: str
    operation: str
    session_id: Optional[str] = None
    audio_file: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class RecoveryAction:
    """Represents a recovery action that can be taken for an error."""
    action_type: str  # "retry", "fallback", "manual", "skip"
    description: str
    automated: bool
    parameters: Optional[Dict[str, Any]] = None


class ProcessingError(Exception):
    """
    Enhanced processing error with categorization and recovery information.
    """
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        recovery_actions: Optional[List[RecoveryAction]] = None,
        user_message: Optional[str] = None,
        technical_details: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext(
            timestamp=datetime.now(),
            component="unknown",
            operation="unknown"
        )
        self.recovery_actions = recovery_actions or []
        self.user_message = user_message or message
        self.technical_details = technical_details
        self.original_exception = original_exception
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses."""
        return {
            "message": self.message,
            "user_message": self.user_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.context.timestamp.isoformat(),
            "component": self.context.component,
            "operation": self.context.operation,
            "recovery_actions": [
                {
                    "type": action.action_type,
                    "description": action.description,
                    "automated": action.automated
                }
                for action in self.recovery_actions
            ],
            "technical_details": self.technical_details
        }


class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.base_delay
        
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay


class ErrorRecoveryManager:
    """
    Manages error recovery strategies and retry mechanisms.
    """
    
    def __init__(self):
        self.error_history: List[ProcessingError] = []
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.retry_configs: Dict[str, RetryConfig] = {
            "transcription": RetryConfig(max_attempts=3, base_delay=2.0),
            "notes_generation": RetryConfig(max_attempts=2, base_delay=5.0),
            "audio_recording": RetryConfig(max_attempts=2, base_delay=1.0),
            "file_operations": RetryConfig(max_attempts=3, base_delay=0.5)
        }
    
    def register_recovery_strategy(
        self, 
        category: ErrorCategory, 
        strategy: Callable[[ProcessingError], List[RecoveryAction]]
    ) -> None:
        """Register a recovery strategy for an error category."""
        self.recovery_strategies[category] = strategy
    
    def handle_error(
        self, 
        error: Union[Exception, ProcessingError], 
        context: Optional[ErrorContext] = None
    ) -> ProcessingError:
        """
        Handle an error and determine recovery actions.
        
        Args:
            error: The error that occurred
            context: Additional context about the error
            
        Returns:
            ProcessingError with recovery information
        """
        if isinstance(error, ProcessingError):
            processing_error = error
        else:
            processing_error = self._convert_to_processing_error(error, context)
        
        # Add to error history
        self.error_history.append(processing_error)
        
        # Determine recovery actions
        if processing_error.category in self.recovery_strategies:
            recovery_actions = self.recovery_strategies[processing_error.category](processing_error)
            processing_error.recovery_actions.extend(recovery_actions)
        
        # Log the error
        self._log_error(processing_error)
        
        return processing_error
    
    def _convert_to_processing_error(
        self, 
        error: Exception, 
        context: Optional[ErrorContext]
    ) -> ProcessingError:
        """Convert a generic exception to a ProcessingError."""
        category, severity = self._categorize_error(error)
        user_message = self._generate_user_message(error, category)
        recovery_actions = self._generate_recovery_actions(error, category)
        
        return ProcessingError(
            message=str(error),
            category=category,
            severity=severity,
            context=context,
            recovery_actions=recovery_actions,
            user_message=user_message,
            technical_details=f"{type(error).__name__}: {str(error)}",
            original_exception=error
        )
    
    def _categorize_error(self, error: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Categorize an error based on its type and message."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Audio device errors
        if "blackhole" in error_str or "audio device" in error_str:
            return ErrorCategory.AUDIO_DEVICE, ErrorSeverity.HIGH
        
        # Audio recording errors
        if "recording" in error_str or error_type == "AudioRecorderError":
            return ErrorCategory.AUDIO_RECORDING, ErrorSeverity.MEDIUM
        
        # Transcription errors
        if "whisper" in error_str or "transcription" in error_str:
            return ErrorCategory.TRANSCRIPTION, ErrorSeverity.MEDIUM
        
        # Notes generation errors
        if "ollama" in error_str or "notes" in error_str or error_type == "OllamaError":
            return ErrorCategory.NOTES_GENERATION, ErrorSeverity.MEDIUM
        
        # File system errors
        if "file" in error_str or "disk" in error_str or error_type in ["FileNotFoundError", "PermissionError"]:
            return ErrorCategory.FILE_SYSTEM, ErrorSeverity.MEDIUM
        
        # Network errors
        if "connection" in error_str or "timeout" in error_str or "network" in error_str:
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        
        # System resource errors
        if "memory" in error_str or "space" in error_str:
            return ErrorCategory.SYSTEM_RESOURCE, ErrorSeverity.HIGH
        
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
    
    def _generate_user_message(self, error: Exception, category: ErrorCategory) -> str:
        """Generate a user-friendly error message."""
        error_str = str(error).lower()
        
        messages = {
            ErrorCategory.AUDIO_DEVICE: (
                "Audio device not available. Please check that BlackHole is installed "
                "and your Multi-Output Device is configured correctly."
            ),
            ErrorCategory.AUDIO_RECORDING: (
                "Recording failed. Please check your audio setup and try again."
            ),
            ErrorCategory.TRANSCRIPTION: (
                "Speech-to-text conversion failed. The audio file has been preserved "
                "and you can retry transcription."
            ),
            ErrorCategory.NOTES_GENERATION: (
                "AI notes generation failed. Please check that Ollama is running "
                "and try again. Your transcript has been preserved."
            ),
            ErrorCategory.FILE_SYSTEM: (
                "File operation failed. Please check disk space and file permissions."
            ),
            ErrorCategory.NETWORK: (
                "Network connection failed. Please check your internet connection "
                "and try again."
            ),
            ErrorCategory.SYSTEM_RESOURCE: (
                "System resources insufficient. Please free up disk space or memory "
                "and try again."
            )
        }
        
        return messages.get(category, f"An error occurred: {str(error)}")
    
    def _generate_recovery_actions(
        self, 
        error: Exception, 
        category: ErrorCategory
    ) -> List[RecoveryAction]:
        """Generate recovery actions based on error category."""
        actions = []
        
        if category == ErrorCategory.AUDIO_DEVICE:
            actions.extend([
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
            ])
        
        elif category == ErrorCategory.TRANSCRIPTION:
            actions.extend([
                RecoveryAction(
                    "retry", 
                    "Retry transcription with same model", 
                    True
                ),
                RecoveryAction(
                    "fallback", 
                    "Try with a different transcription model", 
                    True
                )
            ])
        
        elif category == ErrorCategory.NOTES_GENERATION:
            actions.extend([
                RecoveryAction(
                    "manual", 
                    "Check Ollama is running and model is available", 
                    False
                ),
                RecoveryAction(
                    "retry", 
                    "Retry notes generation", 
                    True
                )
            ])
        
        elif category == ErrorCategory.FILE_SYSTEM:
            actions.extend([
                RecoveryAction(
                    "manual", 
                    "Check disk space and file permissions", 
                    False
                ),
                RecoveryAction(
                    "retry", 
                    "Retry operation", 
                    True
                )
            ])
        
        else:
            actions.append(
                RecoveryAction(
                    "retry", 
                    "Retry the operation", 
                    True
                )
            )
        
        return actions
    
    def _log_error(self, error: ProcessingError) -> None:
        """Log error with appropriate level based on severity."""
        log_message = (
            f"[{error.category.value.upper()}] {error.message} "
            f"(Component: {error.context.component}, Operation: {error.context.operation})"
        )
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        if error.technical_details:
            logger.debug(f"Technical details: {error.technical_details}")
    
    async def retry_with_backoff(
        self,
        operation: Callable,
        operation_name: str,
        retry_config: Optional[RetryConfig] = None,
        context: Optional[ErrorContext] = None
    ) -> Any:
        """
        Execute an operation with retry logic and exponential backoff.
        
        Args:
            operation: The operation to retry
            operation_name: Name of the operation for logging
            retry_config: Retry configuration (uses default if None)
            context: Error context for logging
            
        Returns:
            Result of the successful operation
            
        Raises:
            ProcessingError: If all retry attempts fail
        """
        config = retry_config or self.retry_configs.get(operation_name, RetryConfig())
        last_error = None
        
        for attempt in range(1, config.max_attempts + 1):
            try:
                logger.info(f"Attempting {operation_name} (attempt {attempt}/{config.max_attempts})")
                result = await operation() if hasattr(operation, '__call__') else operation
                
                if attempt > 1:
                    logger.info(f"{operation_name} succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_error = e
                
                if attempt < config.max_attempts:
                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"{operation_name} failed on attempt {attempt}: {e}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"{operation_name} failed after {config.max_attempts} attempts: {e}")
        
        # All attempts failed
        error_context = context or ErrorContext(
            timestamp=datetime.now(),
            component=operation_name,
            operation="retry_operation"
        )
        
        processing_error = self.handle_error(last_error, error_context)
        raise processing_error
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of recent errors for monitoring."""
        if not self.error_history:
            return {"total_errors": 0, "recent_errors": []}
        
        recent_errors = self.error_history[-10:]  # Last 10 errors
        
        category_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "last_error": recent_errors[-1].to_dict() if recent_errors else None
        }
    
    def clear_error_history(self) -> None:
        """Clear the error history (useful for testing or maintenance)."""
        self.error_history.clear()
        logger.info("Error history cleared")


# Global error recovery manager instance
error_recovery_manager = ErrorRecoveryManager()


def handle_processing_error(
    error: Union[Exception, ProcessingError],
    component: str,
    operation: str,
    session_id: Optional[str] = None,
    audio_file: Optional[str] = None,
    additional_data: Optional[Dict[str, Any]] = None
) -> ProcessingError:
    """
    Convenience function to handle processing errors with context.
    
    Args:
        error: The error that occurred
        component: Component where error occurred
        operation: Operation that failed
        session_id: Session ID if applicable
        audio_file: Audio file path if applicable
        additional_data: Additional context data
        
    Returns:
        ProcessingError with recovery information
    """
    context = ErrorContext(
        timestamp=datetime.now(),
        component=component,
        operation=operation,
        session_id=session_id,
        audio_file=audio_file,
        additional_data=additional_data
    )
    
    return error_recovery_manager.handle_error(error, context)


def preserve_intermediate_results(
    session: 'MeetingSession',
    error: ProcessingError,
    file_manager: Optional['FileManager'] = None
) -> None:
    """
    Preserve intermediate results when an error occurs.
    
    Args:
        session: The meeting session with intermediate results
        error: The error that occurred
        file_manager: File manager for preservation operations
    """
    try:
        preservation_log = []
        
        # Always preserve the audio file on error
        if session.audio_file and Path(session.audio_file).exists():
            preservation_log.append(f"Audio file: {session.audio_file}")
        
        # If we have a transcript, ensure it's preserved in session
        if session.transcript:
            preservation_log.append(f"Transcript: {len(session.transcript.full_text)} characters")
        
        # Update session with error details
        session.status = "error"
        session.error_message = error.user_message
        
        # Log preservation details
        if preservation_log:
            logger.info(f"Preserved intermediate results: {', '.join(preservation_log)}")
        
        # Add preservation info to error context
        if error.context.additional_data is None:
            error.context.additional_data = {}
        error.context.additional_data["preserved_results"] = preservation_log
        
    except Exception as preserve_error:
        logger.error(f"Failed to preserve intermediate results: {preserve_error}")