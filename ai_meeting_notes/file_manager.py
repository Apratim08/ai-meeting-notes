import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from .models import MeetingSession

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations, cleanup, and disk space monitoring for meeting sessions."""
    
    def __init__(self, base_dir: str = "meeting_data", max_file_size_mb: float = 500.0, 
                 auto_cleanup_days: int = 7, min_free_space_gb: float = 1.0):
        """
        Initialize FileManager with configuration.
        
        Args:
            base_dir: Base directory for storing meeting files
            max_file_size_mb: Maximum audio file size in MB before warning
            auto_cleanup_days: Days to keep files before auto cleanup
            min_free_space_gb: Minimum free space required in GB
        """
        self.base_dir = Path(base_dir)
        self.max_file_size_mb = max_file_size_mb
        self.auto_cleanup_days = auto_cleanup_days
        self.min_free_space_gb = min_free_space_gb
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.audio_dir = self.base_dir / "audio"
        self.temp_dir = self.base_dir / "temp"
        self.audio_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
    
    def get_audio_file_path(self, session_id: str) -> str:
        """Generate audio file path for a session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"meeting_{session_id}_{timestamp}.wav"
        return str(self.audio_dir / filename)
    
    def get_file_size_mb(self, filepath: str) -> float:
        """Get file size in MB."""
        try:
            if os.path.exists(filepath):
                size_bytes = os.path.getsize(filepath)
                return size_bytes / (1024 * 1024)
            return 0.0
        except OSError as e:
            logger.error(f"Error getting file size for {filepath}: {e}")
            return 0.0
    
    def check_file_size_warning(self, filepath: str) -> Optional[str]:
        """Check if file size exceeds warning threshold."""
        size_mb = self.get_file_size_mb(filepath)
        if size_mb > self.max_file_size_mb:
            return (f"Large file warning: {filepath} is {size_mb:.1f}MB. "
                   f"Processing may take longer than usual.")
        return None
    
    def check_disk_space(self) -> Tuple[bool, float, str]:
        """
        Check available disk space.
        
        Returns:
            Tuple of (has_enough_space, free_space_gb, warning_message)
        """
        try:
            total, used, free = shutil.disk_usage(self.base_dir)
            free_gb = free / (1024**3)
            
            if free_gb < self.min_free_space_gb:
                warning = (f"Low disk space: {free_gb:.1f}GB available. "
                          f"Minimum {self.min_free_space_gb}GB required.")
                return False, free_gb, warning
            
            return True, free_gb, ""
        except OSError as e:
            logger.error(f"Error checking disk space: {e}")
            return False, 0.0, f"Unable to check disk space: {e}"
    
    def estimate_processing_requirements(self, audio_duration: float) -> Dict[str, float]:
        """
        Estimate processing requirements for audio duration.
        
        Args:
            audio_duration: Duration in seconds
            
        Returns:
            Dictionary with estimated requirements
        """
        # Rough estimates based on typical processing
        duration_hours = audio_duration / 3600
        
        return {
            "transcription_time_minutes": duration_hours * 2.5,  # ~2.5 min per hour
            "notes_generation_time_minutes": 2.0,  # ~2 minutes regardless of length
            "total_time_minutes": duration_hours * 2.5 + 2.0,
            "temp_space_mb": duration_hours * 100,  # ~100MB per hour for temp files
        }
    
    def cleanup_session_files(self, session: MeetingSession, preserve_on_error: bool = True) -> bool:
        """
        Clean up files associated with a session.
        
        Args:
            session: MeetingSession to clean up
            preserve_on_error: If True, preserve files if session has errors
            
        Returns:
            True if cleanup was successful
        """
        try:
            # Don't cleanup if there was an error and preserve_on_error is True
            if preserve_on_error and session.status == "error":
                logger.info(f"Preserving files for error session: {session.audio_file}")
                return True
            
            files_cleaned = []
            
            # Clean up audio file
            if session.audio_file and os.path.exists(session.audio_file):
                os.remove(session.audio_file)
                files_cleaned.append(session.audio_file)
            
            # Clean up any temp files (look for files with similar timestamp)
            if session.audio_file:
                audio_path = Path(session.audio_file)
                # Extract timestamp from filename to find related temp files
                for temp_file in self.temp_dir.glob("*"):
                    if temp_file.is_file():
                        # Simple heuristic: if temp file was created around the same time
                        file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)
                        session_time = session.start_time
                        if abs((file_time - session_time).total_seconds()) < 3600:  # Within 1 hour
                            temp_file.unlink()
                            files_cleaned.append(str(temp_file))
            
            if files_cleaned:
                logger.info(f"Cleaned up files: {files_cleaned}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False
    
    def auto_cleanup_old_files(self) -> Dict[str, int]:
        """
        Automatically clean up files older than auto_cleanup_days.
        
        Returns:
            Dictionary with cleanup statistics
        """
        cutoff_date = datetime.now() - timedelta(days=self.auto_cleanup_days)
        stats = {"audio_files_removed": 0, "temp_files_removed": 0, "space_freed_mb": 0.0}
        
        try:
            # Clean up old audio files
            for audio_file in self.audio_dir.glob("*.wav"):
                if audio_file.is_file():
                    file_time = datetime.fromtimestamp(audio_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        size_mb = self.get_file_size_mb(str(audio_file))
                        audio_file.unlink()
                        stats["audio_files_removed"] += 1
                        stats["space_freed_mb"] += size_mb
            
            # Clean up old temp files
            for temp_file in self.temp_dir.glob("*"):
                if temp_file.is_file():
                    file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        size_mb = self.get_file_size_mb(str(temp_file))
                        temp_file.unlink()
                        stats["temp_files_removed"] += 1
                        stats["space_freed_mb"] += size_mb
            
            if stats["audio_files_removed"] > 0 or stats["temp_files_removed"] > 0:
                logger.info(f"Auto cleanup completed: {stats}")
            
        except Exception as e:
            logger.error(f"Error during auto cleanup: {e}")
        
        return stats
    
    def get_storage_info(self) -> Dict[str, any]:
        """Get comprehensive storage information."""
        has_space, free_gb, space_warning = self.check_disk_space()
        
        # Count files and calculate total size
        audio_files = list(self.audio_dir.glob("*.wav"))
        temp_files = list(self.temp_dir.glob("*"))
        
        total_audio_size_mb = sum(self.get_file_size_mb(str(f)) for f in audio_files)
        total_temp_size_mb = sum(self.get_file_size_mb(str(f)) for f in temp_files)
        
        return {
            "has_sufficient_space": has_space,
            "free_space_gb": free_gb,
            "space_warning": space_warning,
            "audio_files_count": len(audio_files),
            "temp_files_count": len(temp_files),
            "total_audio_size_mb": total_audio_size_mb,
            "total_temp_size_mb": total_temp_size_mb,
            "total_size_mb": total_audio_size_mb + total_temp_size_mb,
            "auto_cleanup_days": self.auto_cleanup_days,
            "max_file_size_mb": self.max_file_size_mb,
        }
    
    def manual_cleanup_all(self) -> Dict[str, int]:
        """
        Manually clean up all files (for user-initiated cleanup).
        
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {"audio_files_removed": 0, "temp_files_removed": 0, "space_freed_mb": 0.0}
        
        try:
            # Clean up all audio files
            for audio_file in self.audio_dir.glob("*.wav"):
                if audio_file.is_file():
                    size_mb = self.get_file_size_mb(str(audio_file))
                    audio_file.unlink()
                    stats["audio_files_removed"] += 1
                    stats["space_freed_mb"] += size_mb
            
            # Clean up all temp files
            for temp_file in self.temp_dir.glob("*"):
                if temp_file.is_file():
                    size_mb = self.get_file_size_mb(str(temp_file))
                    temp_file.unlink()
                    stats["temp_files_removed"] += 1
                    stats["space_freed_mb"] += size_mb
            
            logger.info(f"Manual cleanup completed: {stats}")
            
        except Exception as e:
            logger.error(f"Error during manual cleanup: {e}")
        
        return stats
    
    def validate_file_path(self, filepath: str) -> Tuple[bool, str]:
        """
        Validate that a file path is safe and accessible.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path = Path(filepath)
            
            # Check if path is within our managed directories
            if not (path.is_relative_to(self.base_dir)):
                return False, "File path is outside managed directory"
            
            # Check if parent directory exists
            if not path.parent.exists():
                return False, "Parent directory does not exist"
            
            # Check if we have write permissions
            if path.exists() and not os.access(path, os.W_OK):
                return False, "No write permission for file"
            
            if not path.exists() and not os.access(path.parent, os.W_OK):
                return False, "No write permission for directory"
            
            return True, ""
            
        except Exception as e:
            return False, f"Path validation error: {e}"