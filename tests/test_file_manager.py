import os
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pytest

from ai_meeting_notes.file_manager import FileManager
from ai_meeting_notes.models import MeetingSession, MeetingInfo, MeetingNotes


class TestFileManager:
    """Test suite for FileManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def file_manager(self, temp_dir):
        """Create FileManager instance with temporary directory."""
        return FileManager(
            base_dir=temp_dir,
            max_file_size_mb=10.0,  # Small size for testing
            auto_cleanup_days=1,
            min_free_space_gb=0.1  # Small requirement for testing
        )
    
    @pytest.fixture
    def sample_session(self):
        """Create a sample meeting session for testing."""
        return MeetingSession(
            start_time=datetime.now(),
            status="completed"
        )
    
    def test_initialization(self, temp_dir):
        """Test FileManager initialization creates required directories."""
        fm = FileManager(base_dir=temp_dir)
        
        assert fm.base_dir.exists()
        assert fm.audio_dir.exists()
        assert fm.temp_dir.exists()
        assert fm.max_file_size_mb == 500.0  # default value
        assert fm.auto_cleanup_days == 7  # default value
    
    def test_get_audio_file_path(self, file_manager):
        """Test audio file path generation."""
        session_id = "test_session_123"
        file_path = file_manager.get_audio_file_path(session_id)
        
        assert session_id in file_path
        assert file_path.endswith(".wav")
        assert str(file_manager.audio_dir) in file_path
    
    def test_get_file_size_mb(self, file_manager, temp_dir):
        """Test file size calculation."""
        # Create a test file with known content
        test_file = Path(temp_dir) / "test_file.txt"
        test_content = "x" * 1024  # 1KB content
        test_file.write_text(test_content)
        
        size_mb = file_manager.get_file_size_mb(str(test_file))
        expected_size_mb = 1024 / (1024 * 1024)  # Convert to MB
        
        assert abs(size_mb - expected_size_mb) < 0.001  # Allow small floating point difference
    
    def test_get_file_size_mb_nonexistent(self, file_manager):
        """Test file size calculation for non-existent file."""
        size_mb = file_manager.get_file_size_mb("nonexistent_file.txt")
        assert size_mb == 0.0
    
    def test_check_file_size_warning(self, file_manager, temp_dir):
        """Test file size warning detection."""
        # Create a file larger than the warning threshold
        large_file = Path(temp_dir) / "large_file.txt"
        large_content = "x" * (15 * 1024 * 1024)  # 15MB (larger than 10MB threshold)
        large_file.write_bytes(large_content.encode())
        
        warning = file_manager.check_file_size_warning(str(large_file))
        assert warning is not None
        assert "Large file warning" in warning
        assert "15." in warning  # Should mention the size
        
        # Test with small file
        small_file = Path(temp_dir) / "small_file.txt"
        small_file.write_text("small content")
        
        warning = file_manager.check_file_size_warning(str(small_file))
        assert warning is None
    
    def test_check_disk_space(self, file_manager):
        """Test disk space checking."""
        has_space, free_gb, warning = file_manager.check_disk_space()
        
        assert isinstance(has_space, bool)
        assert isinstance(free_gb, float)
        assert isinstance(warning, str)
        assert free_gb >= 0
    
    def test_estimate_processing_requirements(self, file_manager):
        """Test processing requirements estimation."""
        # Test with 1 hour of audio (3600 seconds)
        requirements = file_manager.estimate_processing_requirements(3600)
        
        assert "transcription_time_minutes" in requirements
        assert "notes_generation_time_minutes" in requirements
        assert "total_time_minutes" in requirements
        assert "temp_space_mb" in requirements
        
        # Check that estimates are reasonable
        assert requirements["transcription_time_minutes"] > 0
        assert requirements["notes_generation_time_minutes"] == 2.0
        assert requirements["total_time_minutes"] > requirements["transcription_time_minutes"]
        assert requirements["temp_space_mb"] > 0
    
    def test_cleanup_session_files_success(self, file_manager, sample_session, temp_dir):
        """Test successful session file cleanup."""
        # Create a test audio file
        audio_file = file_manager.audio_dir / "test_audio.wav"
        audio_file.write_text("fake audio content")
        sample_session.audio_file = str(audio_file)
        
        # Create a temp file
        temp_file = file_manager.temp_dir / "temp_file.tmp"
        temp_file.write_text("temp content")
        
        # Cleanup should succeed
        result = file_manager.cleanup_session_files(sample_session, preserve_on_error=False)
        
        assert result is True
        assert not audio_file.exists()
    
    def test_cleanup_session_files_preserve_on_error(self, file_manager, sample_session):
        """Test that files are preserved when session has errors."""
        # Create a test audio file
        audio_file = file_manager.audio_dir / "test_audio_error.wav"
        audio_file.write_text("fake audio content")
        sample_session.audio_file = str(audio_file)
        sample_session.status = "error"
        
        # Cleanup should preserve files
        result = file_manager.cleanup_session_files(sample_session, preserve_on_error=True)
        
        assert result is True
        assert audio_file.exists()  # File should still exist
        
        # Clean up manually for test cleanup
        audio_file.unlink()
    
    def test_auto_cleanup_old_files(self, file_manager):
        """Test automatic cleanup of old files."""
        # Create an old audio file
        old_audio = file_manager.audio_dir / "old_audio.wav"
        old_audio.write_text("old audio content")
        
        # Create an old temp file
        old_temp = file_manager.temp_dir / "old_temp.tmp"
        old_temp.write_text("old temp content")
        
        # Set file modification time to be older than cleanup threshold
        old_time = datetime.now() - timedelta(days=2)
        old_timestamp = old_time.timestamp()
        os.utime(old_audio, (old_timestamp, old_timestamp))
        os.utime(old_temp, (old_timestamp, old_timestamp))
        
        # Run auto cleanup
        stats = file_manager.auto_cleanup_old_files()
        
        assert stats["audio_files_removed"] >= 1
        assert stats["temp_files_removed"] >= 1
        assert stats["space_freed_mb"] > 0
        assert not old_audio.exists()
        assert not old_temp.exists()
    
    def test_get_storage_info(self, file_manager):
        """Test storage information retrieval."""
        # Create some test files
        audio_file = file_manager.audio_dir / "test_audio.wav"
        audio_file.write_text("audio content")
        
        temp_file = file_manager.temp_dir / "test_temp.tmp"
        temp_file.write_text("temp content")
        
        info = file_manager.get_storage_info()
        
        # Check all expected keys are present
        expected_keys = [
            "has_sufficient_space", "free_space_gb", "space_warning",
            "audio_files_count", "temp_files_count", "total_audio_size_mb",
            "total_temp_size_mb", "total_size_mb", "auto_cleanup_days",
            "max_file_size_mb"
        ]
        
        for key in expected_keys:
            assert key in info
        
        assert info["audio_files_count"] >= 1
        assert info["temp_files_count"] >= 1
        assert info["total_size_mb"] > 0
        
        # Clean up test files
        audio_file.unlink()
        temp_file.unlink()
    
    def test_manual_cleanup_all(self, file_manager):
        """Test manual cleanup of all files."""
        # Create test files
        audio_file = file_manager.audio_dir / "test_audio.wav"
        audio_file.write_text("audio content")
        
        temp_file = file_manager.temp_dir / "test_temp.tmp"
        temp_file.write_text("temp content")
        
        # Run manual cleanup
        stats = file_manager.manual_cleanup_all()
        
        assert stats["audio_files_removed"] >= 1
        assert stats["temp_files_removed"] >= 1
        assert stats["space_freed_mb"] > 0
        assert not audio_file.exists()
        assert not temp_file.exists()
    
    def test_validate_file_path_valid(self, file_manager):
        """Test file path validation for valid paths."""
        valid_path = file_manager.audio_dir / "valid_file.wav"
        is_valid, error_msg = file_manager.validate_file_path(str(valid_path))
        
        assert is_valid is True
        assert error_msg == ""
    
    def test_validate_file_path_outside_directory(self, file_manager):
        """Test file path validation for paths outside managed directory."""
        outside_path = "/tmp/outside_file.wav"
        is_valid, error_msg = file_manager.validate_file_path(outside_path)
        
        assert is_valid is False
        assert "outside managed directory" in error_msg
    
    def test_validate_file_path_nonexistent_parent(self, file_manager):
        """Test file path validation for non-existent parent directory."""
        invalid_path = file_manager.base_dir / "nonexistent_dir" / "file.wav"
        is_valid, error_msg = file_manager.validate_file_path(str(invalid_path))
        
        assert is_valid is False
        assert "does not exist" in error_msg


class TestFileManagerIntegration:
    """Integration tests for FileManager with MeetingSession."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def file_manager(self, temp_dir):
        """Create FileManager instance with temporary directory."""
        return FileManager(base_dir=temp_dir, max_file_size_mb=1.0)
    
    def test_full_session_lifecycle(self, file_manager):
        """Test complete session lifecycle with file management."""
        # Create session
        session = MeetingSession(status="recording")
        
        # Generate audio file path
        audio_path = file_manager.get_audio_file_path("test_session")
        session.audio_file = audio_path
        
        # Simulate creating audio file
        Path(audio_path).write_text("fake audio data")
        
        # Check file size
        size_mb = file_manager.get_file_size_mb(audio_path)
        assert size_mb > 0
        
        # Check for size warning (should trigger with 1MB limit)
        warning = file_manager.check_file_size_warning(audio_path)
        # Small file shouldn't trigger warning
        
        # Complete session
        session.status = "completed"
        
        # Cleanup
        cleanup_result = file_manager.cleanup_session_files(session)
        assert cleanup_result is True
        assert not Path(audio_path).exists()
    
    def test_error_session_preservation(self, file_manager):
        """Test that error sessions preserve files for debugging."""
        session = MeetingSession(status="error", error_message="Test error")
        
        # Create audio file
        audio_path = file_manager.get_audio_file_path("error_session")
        session.audio_file = audio_path
        Path(audio_path).write_text("error session audio")
        
        # Cleanup with preservation
        cleanup_result = file_manager.cleanup_session_files(session, preserve_on_error=True)
        assert cleanup_result is True
        assert Path(audio_path).exists()  # Should be preserved
        
        # Manual cleanup
        Path(audio_path).unlink()