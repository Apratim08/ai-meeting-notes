"""
End-to-end integration test with sample meeting recordings.

This test demonstrates the complete pipeline with realistic sample data.
"""

import asyncio
import tempfile
import wave
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np

from ai_meeting_notes.processing_pipeline import ProcessingPipeline
from ai_meeting_notes.models import TranscriptResult, TranscriptSegment, MeetingNotes, MeetingInfo, ActionItem, DiscussionPoint
from ai_meeting_notes.audio_recorder import AudioRecorder
from ai_meeting_notes.transcription import BatchTranscriber
from ai_meeting_notes.notes_generator import NotesGenerator
from ai_meeting_notes.file_manager import FileManager


class TestSampleMeetingProcessing:
    """Test processing pipeline with sample meeting data."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def create_sample_audio_file(self, filepath: str, duration: float = 30.0) -> str:
        """
        Create a sample audio file that simulates a meeting recording.
        
        This creates a WAV file with multiple tones to simulate speech patterns.
        """
        sample_rate = 16000
        frames = int(sample_rate * duration)
        
        # Create more complex audio that simulates speech patterns
        t = np.linspace(0, duration, frames)
        
        # Multiple frequency components to simulate speech
        audio_data = (
            0.3 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
            0.2 * np.sin(2 * np.pi * 800 * t) +  # Mid frequency
            0.1 * np.sin(2 * np.pi * 1600 * t)   # High frequency
        )
        
        # Add some amplitude modulation to simulate speech patterns
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz modulation
        audio_data = audio_data * modulation
        
        # Add some quiet periods to simulate pauses
        for i in range(0, len(audio_data), sample_rate * 5):  # Every 5 seconds
            quiet_start = i
            quiet_end = min(i + sample_rate, len(audio_data))  # 1 second quiet
            audio_data[quiet_start:quiet_end] *= 0.1
        
        # Convert to int16
        audio_int16 = (audio_data * 16000).astype(np.int16)
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Write WAV file
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return filepath
    
    def create_sample_transcript(self, audio_file_path: str) -> TranscriptResult:
        """Create a realistic sample transcript for testing."""
        segments = [
            TranscriptSegment(
                text="Good morning everyone, thank you for joining today's project review meeting.",
                start_time=0.0,
                end_time=4.5,
                confidence=0.92
            ),
            TranscriptSegment(
                text="Let's start by reviewing the progress on the user authentication feature.",
                start_time=5.0,
                end_time=9.2,
                confidence=0.88
            ),
            TranscriptSegment(
                text="Alice, can you give us an update on the backend implementation?",
                start_time=10.0,
                end_time=13.8,
                confidence=0.95
            ),
            TranscriptSegment(
                text="Sure, I've completed the JWT token implementation and the password hashing.",
                start_time=14.5,
                end_time=19.2,
                confidence=0.91
            ),
            TranscriptSegment(
                text="The API endpoints are working, but I need to add rate limiting by Friday.",
                start_time=19.8,
                end_time=24.1,
                confidence=0.89
            ),
            TranscriptSegment(
                text="Great work Alice. Bob, how's the frontend integration coming along?",
                start_time=25.0,
                end_time=29.3,
                confidence=0.93
            )
        ]
        
        full_text = " ".join(segment.text for segment in segments)
        
        return TranscriptResult(
            segments=segments,
            full_text=full_text,
            duration=30.0,
            language="en",
            processing_time=3.2,
            model_name="base",
            audio_file_path=audio_file_path
        )
    
    def create_sample_meeting_notes(self) -> MeetingNotes:
        """Create realistic sample meeting notes for testing."""
        return MeetingNotes(
            meeting_info=MeetingInfo(
                title="Project Review Meeting - User Authentication",
                duration=0.5  # 30 seconds = 0.5 minutes
            ),
            summary="Team review of user authentication feature progress. Backend JWT implementation completed, frontend integration in progress.",
            participants=["Alice", "Bob"],
            agenda_items=[],
            discussion_points=[
                DiscussionPoint(
                    topic="Backend Authentication Implementation",
                    key_points=[
                        "JWT token implementation completed",
                        "Password hashing implemented",
                        "API endpoints working"
                    ],
                    timestamp="0:14"
                ),
                DiscussionPoint(
                    topic="Frontend Integration Status",
                    key_points=[
                        "Integration with backend APIs in progress"
                    ],
                    timestamp="0:25"
                )
            ],
            action_items=[
                ActionItem(
                    task="Add rate limiting to authentication API endpoints",
                    assignee="Alice",
                    due_date="Friday",
                    priority="medium"
                )
            ],
            decisions=[]
        )
    
    @pytest.fixture
    def mock_services_with_sample_data(self, temp_dir):
        """Create mock services that return sample meeting data."""
        # Mock audio recorder
        audio_recorder = Mock(spec=AudioRecorder)
        audio_recorder.check_blackhole_available.return_value = True
        audio_recorder.is_recording.return_value = False
        audio_recorder.get_recording_duration.return_value = 30.0
        audio_recorder.start_recording = Mock()
        
        # Create sample audio file
        sample_audio_path = str(temp_dir / "sample_meeting.wav")
        self.create_sample_audio_file(sample_audio_path, duration=30.0)
        audio_recorder.stop_recording.return_value = sample_audio_path
        
        # Mock transcriber with sample transcript
        transcriber = Mock(spec=BatchTranscriber)
        transcriber.estimate_processing_time.return_value = 3.0
        transcriber.get_progress.return_value = 1.0
        transcriber.transcribe_file.return_value = self.create_sample_transcript(sample_audio_path)
        
        # Mock notes generator with sample notes
        notes_generator = Mock(spec=NotesGenerator)
        notes_generator.check_ollama_available.return_value = True
        notes_generator.generate_notes.return_value = self.create_sample_meeting_notes()
        
        # Mock file manager
        file_manager = Mock(spec=FileManager)
        file_manager.check_disk_space.return_value = (True, 10.0, "")
        file_manager.get_audio_file_path.return_value = sample_audio_path
        file_manager.cleanup_session_files.return_value = True
        
        return {
            'audio_recorder': audio_recorder,
            'transcriber': transcriber,
            'notes_generator': notes_generator,
            'file_manager': file_manager,
            'sample_audio_path': sample_audio_path
        }
    
    @pytest.fixture
    def pipeline_with_sample_data(self, mock_services_with_sample_data):
        """Create a processing pipeline with sample data services."""
        services = mock_services_with_sample_data
        
        return ProcessingPipeline(
            audio_recorder=services['audio_recorder'],
            transcriber=services['transcriber'],
            notes_generator=services['notes_generator'],
            file_manager=services['file_manager'],
            progress_callback=None
        )
    
    @pytest.mark.asyncio
    async def test_complete_sample_meeting_workflow(self, pipeline_with_sample_data, mock_services_with_sample_data):
        """Test complete workflow with sample meeting data."""
        pipeline = pipeline_with_sample_data
        services = mock_services_with_sample_data
        
        # 1. Start recording
        session_id = "sample_meeting_123"
        session = await pipeline.start_recording(session_id)
        
        assert session is not None
        assert session.status == "recording"
        
        # Verify recording started
        services['audio_recorder'].start_recording.assert_called_once()
        
        # 2. Simulate recording in progress
        services['audio_recorder'].is_recording.return_value = True
        
        status = pipeline.get_processing_status()
        assert status["has_session"]
        assert status["status"] == "recording"
        
        # 3. Stop recording and start processing
        session = await pipeline.stop_recording_and_process()
        
        assert session.audio_file == services['sample_audio_path']
        
        # Verify recording stopped
        services['audio_recorder'].stop_recording.assert_called_once()
        
        # 4. Wait for processing to complete
        max_wait = 10  # seconds
        wait_time = 0
        while pipeline.is_processing() and wait_time < max_wait:
            await asyncio.sleep(0.1)
            wait_time += 0.1
        
        # 5. Verify final results
        final_session = pipeline.get_current_session()
        assert final_session is not None
        assert final_session.status == "completed"
        assert final_session.processing_progress == 100.0
        
        # Verify transcript
        transcript = final_session.transcript
        assert transcript is not None
        assert "Good morning everyone" in transcript.full_text
        assert "Alice" in transcript.full_text
        assert "Bob" in transcript.full_text
        assert len(transcript.segments) == 6
        assert transcript.language == "en"
        assert transcript.duration == 30.0
        
        # Verify notes
        notes = final_session.notes
        assert notes is not None
        assert "Project Review Meeting" in notes.meeting_info.title
        assert "Alice" in notes.participants
        assert "Bob" in notes.participants
        assert len(notes.discussion_points) == 2
        assert len(notes.action_items) == 1
        
        # Verify action item details
        action_item = notes.action_items[0]
        assert "rate limiting" in action_item.task
        assert action_item.assignee == "Alice"
        assert action_item.due_date == "Friday"
        
        # Verify discussion points
        backend_discussion = next(
            (dp for dp in notes.discussion_points if "Backend" in dp.topic), 
            None
        )
        assert backend_discussion is not None
        assert "JWT token" in str(backend_discussion.key_points)
        
        # 6. Verify processing steps were called
        services['transcriber'].transcribe_file.assert_called_once_with(services['sample_audio_path'])
        services['notes_generator'].generate_notes.assert_called_once()
        
        # 7. Test formatted output
        formatted_notes = notes.to_formatted_text()
        assert "# Meeting Notes" in formatted_notes
        assert "## Participants" in formatted_notes
        assert "## Action Items" in formatted_notes
        assert "Alice" in formatted_notes
        assert "rate limiting" in formatted_notes
    
    @pytest.mark.asyncio
    async def test_sample_meeting_with_transcription_failure(self, pipeline_with_sample_data, mock_services_with_sample_data):
        """Test sample meeting workflow with transcription failure and recovery."""
        pipeline = pipeline_with_sample_data
        services = mock_services_with_sample_data
        
        # Start and stop recording
        await pipeline.start_recording("test_session")
        services['audio_recorder'].is_recording.return_value = True
        
        # Make transcription fail initially
        services['transcriber'].transcribe_file.side_effect = Exception("Transcription service unavailable")
        
        session = await pipeline.stop_recording_and_process()
        
        # Wait for processing to complete (should fail)
        max_wait = 5
        wait_time = 0
        while pipeline.is_processing() and wait_time < max_wait:
            await asyncio.sleep(0.1)
            wait_time += 0.1
        
        # Verify error state
        final_session = pipeline.get_current_session()
        assert final_session.status == "error"
        assert "Transcription service unavailable" in final_session.error_message
        
        # Verify audio file is preserved
        assert final_session.audio_file == services['sample_audio_path']
        assert Path(services['sample_audio_path']).exists()
        
        # Now retry with working transcription
        services['transcriber'].transcribe_file.side_effect = None
        services['transcriber'].transcribe_file.return_value = self.create_sample_transcript(services['sample_audio_path'])
        
        await pipeline.retry_transcription(final_session)
        
        # Wait for retry to complete
        max_wait = 5
        wait_time = 0
        while pipeline.is_processing() and wait_time < max_wait:
            await asyncio.sleep(0.1)
            wait_time += 0.1
        
        # Verify successful completion after retry
        assert final_session.status == "completed"
        assert final_session.transcript is not None
        assert final_session.notes is not None
    
    @pytest.mark.asyncio
    async def test_sample_meeting_with_notes_failure(self, pipeline_with_sample_data, mock_services_with_sample_data):
        """Test sample meeting workflow with notes generation failure and recovery."""
        pipeline = pipeline_with_sample_data
        services = mock_services_with_sample_data
        
        # Start and stop recording
        await pipeline.start_recording("test_session")
        services['audio_recorder'].is_recording.return_value = True
        
        # Make notes generation fail
        from ai_meeting_notes.notes_generator import OllamaError
        services['notes_generator'].generate_notes.side_effect = OllamaError("Ollama service unavailable")
        
        session = await pipeline.stop_recording_and_process()
        
        # Wait for processing to complete (should fail at notes generation)
        max_wait = 5
        wait_time = 0
        while pipeline.is_processing() and wait_time < max_wait:
            await asyncio.sleep(0.1)
            wait_time += 0.1
        
        # Verify error state but transcript preserved
        final_session = pipeline.get_current_session()
        assert final_session.status == "error"
        assert "Ollama service unavailable" in final_session.error_message
        assert final_session.transcript is not None  # Transcript should be preserved
        
        # Now retry notes generation with working service
        services['notes_generator'].generate_notes.side_effect = None
        services['notes_generator'].generate_notes.return_value = self.create_sample_meeting_notes()
        
        await pipeline.retry_notes_generation(final_session)
        
        # Wait for retry to complete
        max_wait = 5
        wait_time = 0
        while pipeline.is_processing() and wait_time < max_wait:
            await asyncio.sleep(0.1)
            wait_time += 0.1
        
        # Verify successful completion after retry
        assert final_session.status == "completed"
        assert final_session.notes is not None
    
    def test_sample_audio_file_properties(self, temp_dir):
        """Test that sample audio files have correct properties."""
        audio_file = str(temp_dir / "test_sample.wav")
        self.create_sample_audio_file(audio_file, duration=15.0)
        
        # Verify file exists and has correct properties
        assert Path(audio_file).exists()
        
        with wave.open(audio_file, 'rb') as wav_file:
            assert wav_file.getnchannels() == 1  # Mono
            assert wav_file.getsampwidth() == 2  # 16-bit
            assert wav_file.getframerate() == 16000  # 16kHz
            
            # Check duration
            frames = wav_file.getnframes()
            duration = frames / wav_file.getframerate()
            assert abs(duration - 15.0) < 0.1  # Within 0.1 seconds
    
    def test_sample_transcript_properties(self, temp_dir):
        """Test that sample transcript has realistic properties."""
        audio_file = str(temp_dir / "test.wav")
        transcript = self.create_sample_transcript(audio_file)
        
        # Verify transcript structure
        assert len(transcript.segments) == 6
        assert transcript.duration == 30.0
        assert transcript.language == "en"
        assert transcript.processing_time > 0
        
        # Verify content
        assert "Good morning everyone" in transcript.full_text
        assert "Alice" in transcript.full_text
        assert "Bob" in transcript.full_text
        assert "authentication" in transcript.full_text
        
        # Verify segment properties
        for segment in transcript.segments:
            assert segment.confidence > 0.8  # High confidence
            assert segment.start_time < segment.end_time
            assert len(segment.text.strip()) > 0
        
        # Verify average confidence
        assert transcript.average_confidence > 0.9
    
    def test_sample_meeting_notes_properties(self):
        """Test that sample meeting notes have realistic structure."""
        notes = self.create_sample_meeting_notes()
        
        # Verify basic structure
        assert notes.meeting_info.title is not None
        assert len(notes.participants) == 2
        assert "Alice" in notes.participants
        assert "Bob" in notes.participants
        
        # Verify content
        assert len(notes.discussion_points) == 2
        assert len(notes.action_items) == 1
        
        # Verify action item
        action_item = notes.action_items[0]
        assert action_item.assignee == "Alice"
        assert action_item.due_date == "Friday"
        assert "rate limiting" in action_item.task
        
        # Verify discussion points have meaningful content
        for dp in notes.discussion_points:
            assert len(dp.key_points) > 0
            assert len(dp.topic) > 0
        
        # Test formatted output
        formatted = notes.to_formatted_text()
        assert "# Meeting Notes" in formatted
        assert "Alice" in formatted
        assert "rate limiting" in formatted


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])