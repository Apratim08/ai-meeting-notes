"""
Unit tests for the NotesGenerator class.
"""

import json
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from ai_meeting_notes.notes_generator import NotesGenerator, OllamaError
from ai_meeting_notes.models import MeetingNotes, MeetingInfo, AgendaItem, ActionItem
from ai_meeting_notes.config import AppConfig, LLMConfig


class TestNotesGenerator:
    """Test cases for NotesGenerator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AppConfig(
            llm=LLMConfig(
                model_name="test-model",
                temperature=0.3,
                max_tokens=1000,
                max_retries=2,
                retry_delay=0.1,  # Fast retry for tests
                ollama_url="http://localhost:11434"
            )
        )
    
    @pytest.fixture
    def notes_generator(self, config):
        """Create NotesGenerator instance for testing."""
        return NotesGenerator(config)
    
    @pytest.fixture
    def sample_transcript(self):
        """Sample meeting transcript for testing."""
        return """
        John: Good morning everyone, let's start our weekly team meeting.
        Sarah: Hi John, I have an update on the project timeline.
        John: Great, please go ahead Sarah.
        Sarah: We've completed the design phase and are ready to move to development. 
        I think we should assign the frontend work to Mike and backend to Lisa.
        Mike: Sounds good, I can start on the UI components next week.
        Lisa: I'll handle the API development. When do we need this completed?
        John: Let's target end of month. Sarah, can you create tickets for this?
        Sarah: Absolutely, I'll have them ready by Friday.
        John: Perfect. Any other items? No? Great, meeting adjourned.
        """
    
    @pytest.fixture
    def sample_ollama_response(self):
        """Sample Ollama API response."""
        return {
            "response": json.dumps({
                "meeting_info": {
                    "title": "Weekly Team Meeting",
                    "platform": None,
                    "duration": 15.0
                },
                "participants": ["John", "Sarah", "Mike", "Lisa"],
                "summary": "Team discussed project timeline and assigned development tasks for the upcoming sprint.",
                "agenda_items": [
                    {
                        "title": "Project Timeline Update",
                        "description": "Review of design phase completion and next steps",
                        "time_discussed": "Beginning of meeting"
                    }
                ],
                "discussion_points": [
                    {
                        "topic": "Development Task Assignment",
                        "key_points": [
                            "Frontend work assigned to Mike",
                            "Backend API development assigned to Lisa",
                            "Target completion by end of month"
                        ],
                        "timestamp": "Mid-meeting"
                    }
                ],
                "decisions": [
                    {
                        "decision": "Proceed with development phase",
                        "rationale": "Design phase completed successfully",
                        "timestamp": None
                    }
                ],
                "action_items": [
                    {
                        "task": "Create development tickets",
                        "assignee": "Sarah",
                        "due_date": "Friday",
                        "priority": "high"
                    },
                    {
                        "task": "Start UI components development",
                        "assignee": "Mike",
                        "due_date": "Next week",
                        "priority": "medium"
                    }
                ]
            })
        }
    
    def test_init_with_default_config(self):
        """Test NotesGenerator initialization with default config."""
        generator = NotesGenerator()
        assert generator.model_name == "llama3.1:8b"
        assert generator.temperature == 0.3
        assert generator.ollama_url == "http://localhost:11434"
        assert not generator.is_processing()
    
    def test_init_with_custom_config(self, config):
        """Test NotesGenerator initialization with custom config."""
        generator = NotesGenerator(config)
        assert generator.model_name == "test-model"
        assert generator.temperature == 0.3
        assert generator.max_retries == 2
    
    @patch('requests.get')
    def test_check_ollama_available_success(self, mock_get, notes_generator):
        """Test successful Ollama availability check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "test-model"},
                {"name": "other-model"}
            ]
        }
        mock_get.return_value = mock_response
        
        assert notes_generator.check_ollama_available() is True
        mock_get.assert_called_once_with(
            "http://localhost:11434/api/tags", 
            timeout=5
        )
    
    @patch('requests.get')
    def test_check_ollama_available_model_not_found(self, mock_get, notes_generator):
        """Test Ollama availability check when model is not found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "other-model"}]
        }
        mock_get.return_value = mock_response
        
        assert notes_generator.check_ollama_available() is False
    
    @patch('requests.get')
    def test_check_ollama_available_connection_error(self, mock_get, notes_generator):
        """Test Ollama availability check with connection error."""
        mock_get.side_effect = ConnectionError("Connection failed")
        
        assert notes_generator.check_ollama_available() is False
    
    @patch('ai_meeting_notes.notes_generator.NotesGenerator._call_ollama_api')
    def test_generate_notes_success(self, mock_call_api, notes_generator, sample_transcript, sample_ollama_response):
        """Test successful notes generation."""
        mock_call_api.return_value = sample_ollama_response["response"]
        
        notes = notes_generator.generate_notes(sample_transcript, 15.0)
        
        assert isinstance(notes, MeetingNotes)
        assert notes.meeting_info.title == "Weekly Team Meeting"
        assert notes.meeting_info.duration == 15.0
        assert len(notes.participants) == 4
        assert "John" in notes.participants
        assert len(notes.action_items) == 2
        assert notes.action_items[0].task == "Create development tickets"
        assert notes.action_items[0].assignee == "Sarah"
        assert not notes_generator.is_processing()
    
    def test_generate_notes_empty_transcript(self, notes_generator):
        """Test notes generation with empty transcript."""
        with pytest.raises(ValueError, match="Transcript cannot be empty"):
            notes_generator.generate_notes("")
    
    @patch('ai_meeting_notes.notes_generator.NotesGenerator._call_ollama_api')
    def test_generate_notes_with_retry_success(self, mock_call_api, notes_generator, sample_transcript, sample_ollama_response):
        """Test notes generation with retry on first failure."""
        # First call fails, second succeeds
        mock_call_api.side_effect = [
            OllamaError("API Error"),
            sample_ollama_response["response"]
        ]
        
        notes = notes_generator.generate_notes(sample_transcript)
        
        assert isinstance(notes, MeetingNotes)
        assert mock_call_api.call_count == 2
    
    @patch('ai_meeting_notes.notes_generator.NotesGenerator._call_ollama_api')
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_generate_notes_max_retries_exceeded(self, mock_sleep, mock_call_api, notes_generator, sample_transcript):
        """Test notes generation when max retries are exceeded."""
        mock_call_api.side_effect = OllamaError("Persistent error")
        
        with pytest.raises(OllamaError, match="Failed to generate notes after 2 attempts"):
            notes_generator.generate_notes(sample_transcript)
        
        assert mock_call_api.call_count == 2
        assert mock_sleep.call_count == 1  # Sleep called between retries
    
    @patch('requests.post')
    def test_call_ollama_api_success(self, mock_post, notes_generator):
        """Test successful Ollama API call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Generated notes"}
        mock_post.return_value = mock_response
        
        result = notes_generator._call_ollama_api("Test prompt")
        
        assert result == "Generated notes"
        mock_post.assert_called_once()
        
        # Check the request payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['model'] == "test-model"
        assert payload['prompt'] == "Test prompt"
        assert payload['options']['temperature'] == 0.3
    
    @patch('requests.post')
    def test_call_ollama_api_http_error(self, mock_post, notes_generator):
        """Test Ollama API call with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        with pytest.raises(OllamaError, match="Ollama API error: 500"):
            notes_generator._call_ollama_api("Test prompt")
    
    @patch('requests.post')
    def test_call_ollama_api_timeout(self, mock_post, notes_generator):
        """Test Ollama API call timeout."""
        mock_post.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(OllamaError, match="Ollama API request timed out"):
            notes_generator._call_ollama_api("Test prompt")
    
    def test_parse_response_to_notes_success(self, notes_generator, sample_ollama_response):
        """Test successful response parsing."""
        response_text = sample_ollama_response["response"]
        
        notes = notes_generator._parse_response_to_notes(response_text, 15.0)
        
        assert isinstance(notes, MeetingNotes)
        assert notes.meeting_info.title == "Weekly Team Meeting"
        assert len(notes.participants) == 4
        assert len(notes.action_items) == 2
    
    def test_parse_response_to_notes_invalid_json(self, notes_generator):
        """Test response parsing with invalid JSON."""
        invalid_response = "This is not JSON"
        
        with pytest.raises(OllamaError, match="No JSON object found in response"):
            notes_generator._parse_response_to_notes(invalid_response, 15.0)
    
    def test_parse_response_to_notes_malformed_json(self, notes_generator):
        """Test response parsing with malformed JSON."""
        malformed_json = '{"meeting_info": {'
        
        with pytest.raises(OllamaError, match="Failed to parse meeting notes"):
            notes_generator._parse_response_to_notes(malformed_json, 15.0)
    
    def test_create_prompt(self, notes_generator):
        """Test prompt creation."""
        transcript = "Sample meeting transcript"
        duration = 30.0
        
        prompt = notes_generator._create_prompt(transcript, duration)
        
        assert "Sample meeting transcript" in prompt
        assert "Duration: 30.0 minutes" in prompt
        assert "JSON" in prompt
        assert "meeting_info" in prompt
        assert "participants" in prompt
    
    def test_create_prompt_no_duration(self, notes_generator):
        """Test prompt creation without duration."""
        transcript = "Sample meeting transcript"
        
        prompt = notes_generator._create_prompt(transcript, None)
        
        assert "Sample meeting transcript" in prompt
        assert "Duration:" not in prompt
        assert '"duration": null' in prompt
    
    def test_retry_generation(self, notes_generator, sample_transcript):
        """Test retry generation method."""
        with patch.object(notes_generator, 'generate_notes') as mock_generate:
            mock_notes = MeetingNotes(
                meeting_info=MeetingInfo(),
                participants=[],
                summary="Test summary"
            )
            mock_generate.return_value = mock_notes
            
            result = notes_generator.retry_generation(sample_transcript, 15.0)
            
            assert result == mock_notes
            mock_generate.assert_called_once_with(sample_transcript, 15.0)
    
    def test_get_last_error(self, notes_generator):
        """Test getting last error message."""
        assert notes_generator.get_last_error() is None
        
        # Simulate an error
        notes_generator._last_error = "Test error"
        assert notes_generator.get_last_error() == "Test error"
    
    def test_processing_state_management(self, notes_generator, sample_transcript):
        """Test processing state is managed correctly."""
        assert not notes_generator.is_processing()
        
        with patch.object(notes_generator, '_generate_notes_with_retry') as mock_generate:
            mock_generate.return_value = MeetingNotes(
                meeting_info=MeetingInfo(),
                participants=[],
                summary="Test"
            )
            
            notes_generator.generate_notes(sample_transcript)
            
            # Processing should be False after completion
            assert not notes_generator.is_processing()


class TestMeetingNotesFormatting:
    """Test cases for MeetingNotes formatting methods."""
    
    @pytest.fixture
    def sample_notes(self):
        """Create sample meeting notes for testing."""
        return MeetingNotes(
            meeting_info=MeetingInfo(
                title="Test Meeting",
                date=datetime(2024, 1, 15, 14, 30),
                duration=45.0,
                platform="Zoom"
            ),
            participants=["Alice", "Bob", "Charlie"],
            summary="Discussed project progress and next steps.",
            agenda_items=[
                AgendaItem(
                    title="Project Update",
                    description="Review current status",
                    time_discussed="14:35"
                )
            ],
            action_items=[
                ActionItem(
                    task="Complete documentation",
                    assignee="Alice",
                    due_date="2024-01-20",
                    priority="high"
                )
            ]
        )
    
    def test_to_formatted_text(self, sample_notes):
        """Test formatted text output."""
        formatted = sample_notes.to_formatted_text()
        
        assert "# Meeting Notes" in formatted
        assert "Test Meeting" in formatted
        assert "2024-01-15 14:30" in formatted
        assert "45.0 minutes" in formatted
        assert "Alice" in formatted
        assert "Bob" in formatted
        assert "Charlie" in formatted
        assert "Project Update" in formatted
        assert "Complete documentation" in formatted
        assert "[HIGH]" in formatted
    
    def test_to_formatted_text_minimal(self):
        """Test formatted text with minimal data."""
        minimal_notes = MeetingNotes(
            meeting_info=MeetingInfo(),
            participants=[],
            summary=""
        )
        
        formatted = minimal_notes.to_formatted_text()
        
        assert "# Meeting Notes" in formatted
        assert "Participants" not in formatted
        assert "Summary" not in formatted


# Integration test with mock Ollama server
@pytest.mark.integration
class TestNotesGeneratorIntegration:
    """Integration tests for NotesGenerator with mocked Ollama."""
    
    @patch('requests.get')
    @patch('requests.post')
    def test_full_workflow_success(self, mock_post, mock_get):
        """Test complete workflow from availability check to notes generation."""
        # Mock availability check
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "models": [{"name": "llama3.1:8b"}]
        }
        mock_get.return_value = mock_get_response
        
        # Mock notes generation
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            "response": json.dumps({
                "meeting_info": {"title": "Integration Test", "platform": None, "duration": None},
                "participants": ["Tester"],
                "summary": "Integration test meeting",
                "agenda_items": [],
                "discussion_points": [],
                "decisions": [],
                "action_items": []
            })
        }
        mock_post.return_value = mock_post_response
        
        generator = NotesGenerator()
        
        # Check availability
        assert generator.check_ollama_available() is True
        
        # Generate notes
        transcript = "Tester: This is a test meeting for integration testing."
        notes = generator.generate_notes(transcript)
        
        assert isinstance(notes, MeetingNotes)
        assert notes.meeting_info.title == "Integration Test"
        assert "Tester" in notes.participants
        assert notes.summary == "Integration test meeting"