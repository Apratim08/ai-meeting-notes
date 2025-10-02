"""
Notes generation service using Ollama for processing meeting transcripts.
"""

import json
import logging
import re
import time
from datetime import datetime
from typing import Optional, Dict, Any

import requests
from pydantic import ValidationError

from .models import MeetingNotes, MeetingInfo, AgendaItem, DiscussionPoint, ActionItem, Decision, TranscriptResult
from .config import AppConfig

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Custom exception for Ollama-related errors."""
    pass


class NotesGenerator:
    """
    Generates structured meeting notes from transcripts using Ollama LLM.
    
    Handles retry logic, error recovery, and structured output parsing.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.ollama_url = self.config.llm.ollama_url
        self.model_name = self.config.llm.model_name
        self.temperature = self.config.llm.temperature
        self.max_tokens = self.config.llm.max_tokens
        self.max_retries = self.config.llm.max_retries
        self.retry_delay = self.config.llm.retry_delay
        
        self._is_processing = False
        self._last_error: Optional[str] = None
    
    def check_ollama_available(self) -> bool:
        """
        Check if Ollama is running and the required model is available.
        
        Returns:
            bool: True if Ollama is available and model exists
        """
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.error(f"Ollama not responding: {response.status_code}")
                return False
            
            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            if self.model_name not in model_names:
                logger.error(f"Model {self.model_name} not found. Available models: {model_names}")
                return False
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking Ollama: {e}")
            return False
    
    def is_processing(self) -> bool:
        """Check if notes generation is currently in progress."""
        return self._is_processing
    
    def get_last_error(self) -> Optional[str]:
        """Get the last error message if any."""
        return self._last_error
    
    def generate_notes(self, transcript: str, meeting_duration: Optional[float] = None) -> MeetingNotes:
        """
        Generate structured meeting notes from a transcript.
        
        Args:
            transcript: The meeting transcript text
            meeting_duration: Duration of the meeting in minutes (optional)
            
        Returns:
            MeetingNotes: Structured meeting notes
            
        Raises:
            OllamaError: If notes generation fails after all retries
        """
        if not transcript or not transcript.strip():
            raise ValueError("Transcript cannot be empty")
        
        self._is_processing = True
        self._last_error = None
        
        try:
            return self._generate_notes_with_retry(transcript, meeting_duration)
        finally:
            self._is_processing = False
    
    def retry_generation(self, transcript: str, meeting_duration: Optional[float] = None) -> MeetingNotes:
        """
        Retry notes generation for a previously failed transcript.
        
        Args:
            transcript: The meeting transcript text
            meeting_duration: Duration of the meeting in minutes (optional)
            
        Returns:
            MeetingNotes: Structured meeting notes
        """
        logger.info("Retrying notes generation...")
        return self.generate_notes(transcript, meeting_duration)
    
    def _generate_notes_with_retry(self, transcript: str, meeting_duration: Optional[float]) -> MeetingNotes:
        """Generate notes with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Generating notes (attempt {attempt + 1}/{self.max_retries})")
                return self._generate_notes_single_attempt(transcript, meeting_duration)
                
            except Exception as e:
                last_exception = e
                self._last_error = str(e)
                logger.warning(f"Notes generation attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
        
        # All retries failed
        error_msg = f"Failed to generate notes after {self.max_retries} attempts. Last error: {last_exception}"
        logger.error(error_msg)
        raise OllamaError(error_msg)
    
    def _generate_notes_single_attempt(self, transcript: str, meeting_duration: Optional[float]) -> MeetingNotes:
        """Single attempt at generating notes."""
        # Create the prompt
        prompt = self._create_prompt(transcript, meeting_duration)
        
        # Call Ollama API
        response_text = self._call_ollama_api(prompt)
        
        # Parse the response into structured notes
        notes = self._parse_response_to_notes(response_text, meeting_duration)
        
        return notes
    
    def _create_prompt(self, transcript: str, meeting_duration: Optional[float]) -> str:
        """Create a structured prompt for the LLM."""
        duration_text = f" (Duration: {meeting_duration:.1f} minutes)" if meeting_duration else ""
        
        prompt = f"""You are an AI assistant that creates structured meeting notes from transcripts. 

Please analyze the following meeting transcript{duration_text} and create comprehensive meeting notes in JSON format.

TRANSCRIPT:
{transcript}

Please provide the output as a valid JSON object with the following structure:
{{
    "meeting_info": {{
        "title": "Brief descriptive title for the meeting (or null if unclear)",
        "platform": "Meeting platform if mentioned (Zoom/Teams/Meet/etc, or null)",
        "duration": {meeting_duration or "null"}
    }},
    "participants": ["List of participant names mentioned in the transcript"],
    "summary": "2-3 sentence summary of the meeting's main purpose and outcomes",
    "agenda_items": [
        {{
            "title": "Agenda item title",
            "description": "Description of what was discussed",
            "time_discussed": "Approximate time or null"
        }}
    ],
    "discussion_points": [
        {{
            "topic": "Main topic discussed",
            "key_points": ["List of key points made about this topic"],
            "timestamp": "Approximate time or null"
        }}
    ],
    "decisions": [
        {{
            "decision": "Decision that was made",
            "rationale": "Why this decision was made (or null)",
            "timestamp": "When decision was made (or null)"
        }}
    ],
    "action_items": [
        {{
            "task": "Specific task to be completed",
            "assignee": "Person assigned (or null if unclear)",
            "due_date": "Due date if mentioned (or null)",
            "priority": "high/medium/low based on urgency"
        }}
    ]
}}

Guidelines:
- Extract only information that is clearly present in the transcript
- Use null for missing information rather than making assumptions
- Focus on concrete decisions and action items
- Group related discussion points together
- Identify participants by names mentioned in the transcript
- Keep descriptions concise but informative
- Ensure the JSON is valid and properly formatted

Respond with ONLY the JSON object, no additional text or formatting."""

        return prompt
    
    def _call_ollama_api(self, prompt: str) -> str:
        """Call the Ollama API to generate notes with enhanced error handling."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }
        
        try:
            logger.debug(f"Calling Ollama API with model {self.model_name}")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=300  # 5 minute timeout for generation
            )
            
            if response.status_code == 404:
                raise OllamaError(
                    f"Model '{self.model_name}' not found. Please ensure the model is installed with: "
                    f"ollama pull {self.model_name}"
                )
            elif response.status_code == 503:
                raise OllamaError(
                    "Ollama service unavailable. Please check that Ollama is running."
                )
            elif response.status_code != 200:
                error_detail = ""
                try:
                    error_data = response.json()
                    error_detail = error_data.get("error", response.text)
                except:
                    error_detail = response.text
                
                raise OllamaError(f"Ollama API error: {response.status_code} - {error_detail}")
            
            result = response.json()
            
            if "response" not in result:
                raise OllamaError(f"Invalid Ollama response format: {result}")
            
            response_text = result["response"]
            
            # Validate response is not empty
            if not response_text or not response_text.strip():
                raise OllamaError("Ollama returned empty response")
            
            return response_text
            
        except requests.exceptions.Timeout:
            raise OllamaError(
                "Ollama API request timed out. The transcript may be too long or the model is overloaded. "
                "Try again or use a smaller model."
            )
        except requests.exceptions.ConnectionError as e:
            raise OllamaError(
                f"Cannot connect to Ollama service at {self.ollama_url}. "
                f"Please ensure Ollama is running and accessible. Error: {e}"
            )
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Network error connecting to Ollama: {e}")
        except json.JSONDecodeError as e:
            raise OllamaError(f"Invalid JSON response from Ollama: {e}")
    
    def _parse_response_to_notes(self, response_text: str, meeting_duration: Optional[float]) -> MeetingNotes:
        """Parse the LLM response into structured MeetingNotes."""
        try:
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in response")
            
            json_text = json_match.group(0)
            data = json.loads(json_text)
            
            # Create MeetingInfo
            meeting_info_data = data.get("meeting_info", {})
            meeting_info = MeetingInfo(
                title=meeting_info_data.get("title"),
                date=datetime.now(),
                duration=meeting_duration,
                platform=meeting_info_data.get("platform")
            )
            
            # Parse agenda items
            agenda_items = []
            for item_data in data.get("agenda_items", []):
                agenda_items.append(AgendaItem(
                    title=item_data.get("title", ""),
                    description=item_data.get("description", ""),
                    time_discussed=item_data.get("time_discussed")
                ))
            
            # Parse discussion points
            discussion_points = []
            for point_data in data.get("discussion_points", []):
                discussion_points.append(DiscussionPoint(
                    topic=point_data.get("topic", ""),
                    key_points=point_data.get("key_points", []),
                    timestamp=point_data.get("timestamp")
                ))
            
            # Parse action items
            action_items = []
            for action_data in data.get("action_items", []):
                action_items.append(ActionItem(
                    task=action_data.get("task", ""),
                    assignee=action_data.get("assignee"),
                    due_date=action_data.get("due_date"),
                    priority=action_data.get("priority", "medium")
                ))
            
            # Parse decisions
            decisions = []
            for decision_data in data.get("decisions", []):
                decisions.append(Decision(
                    decision=decision_data.get("decision", ""),
                    rationale=decision_data.get("rationale"),
                    timestamp=decision_data.get("timestamp")
                ))
            
            # Create the complete MeetingNotes object
            notes = MeetingNotes(
                meeting_info=meeting_info,
                agenda_items=agenda_items,
                discussion_points=discussion_points,
                action_items=action_items,
                decisions=decisions,
                participants=data.get("participants", []),
                summary=data.get("summary", "")
            )
            
            logger.info("Successfully generated structured meeting notes")
            return notes
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text}")
            raise OllamaError(f"Invalid JSON in LLM response: {e}")
        except ValidationError as e:
            logger.error(f"Failed to validate meeting notes structure: {e}")
            raise OllamaError(f"Invalid meeting notes structure: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            raise OllamaError(f"Failed to parse meeting notes: {e}")