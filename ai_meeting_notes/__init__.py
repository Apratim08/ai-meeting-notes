"""AI Meeting Notes - Audio capture and transcription with AI-generated notes."""

__version__ = "0.1.0"

from .file_manager import FileManager
from .models import (
    MeetingSession, 
    MeetingNotes, 
    TranscriptResult, 
    TranscriptSegment,
    MeetingInfo,
    AgendaItem,
    DiscussionPoint,
    ActionItem,
    Decision
)

__all__ = [
    "FileManager",
    "MeetingSession",
    "MeetingNotes", 
    "TranscriptResult",
    "TranscriptSegment",
    "MeetingInfo",
    "AgendaItem",
    "DiscussionPoint", 
    "ActionItem",
    "Decision"
]