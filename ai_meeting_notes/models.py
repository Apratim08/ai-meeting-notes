from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """Individual segment of transcribed text with metadata."""
    text: str
    start_time: float
    end_time: float
    confidence: float = Field(ge=0.0, le=1.0)
    speaker_id: Optional[str] = None
    speaker: Optional[str] = None  # "You", "Others", or specific speaker name
    is_uncertain: bool = Field(default=False)  # True if confidence < 0.7


class TranscriptResult(BaseModel):
    """Complete transcription result with metadata."""
    segments: List[TranscriptSegment]
    full_text: str
    duration: float
    language: str
    processing_time: float
    model_name: str
    created_at: datetime = Field(default_factory=datetime.now)
    audio_file_path: str
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all segments."""
        if not self.segments:
            return 0.0
        return sum(seg.confidence for seg in self.segments) / len(self.segments)
    
    @property
    def uncertain_segments_count(self) -> int:
        """Count segments with low confidence."""
        return sum(1 for seg in self.segments if seg.is_uncertain)


class MeetingInfo(BaseModel):
    """Basic meeting information."""
    title: Optional[str] = None
    date: datetime = Field(default_factory=datetime.now)
    duration: Optional[float] = None  # in minutes
    platform: Optional[str] = None  # e.g., "Zoom", "Teams", "Google Meet"


class AgendaItem(BaseModel):
    """Individual agenda item discussed in the meeting."""
    title: str
    description: str
    time_discussed: Optional[str] = None
    speaker: Optional[str] = None


class DiscussionPoint(BaseModel):
    """Key discussion point from the meeting."""
    topic: str
    key_points: List[str]
    timestamp: Optional[str] = None
    speakers: Optional[List[str]] = None


class ActionItem(BaseModel):
    """Action item identified during the meeting."""
    task: str
    assignee: Optional[str] = None
    due_date: Optional[str] = None
    priority: str = "medium"  # low, medium, high


class Decision(BaseModel):
    """Decision made during the meeting."""
    decision: str
    rationale: Optional[str] = None
    timestamp: Optional[str] = None
    decided_by: Optional[str] = None


class MeetingNotes(BaseModel):
    """Complete structured meeting notes."""
    meeting_info: MeetingInfo
    agenda_items: List[AgendaItem] = Field(default_factory=list)
    discussion_points: List[DiscussionPoint] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    participants: List[str] = Field(default_factory=list)
    summary: str = ""
    
    def to_formatted_text(self) -> str:
        """Convert meeting notes to formatted text for display."""
        lines = []
        
        # Meeting Info
        lines.append(f"# Meeting Notes")
        if self.meeting_info.title:
            lines.append(f"**Title:** {self.meeting_info.title}")
        lines.append(f"**Date:** {self.meeting_info.date.strftime('%Y-%m-%d %H:%M')}")
        if self.meeting_info.duration:
            lines.append(f"**Duration:** {self.meeting_info.duration:.1f} minutes")
        lines.append("")
        
        # Participants
        if self.participants:
            lines.append("## Participants")
            for participant in self.participants:
                lines.append(f"- {participant}")
            lines.append("")
        
        # Summary
        if self.summary:
            lines.append("## Summary")
            lines.append(self.summary)
            lines.append("")
        
        # Agenda Items
        if self.agenda_items:
            lines.append("## Agenda Items")
            for item in self.agenda_items:
                title_with_speaker = f"{item.title}"
                if item.speaker:
                    title_with_speaker += f" (led by {item.speaker})"
                lines.append(f"### {title_with_speaker}")
                lines.append(item.description)
                if item.time_discussed:
                    lines.append(f"*Discussed at: {item.time_discussed}*")
                lines.append("")

        # Discussion Points
        if self.discussion_points:
            lines.append("## Discussion Points")
            for point in self.discussion_points:
                topic_with_speakers = f"{point.topic}"
                if point.speakers:
                    speakers_str = ", ".join(point.speakers)
                    topic_with_speakers += f" ({speakers_str})"
                lines.append(f"### {topic_with_speakers}")
                for key_point in point.key_points:
                    lines.append(f"- {key_point}")
                if point.timestamp:
                    lines.append(f"*Time: {point.timestamp}*")
                lines.append("")

        # Decisions
        if self.decisions:
            lines.append("## Decisions Made")
            for decision in self.decisions:
                decision_text = f"**{decision.decision}**"
                if decision.decided_by:
                    decision_text += f" (by {decision.decided_by})"
                lines.append(f"- {decision_text}")
                if decision.rationale:
                    lines.append(f"  - Rationale: {decision.rationale}")
                if decision.timestamp:
                    lines.append(f"  - Time: {decision.timestamp}")
                lines.append("")
        
        # Action Items
        if self.action_items:
            lines.append("## Action Items")
            for item in self.action_items:
                assignee_text = f" ({item.assignee})" if item.assignee else ""
                priority_text = f" [{item.priority.upper()}]" if item.priority != "medium" else ""
                lines.append(f"- {item.task}{assignee_text}{priority_text}")
                if item.due_date:
                    lines.append(f"  - Due: {item.due_date}")
            lines.append("")
        
        return "\n".join(lines)


class MeetingSession(BaseModel):
    """Current meeting session state."""
    start_time: datetime = Field(default_factory=datetime.now)
    audio_file: Optional[str] = None
    transcript: Optional[TranscriptResult] = None
    notes: Optional[MeetingNotes] = None
    status: str = "idle"  # "idle", "recording", "transcribing", "generating_notes", "completed", "error"
    error_message: Optional[str] = None
    processing_progress: float = 0.0
    
    def get_status_display(self) -> str:
        """Get user-friendly status display."""
        status_map = {
            "idle": "Ready to start recording",
            "recording": "Recording in progress...",
            "transcribing": f"Transcribing audio... ({self.processing_progress:.0f}%)",
            "generating_notes": f"Generating meeting notes... ({self.processing_progress:.0f}%)",
            "completed": "Meeting notes ready",
            "error": f"Error: {self.error_message or 'Unknown error'}"
        }
        return status_map.get(self.status, self.status)