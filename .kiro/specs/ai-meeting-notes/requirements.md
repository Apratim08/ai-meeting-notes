# Requirements Document

## Introduction

An AI-powered meeting notes application that captures audio from system speakers when manually triggered, transcribes it to text, and uses an open-source LLM to generate structured meeting notes with agenda items, discussion points, and action items. The application provides a simple web interface for current meeting monitoring.

## Requirements

### Requirement 1

**User Story:** As a meeting participant, I want to manually start audio capture from my system speakers, so that I can control when meeting recording begins and ends.

#### Acceptance Criteria

1. WHEN I manually start the application THEN the system SHALL detect and connect to the default system audio output device
2. WHEN audio is playing through system speakers THEN the system SHALL capture the audio stream in real-time
3. WHEN no audio is detected for 30 seconds THEN the system SHALL pause recording to avoid capturing silence
4. WHEN audio resumes after a pause THEN the system SHALL automatically resume recording
5. IF the system audio device is unavailable THEN the system SHALL display an error message and suggest troubleshooting steps

### Requirement 2

**User Story:** As a meeting participant, I want the captured audio to be transcribed to text in real-time, so that I can see the conversation as it happens.

#### Acceptance Criteria

1. WHEN audio is captured THEN the system SHALL convert speech to text using an open-source speech recognition model
2. WHEN transcription is generated THEN the system SHALL display the text in real-time with timestamps
3. WHEN multiple speakers are detected THEN the system SHALL attempt to identify different speakers in the transcript
4. IF transcription confidence is below 70% THEN the system SHALL mark uncertain text with indicators
5. WHEN transcription is complete THEN the system SHALL save the full transcript with metadata

### Requirement 3

**User Story:** As a meeting participant, I want the transcribed text to be processed into structured meeting notes when the meeting ends, so that I can quickly review key points without reading the entire transcript.

#### Acceptance Criteria

1. WHEN no audio is detected for 1 minute OR when I manually stop recording THEN the system SHALL process the transcript using an open-source LLM
2. WHEN processing the transcript THEN the system SHALL generate notes in a structured format containing agenda items, discussion points, and action items
3. WHEN generating notes THEN the system SHALL identify and extract key decisions made during the meeting
4. WHEN generating notes THEN the system SHALL identify participants mentioned by name
5. IF the transcript is longer than the LLM context window THEN the system SHALL process it in chunks and merge results

### Requirement 4

**User Story:** As a meeting participant, I want to view the current meeting status and generated notes through a simple web interface, so that I can monitor the transcription and copy the final notes.

#### Acceptance Criteria

1. WHEN accessing the web interface THEN the system SHALL display the current meeting status (recording, processing, or idle)
2. WHEN a meeting is in progress THEN the system SHALL display real-time transcription text
3. WHEN notes are generated THEN the system SHALL display the structured notes in a readable format
4. WHEN viewing generated notes THEN the user SHALL be able to copy the text content
5. WHEN starting a new meeting THEN the system SHALL clear previous meeting data from the interface

### Requirement 5

**User Story:** As a meeting participant, I want the application to handle errors gracefully, so that technical issues don't interrupt my meeting or cause data loss.

#### Acceptance Criteria

1. WHEN an audio capture error occurs THEN the system SHALL attempt to reconnect automatically and log the issue
2. WHEN transcription fails THEN the system SHALL retry with fallback models and preserve the original audio
3. WHEN LLM processing fails THEN the system SHALL display the transcript and allow manual retry later
4. WHEN the web server encounters errors THEN the system SHALL display user-friendly error messages
5. WHEN any component fails THEN the system SHALL ensure no data loss and maintain system stability