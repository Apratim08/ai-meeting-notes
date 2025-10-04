#!/usr/bin/env python3
import json
from ai_meeting_notes.notes_generator import NotesGenerator
from ai_meeting_notes.config import AppConfig

# Load config (uses qwen2.5:14b)
config = AppConfig.load_from_env()

print(f"Testing with model: {config.llm.model_name}")
print(f"Generating notes from transcript...\n")

# Load transcript
with open('temp/transcripts/transcript.json', 'r') as f:
    data = json.load(f)
    
transcript_text = data['transcript']['full_text']
duration_minutes = data['transcript']['duration'] / 60.0

# Generate notes
generator = NotesGenerator(config)
notes = generator.generate_notes(transcript_text, duration_minutes)

# Save results
with open('temp/transcripts/notes_qwen.json', 'w') as f:
    json.dump(notes.model_dump(), f, indent=2, default=str)

# Print summary
print("✅ Notes generated!\n")
print(notes.to_formatted_text())
print(f"\n💾 Saved to: temp/transcripts/notes_qwen.json")
