#!/usr/bin/env python3
"""
Demo script for the NotesGenerator service.

This script demonstrates how to use the NotesGenerator to convert
meeting transcripts into structured notes using Ollama.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import ai_meeting_notes
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_meeting_notes.notes_generator import NotesGenerator, OllamaError
from ai_meeting_notes.config import AppConfig, LLMConfig


def main():
    """Run the notes generation demo."""
    print("🤖 AI Meeting Notes - Notes Generation Demo")
    print("=" * 50)
    
    # Sample meeting transcript
    sample_transcript = """
    John: Good morning everyone, let's start our weekly team meeting. Today is March 15th.
    
    Sarah: Hi John, I have an update on the project timeline. We've completed the design phase 
    and are ready to move to development.
    
    John: Great news Sarah! What's our next step?
    
    Sarah: I think we should assign the frontend work to Mike and backend to Lisa. 
    The design mockups are ready and the API specifications are documented.
    
    Mike: Sounds good to me. I can start on the UI components next week. 
    How long do we have for this phase?
    
    Lisa: I'll handle the API development. I estimate about 3 weeks for the backend work.
    
    John: Perfect. Let's target end of month for completion. Sarah, can you create 
    the development tickets and set up the project board?
    
    Sarah: Absolutely, I'll have them ready by Friday. I'll also schedule daily standups 
    starting Monday.
    
    Mike: Should we also plan for code reviews? I think we should have at least 
    two reviewers for each pull request.
    
    Lisa: Good idea. Let's make that a requirement.
    
    John: Agreed. We decided that all code must be reviewed by at least two team members 
    before merging. Any other items? No? Great, meeting adjourned.
    """
    
    # Create configuration
    config = AppConfig(
        llm=LLMConfig(
            model_name="llama3.1:8b",  # Make sure this model is available
            temperature=0.3,
            max_retries=2,
            retry_delay=1.0
        )
    )
    
    # Create notes generator
    generator = NotesGenerator(config)
    
    # Check if Ollama is available
    print("🔍 Checking Ollama availability...")
    if not generator.check_ollama_available():
        print("❌ Ollama is not available or the required model is not installed.")
        print("\nTo fix this:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Pull the model: ollama pull llama3.1:8b")
        print("3. Make sure Ollama is running: ollama serve")
        return
    
    print("✅ Ollama is available!")
    
    # Generate notes
    print("\n📝 Generating meeting notes...")
    print("Transcript length:", len(sample_transcript), "characters")
    
    try:
        # This would normally take 1-2 minutes with a real LLM
        notes = generator.generate_notes(sample_transcript, meeting_duration=12.5)
        
        print("✅ Notes generated successfully!")
        print("\n" + "=" * 50)
        print("📋 GENERATED MEETING NOTES")
        print("=" * 50)
        
        # Display formatted notes
        formatted_notes = notes.to_formatted_text()
        print(formatted_notes)
        
        print("\n" + "=" * 50)
        print("📊 NOTES SUMMARY")
        print("=" * 50)
        print(f"Participants: {len(notes.participants)}")
        print(f"Agenda Items: {len(notes.agenda_items)}")
        print(f"Discussion Points: {len(notes.discussion_points)}")
        print(f"Action Items: {len(notes.action_items)}")
        print(f"Decisions: {len(notes.decisions)}")
        
        if notes.action_items:
            print("\n🎯 ACTION ITEMS:")
            for i, item in enumerate(notes.action_items, 1):
                assignee = f" ({item.assignee})" if item.assignee else ""
                priority = f" [{item.priority.upper()}]" if item.priority != "medium" else ""
                print(f"  {i}. {item.task}{assignee}{priority}")
        
    except OllamaError as e:
        print(f"❌ Failed to generate notes: {e}")
        print("\nThis could be due to:")
        print("- Ollama server not running")
        print("- Model not available")
        print("- Network connectivity issues")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()