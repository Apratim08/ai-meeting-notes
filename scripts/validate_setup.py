#!/usr/bin/env python3
"""Validation script to test the project setup."""

import sys
from pathlib import Path

def validate_project_structure():
    """Validate that all required directories and files exist."""
    required_files = [
        "ai_meeting_notes/__init__.py",
        "ai_meeting_notes/config.py",
        "ai_meeting_notes/audio_recorder.py",
        "ai_meeting_notes/transcription.py",
        "ai_meeting_notes/notes_generator.py",
        "ai_meeting_notes/models.py",
        "ai_meeting_notes/file_manager.py",
        "ai_meeting_notes/main.py",
        "ai_meeting_notes/api/__init__.py",
        "requirements.txt",
        "setup.py",
        "README.md",
        ".env.example",
        ".gitignore"
    ]
    
    required_dirs = [
        "ai_meeting_notes",
        "ai_meeting_notes/api",
        "ai_meeting_notes/templates",
        "ai_meeting_notes/static",
        "tests"
    ]
    
    print("Validating project structure...")
    
    # Check directories
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            print(f"❌ Missing directory: {dir_path}")
            return False
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    # Check files
    for file_path in required_files:
        if not Path(file_path).is_file():
            print(f"❌ Missing file: {file_path}")
            return False
        else:
            print(f"✅ File exists: {file_path}")
    
    return True

def validate_configuration():
    """Validate configuration system (requires dependencies)."""
    try:
        from ai_meeting_notes.config import config
        print("\n✅ Configuration system loaded successfully")
        print(f"   Audio sample rate: {config.audio.sample_rate}")
        print(f"   Whisper model: {config.transcription.model_name}")
        print(f"   LLM model: {config.llm.model_name}")
        print(f"   Server port: {config.server.port}")
        
        # Test directory creation
        config.ensure_directories()
        if config.files.temp_dir.exists():
            print(f"✅ Temp directory created: {config.files.temp_dir}")
        
        return True
    except ImportError as e:
        print(f"⚠️  Configuration validation skipped (dependencies not installed): {e}")
        return True  # This is expected before pip install
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def main():
    """Run all validation checks."""
    print("AI Meeting Notes - Project Setup Validation")
    print("=" * 50)
    
    structure_ok = validate_project_structure()
    config_ok = validate_configuration()
    
    print("\n" + "=" * 50)
    if structure_ok and config_ok:
        print("✅ Project setup validation PASSED")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install Ollama and pull model: ollama pull llama3.1:8b")
        print("3. Set up BlackHole audio driver (see README.md)")
        print("4. Run the application: python -m ai_meeting_notes.main")
        return 0
    else:
        print("❌ Project setup validation FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())