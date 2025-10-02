"""
Setup validation and user guidance for AI Meeting Notes application.

Provides comprehensive system validation, setup instructions, and troubleshooting
for BlackHole audio routing, Ollama LLM setup, and system requirements.
"""

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import requests
import sounddevice as sd

from .config import config

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Status levels for validation checks."""
    PASS = "pass"
    WARNING = "warning" 
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    status: ValidationStatus
    message: str
    details: Optional[str] = None
    fix_instructions: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "fix_instructions": self.fix_instructions
        }


@dataclass
class SetupValidationReport:
    """Complete setup validation report."""
    overall_status: ValidationStatus
    results: List[ValidationResult]
    setup_complete: bool
    next_steps: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_status": self.overall_status.value,
            "results": [result.to_dict() for result in self.results],
            "setup_complete": self.setup_complete,
            "next_steps": self.next_steps
        }


class SetupValidator:
    """
    Comprehensive setup validation for AI Meeting Notes.
    
    Validates system requirements, audio setup, LLM availability,
    and provides detailed setup instructions and troubleshooting.
    """
    
    def __init__(self):
        self.config = config
        self.validation_results: List[ValidationResult] = []
    
    def run_full_validation(self) -> SetupValidationReport:
        """
        Run complete setup validation and return comprehensive report.
        
        Returns:
            SetupValidationReport: Complete validation results with next steps
        """
        logger.info("Starting full setup validation...")
        self.validation_results.clear()
        
        # System requirements
        self._validate_operating_system()
        self._validate_python_version()
        self._validate_dependencies()
        
        # Audio setup
        self._validate_blackhole_installation()
        self._validate_multi_output_device()
        self._validate_audio_permissions()
        
        # LLM setup
        self._validate_ollama_installation()
        self._validate_ollama_model()
        
        # File system
        self._validate_disk_space()
        self._validate_temp_directory()
        
        # Generate report
        return self._generate_report()
    
    def validate_audio_setup_only(self) -> SetupValidationReport:
        """
        Validate only audio-related setup components.
        
        Returns:
            SetupValidationReport: Audio setup validation results
        """
        logger.info("Starting audio setup validation...")
        self.validation_results.clear()
        
        self._validate_blackhole_installation()
        self._validate_multi_output_device()
        self._validate_audio_permissions()
        
        return self._generate_report()
    
    def validate_llm_setup_only(self) -> SetupValidationReport:
        """
        Validate only LLM-related setup components.
        
        Returns:
            SetupValidationReport: LLM setup validation results
        """
        logger.info("Starting LLM setup validation...")
        self.validation_results.clear()
        
        self._validate_ollama_installation()
        self._validate_ollama_model()
        
        return self._generate_report()
    
    def get_setup_wizard_steps(self) -> List[Dict[str, Any]]:
        """
        Get step-by-step setup wizard instructions.
        
        Returns:
            List of setup steps with instructions and validation
        """
        return [
            {
                "step": 1,
                "title": "Install BlackHole Audio Driver",
                "description": "Install BlackHole virtual audio driver for system audio capture",
                "instructions": self._get_blackhole_installation_instructions(),
                "validation_method": "blackhole_installation",
                "required": True
            },
            {
                "step": 2,
                "title": "Create Multi-Output Device",
                "description": "Set up audio routing to capture meeting audio",
                "instructions": self._get_multi_output_setup_instructions(),
                "validation_method": "multi_output_device",
                "required": True
            },
            {
                "step": 3,
                "title": "Grant Audio Permissions",
                "description": "Allow the application to access microphone/audio input",
                "instructions": self._get_audio_permissions_instructions(),
                "validation_method": "audio_permissions",
                "required": True
            },
            {
                "step": 4,
                "title": "Install Ollama",
                "description": "Install Ollama for local LLM processing",
                "instructions": self._get_ollama_installation_instructions(),
                "validation_method": "ollama_installation",
                "required": True
            },
            {
                "step": 5,
                "title": "Download LLM Model",
                "description": "Download the required language model for notes generation",
                "instructions": self._get_model_download_instructions(),
                "validation_method": "ollama_model",
                "required": True
            },
            {
                "step": 6,
                "title": "Test Complete Setup",
                "description": "Verify all components are working together",
                "instructions": self._get_integration_test_instructions(),
                "validation_method": "full_validation",
                "required": False
            }
        ]
    
    def _validate_operating_system(self) -> None:
        """Validate operating system compatibility."""
        system = platform.system()
        
        if system == "Darwin":  # macOS
            version = platform.mac_ver()[0]
            major_version = int(version.split('.')[0]) if version else 0
            
            if major_version >= 10:  # macOS 10.0+
                self.validation_results.append(ValidationResult(
                    name="operating_system",
                    status=ValidationStatus.PASS,
                    message=f"macOS {version} is supported"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    name="operating_system",
                    status=ValidationStatus.WARNING,
                    message=f"macOS {version} may have compatibility issues",
                    details="Recommend macOS 10.15 or later for best compatibility"
                ))
        else:
            self.validation_results.append(ValidationResult(
                name="operating_system",
                status=ValidationStatus.FAIL,
                message=f"{system} is not supported",
                details="This application is designed specifically for macOS",
                fix_instructions="Please run this application on macOS"
            ))
    
    def _validate_python_version(self) -> None:
        """Validate Python version compatibility."""
        version = sys.version_info
        
        if version >= (3, 8):
            self.validation_results.append(ValidationResult(
                name="python_version",
                status=ValidationStatus.PASS,
                message=f"Python {version[0]}.{version[1]}.{version[2]} is supported"
            ))
        else:
            self.validation_results.append(ValidationResult(
                name="python_version",
                status=ValidationStatus.FAIL,
                message=f"Python {version[0]}.{version[1]} is too old",
                details="Python 3.8 or later is required",
                fix_instructions="Please upgrade to Python 3.8 or later"
            ))
    
    def _validate_dependencies(self) -> None:
        """Validate required Python dependencies."""
        required_packages = [
            "sounddevice", "numpy", "fastapi", "uvicorn", 
            "requests", "pydantic", "faster-whisper"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if not missing_packages:
            self.validation_results.append(ValidationResult(
                name="dependencies",
                status=ValidationStatus.PASS,
                message="All required Python packages are installed"
            ))
        else:
            self.validation_results.append(ValidationResult(
                name="dependencies",
                status=ValidationStatus.FAIL,
                message=f"Missing required packages: {', '.join(missing_packages)}",
                fix_instructions=f"Install missing packages: pip install {' '.join(missing_packages)}"
            ))
    
    def _validate_blackhole_installation(self) -> None:
        """Validate BlackHole virtual audio driver installation."""
        try:
            devices = sd.query_devices()
            blackhole_found = False
            blackhole_devices = []
            
            for device in devices:
                if isinstance(device, dict):
                    name = device.get('name', '').lower()
                    if 'blackhole' in name:
                        blackhole_found = True
                        blackhole_devices.append(device['name'])
            
            if blackhole_found:
                self.validation_results.append(ValidationResult(
                    name="blackhole_installation",
                    status=ValidationStatus.PASS,
                    message="BlackHole audio driver is installed",
                    details=f"Found devices: {', '.join(blackhole_devices)}"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    name="blackhole_installation",
                    status=ValidationStatus.FAIL,
                    message="BlackHole audio driver not found",
                    fix_instructions=self._get_blackhole_installation_instructions()
                ))
                
        except Exception as e:
            self.validation_results.append(ValidationResult(
                name="blackhole_installation",
                status=ValidationStatus.FAIL,
                message="Could not check audio devices",
                details=str(e),
                fix_instructions="Check system audio settings and try again"
            ))
    
    def _validate_multi_output_device(self) -> None:
        """Validate Multi-Output Device configuration."""
        try:
            devices = sd.query_devices()
            multi_output_found = False
            multi_output_devices = []
            
            for device in devices:
                if isinstance(device, dict):
                    name = device.get('name', '').lower()
                    if 'multi-output' in name or 'aggregate' in name:
                        multi_output_found = True
                        multi_output_devices.append(device['name'])
            
            if multi_output_found:
                self.validation_results.append(ValidationResult(
                    name="multi_output_device",
                    status=ValidationStatus.PASS,
                    message="Multi-Output Device is configured",
                    details=f"Found devices: {', '.join(multi_output_devices)}"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    name="multi_output_device",
                    status=ValidationStatus.WARNING,
                    message="Multi-Output Device not found",
                    details="You'll need to create this for meeting audio capture",
                    fix_instructions=self._get_multi_output_setup_instructions()
                ))
                
        except Exception as e:
            self.validation_results.append(ValidationResult(
                name="multi_output_device",
                status=ValidationStatus.FAIL,
                message="Could not check Multi-Output Device",
                details=str(e)
            ))
    
    def _validate_audio_permissions(self) -> None:
        """Validate audio input permissions."""
        try:
            # Try to query default input device
            default_input = sd.query_devices(kind='input')
            
            if default_input:
                self.validation_results.append(ValidationResult(
                    name="audio_permissions",
                    status=ValidationStatus.PASS,
                    message="Audio input permissions are granted"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    name="audio_permissions",
                    status=ValidationStatus.WARNING,
                    message="No audio input device found",
                    fix_instructions=self._get_audio_permissions_instructions()
                ))
                
        except Exception as e:
            self.validation_results.append(ValidationResult(
                name="audio_permissions",
                status=ValidationStatus.FAIL,
                message="Audio permissions may be denied",
                details=str(e),
                fix_instructions=self._get_audio_permissions_instructions()
            ))
    
    def _validate_ollama_installation(self) -> None:
        """Validate Ollama installation and availability."""
        try:
            # Check if ollama command exists
            result = subprocess.run(['which', 'ollama'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Check if Ollama service is running
                try:
                    response = requests.get(f"{self.config.llm.ollama_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        self.validation_results.append(ValidationResult(
                            name="ollama_installation",
                            status=ValidationStatus.PASS,
                            message="Ollama is installed and running"
                        ))
                    else:
                        self.validation_results.append(ValidationResult(
                            name="ollama_installation",
                            status=ValidationStatus.WARNING,
                            message="Ollama is installed but not running",
                            fix_instructions="Start Ollama service: ollama serve"
                        ))
                except requests.RequestException:
                    self.validation_results.append(ValidationResult(
                        name="ollama_installation",
                        status=ValidationStatus.WARNING,
                        message="Ollama is installed but service is not accessible",
                        fix_instructions="Start Ollama service: ollama serve"
                    ))
            else:
                self.validation_results.append(ValidationResult(
                    name="ollama_installation",
                    status=ValidationStatus.FAIL,
                    message="Ollama is not installed",
                    fix_instructions=self._get_ollama_installation_instructions()
                ))
                
        except subprocess.TimeoutExpired:
            self.validation_results.append(ValidationResult(
                name="ollama_installation",
                status=ValidationStatus.FAIL,
                message="Could not check Ollama installation",
                details="Command timeout"
            ))
        except Exception as e:
            self.validation_results.append(ValidationResult(
                name="ollama_installation",
                status=ValidationStatus.FAIL,
                message="Error checking Ollama installation",
                details=str(e)
            ))
    
    def _validate_ollama_model(self) -> None:
        """Validate required Ollama model availability."""
        try:
            response = requests.get(f"{self.config.llm.ollama_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                if self.config.llm.model_name in model_names:
                    self.validation_results.append(ValidationResult(
                        name="ollama_model",
                        status=ValidationStatus.PASS,
                        message=f"Model {self.config.llm.model_name} is available"
                    ))
                else:
                    self.validation_results.append(ValidationResult(
                        name="ollama_model",
                        status=ValidationStatus.FAIL,
                        message=f"Model {self.config.llm.model_name} not found",
                        details=f"Available models: {', '.join(model_names) if model_names else 'None'}",
                        fix_instructions=self._get_model_download_instructions()
                    ))
            else:
                self.validation_results.append(ValidationResult(
                    name="ollama_model",
                    status=ValidationStatus.FAIL,
                    message="Could not check available models",
                    details=f"Ollama API returned status {response.status_code}"
                ))
                
        except requests.RequestException as e:
            self.validation_results.append(ValidationResult(
                name="ollama_model",
                status=ValidationStatus.FAIL,
                message="Could not connect to Ollama to check models",
                details=str(e),
                fix_instructions="Ensure Ollama is running: ollama serve"
            ))
    
    def _validate_disk_space(self) -> None:
        """Validate available disk space for audio files and models."""
        try:
            temp_dir = Path(self.config.files.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Get disk usage
            stat = os.statvfs(str(temp_dir))
            free_bytes = stat.f_bavail * stat.f_frsize
            free_gb = free_bytes / (1024**3)
            
            if free_gb >= 10:  # 10GB recommended
                self.validation_results.append(ValidationResult(
                    name="disk_space",
                    status=ValidationStatus.PASS,
                    message=f"Sufficient disk space available ({free_gb:.1f} GB free)"
                ))
            elif free_gb >= 5:  # 5GB minimum
                self.validation_results.append(ValidationResult(
                    name="disk_space",
                    status=ValidationStatus.WARNING,
                    message=f"Limited disk space ({free_gb:.1f} GB free)",
                    details="Recommend at least 10GB for optimal operation"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    name="disk_space",
                    status=ValidationStatus.FAIL,
                    message=f"Insufficient disk space ({free_gb:.1f} GB free)",
                    details="At least 5GB required for audio files and models",
                    fix_instructions="Free up disk space or change temp directory location"
                ))
                
        except Exception as e:
            self.validation_results.append(ValidationResult(
                name="disk_space",
                status=ValidationStatus.WARNING,
                message="Could not check disk space",
                details=str(e)
            ))
    
    def _validate_temp_directory(self) -> None:
        """Validate temp directory accessibility."""
        try:
            temp_dir = Path(self.config.files.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write access
            test_file = temp_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            
            self.validation_results.append(ValidationResult(
                name="temp_directory",
                status=ValidationStatus.PASS,
                message=f"Temp directory is accessible: {temp_dir}"
            ))
            
        except Exception as e:
            self.validation_results.append(ValidationResult(
                name="temp_directory",
                status=ValidationStatus.FAIL,
                message="Cannot access temp directory",
                details=str(e),
                fix_instructions="Check directory permissions or change temp directory location"
            ))
    
    def _generate_report(self) -> SetupValidationReport:
        """Generate comprehensive validation report."""
        # Determine overall status
        has_failures = any(r.status == ValidationStatus.FAIL for r in self.validation_results)
        has_warnings = any(r.status == ValidationStatus.WARNING for r in self.validation_results)
        
        if has_failures:
            overall_status = ValidationStatus.FAIL
        elif has_warnings:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASS
        
        # Determine if setup is complete
        critical_checks = ["blackhole_installation", "ollama_installation", "ollama_model"]
        setup_complete = all(
            any(r.name == check and r.status == ValidationStatus.PASS 
                for r in self.validation_results)
            for check in critical_checks
        )
        
        # Generate next steps
        next_steps = []
        for result in self.validation_results:
            if result.status == ValidationStatus.FAIL and result.fix_instructions:
                next_steps.append(f"Fix {result.name}: {result.fix_instructions}")
        
        if not next_steps and setup_complete:
            next_steps.append("Setup is complete! You can start using AI Meeting Notes.")
        elif not next_steps:
            next_steps.append("Review warnings and consider addressing them for optimal performance.")
        
        return SetupValidationReport(
            overall_status=overall_status,
            results=self.validation_results.copy(),
            setup_complete=setup_complete,
            next_steps=next_steps
        )
    
    # Setup instruction methods
    
    def _get_blackhole_installation_instructions(self) -> str:
        """Get detailed BlackHole installation instructions."""
        return """
# Install BlackHole Audio Driver

## Method 1: Download from GitHub (Recommended)
1. Visit: https://github.com/ExistentialAudio/BlackHole
2. Download the latest release (BlackHole.2ch.pkg)
3. Double-click the installer and follow the prompts
4. Restart your Mac after installation

## Method 2: Install via Homebrew
```bash
brew install blackhole-2ch
```

## Verification
After installation, you should see "BlackHole 2ch" in:
- System Preferences > Sound > Input
- Audio MIDI Setup > Audio Devices

## Troubleshooting
- If BlackHole doesn't appear, restart your Mac
- Check System Preferences > Security & Privacy for blocked installations
- Ensure you downloaded the 2ch version (not 16ch or 64ch)
"""
    
    def _get_multi_output_setup_instructions(self) -> str:
        """Get Multi-Output Device setup instructions."""
        return """
# Create Multi-Output Device

## Step-by-Step Instructions
1. Open "Audio MIDI Setup" (Applications > Utilities > Audio MIDI Setup)
2. Click the "+" button in the bottom-left corner
3. Select "Create Multi-Output Device"
4. In the new Multi-Output Device panel:
   - Check the box next to your speakers/headphones (e.g., "MacBook Pro Speakers")
   - Check the box next to "BlackHole 2ch"
   - Right-click on "BlackHole 2ch" and select "Use This Device For Sound Output"
5. Rename the device to "Meeting Audio" for easy identification
6. Close Audio MIDI Setup

## Configure System Audio
1. Go to System Preferences > Sound > Output
2. Select your new "Meeting Audio" device
3. Test by playing some audio - you should hear it normally

## During Meetings
- Set system output to "Meeting Audio" before joining meetings
- Audio will play through your speakers AND be captured by BlackHole
- Switch back to regular speakers when not recording meetings

## Troubleshooting
- If you don't hear audio: Ensure your speakers are checked in the Multi-Output Device
- If audio is choppy: Try increasing buffer size in Audio MIDI Setup preferences
- If BlackHole isn't listed: Restart Audio MIDI Setup after BlackHole installation
"""
    
    def _get_audio_permissions_instructions(self) -> str:
        """Get audio permissions setup instructions."""
        return """
# Grant Audio Permissions

## macOS Audio Permissions
1. Go to System Preferences > Security & Privacy > Privacy
2. Select "Microphone" from the left sidebar
3. Ensure your terminal/Python application is checked
4. If not listed, click "+" and add your Python executable or terminal app

## For Terminal Applications
If running from Terminal:
1. The first time you run the app, macOS will ask for microphone permission
2. Click "OK" to grant permission
3. If you accidentally denied it, go to Privacy settings and manually enable it

## For Packaged Applications
If using a packaged version:
1. The app should automatically request permissions on first run
2. Grant permission when prompted
3. Check Privacy settings if audio input isn't working

## Verification
- Run the app and check if audio devices are detected
- Look for "BlackHole 2ch" in the available input devices
- Test recording a short audio clip to verify permissions
"""
    
    def _get_ollama_installation_instructions(self) -> str:
        """Get Ollama installation instructions."""
        return """
# Install Ollama

## Method 1: Download from Website (Recommended)
1. Visit: https://ollama.ai
2. Click "Download for macOS"
3. Open the downloaded .dmg file
4. Drag Ollama to Applications folder
5. Launch Ollama from Applications

## Method 2: Install via Homebrew
```bash
brew install ollama
```

## Method 3: Install via curl
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

## Start Ollama Service
After installation, start the Ollama service:
```bash
ollama serve
```

Or launch the Ollama app from Applications (it will start the service automatically).

## Verification
Check if Ollama is running:
```bash
ollama list
```

You should see a list of installed models (may be empty initially).

## Troubleshooting
- If `ollama` command not found: Restart your terminal or add to PATH
- If service won't start: Check if port 11434 is already in use
- For permission issues: Ensure Ollama has necessary system permissions
"""
    
    def _get_model_download_instructions(self) -> str:
        """Get model download instructions."""
        return f"""
# Download Required LLM Model

## Download {self.config.llm.model_name}
```bash
ollama pull {self.config.llm.model_name}
```

This will download the model (approximately 4-8GB depending on the model).

## Alternative Models
If the default model is too large or slow, you can try:
- `llama3.1:8b` (recommended, ~4.7GB)
- `llama3.2:3b` (smaller, faster, ~2GB)
- `llama3.2:1b` (smallest, ~1.3GB)

To use a different model:
1. Download it: `ollama pull <model-name>`
2. Update your configuration to use the new model name

## Verify Installation
```bash
ollama list
```

You should see {self.config.llm.model_name} in the list.

## Test the Model
```bash
ollama run {self.config.llm.model_name} "Hello, how are you?"
```

The model should respond with a greeting.

## Troubleshooting
- If download is slow: Models are large, be patient
- If download fails: Check internet connection and disk space
- If model won't run: Ensure you have enough RAM (8GB+ recommended)
"""
    
    def _get_integration_test_instructions(self) -> str:
        """Get integration test instructions."""
        return """
# Test Complete Setup

## Quick System Test
1. Start the AI Meeting Notes application
2. Check the health endpoint: http://localhost:8000/api/health
3. All services should show as "healthy"

## Audio Test
1. Set system output to your "Meeting Audio" device
2. Play some audio (music, video, etc.)
3. You should hear the audio normally
4. Start recording in the app - it should detect audio input

## Full Integration Test
1. Start a test recording
2. Play some speech audio (YouTube video, podcast, etc.)
3. Stop recording after 30-60 seconds
4. Wait for transcription to complete
5. Wait for notes generation to complete
6. Verify you get structured meeting notes

## Troubleshooting Common Issues
- No audio detected: Check Multi-Output Device configuration
- Transcription fails: Verify faster-whisper is installed
- Notes generation fails: Check Ollama is running and model is available
- Web interface not loading: Check if port 8000 is available

## Performance Expectations
- Transcription: ~2-3 minutes for 1 hour of audio
- Notes generation: ~1-2 minutes regardless of meeting length
- Memory usage: 2-4GB during processing
"""


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information for troubleshooting."""
    info = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "path": sys.path[:3]  # First few paths only
        }
    }
    
    # Add macOS specific info
    if platform.system() == "Darwin":
        mac_ver = platform.mac_ver()
        info["platform"]["mac_version"] = mac_ver[0]
        info["platform"]["mac_dev_stage"] = mac_ver[1]
        info["platform"]["mac_machine"] = mac_ver[2]
    
    # Add audio device info
    try:
        devices = sd.query_devices()
        info["audio"] = {
            "default_input": sd.query_devices(kind='input'),
            "default_output": sd.query_devices(kind='output'),
            "device_count": len(devices) if devices else 0
        }
    except Exception as e:
        info["audio"] = {"error": str(e)}
    
    return info