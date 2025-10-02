"""
Tests for setup validation and user guidance functionality.

Tests different system configurations and validation scenarios.
"""

import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import requests

from ai_meeting_notes.setup_validator import (
    SetupValidator, ValidationStatus, ValidationResult, 
    SetupValidationReport, get_system_info
)
from ai_meeting_notes.config import config


class TestSetupValidator:
    """Test setup validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SetupValidator()
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation and serialization."""
        result = ValidationResult(
            name="test_check",
            status=ValidationStatus.PASS,
            message="Test passed",
            details="Additional details",
            fix_instructions="Fix instructions"
        )
        
        assert result.name == "test_check"
        assert result.status == ValidationStatus.PASS
        assert result.message == "Test passed"
        
        # Test serialization
        result_dict = result.to_dict()
        assert result_dict["name"] == "test_check"
        assert result_dict["status"] == "pass"
        assert result_dict["message"] == "Test passed"
    
    def test_validation_report_creation(self):
        """Test SetupValidationReport creation and serialization."""
        results = [
            ValidationResult("test1", ValidationStatus.PASS, "Test 1 passed"),
            ValidationResult("test2", ValidationStatus.FAIL, "Test 2 failed")
        ]
        
        report = SetupValidationReport(
            overall_status=ValidationStatus.FAIL,
            results=results,
            setup_complete=False,
            next_steps=["Fix test2"]
        )
        
        assert report.overall_status == ValidationStatus.FAIL
        assert len(report.results) == 2
        assert not report.setup_complete
        
        # Test serialization
        report_dict = report.to_dict()
        assert report_dict["overall_status"] == "fail"
        assert len(report_dict["results"]) == 2
        assert report_dict["next_steps"] == ["Fix test2"]
    
    @patch('platform.system')
    def test_validate_operating_system_macos(self, mock_system):
        """Test macOS validation."""
        mock_system.return_value = "Darwin"
        
        with patch('platform.mac_ver', return_value=('12.0', '', '')):
            self.validator._validate_operating_system()
            
            result = self.validator.validation_results[-1]
            assert result.name == "operating_system"
            assert result.status == ValidationStatus.PASS
            assert "macOS 12.0 is supported" in result.message
    
    @patch('platform.system')
    def test_validate_operating_system_unsupported(self, mock_system):
        """Test unsupported OS validation."""
        mock_system.return_value = "Windows"
        
        self.validator._validate_operating_system()
        
        result = self.validator.validation_results[-1]
        assert result.name == "operating_system"
        assert result.status == ValidationStatus.FAIL
        assert "Windows is not supported" in result.message
    
    def test_validate_python_version_supported(self):
        """Test Python version validation with supported version."""
        with patch('sys.version_info', (3, 9, 0)):
            self.validator._validate_python_version()
            
            result = self.validator.validation_results[-1]
            assert result.name == "python_version"
            assert result.status == ValidationStatus.PASS
    
    def test_validate_python_version_unsupported(self):
        """Test Python version validation with unsupported version."""
        with patch('sys.version_info', (3, 7, 0)):
            self.validator._validate_python_version()
            
            result = self.validator.validation_results[-1]
            assert result.name == "python_version"
            assert result.status == ValidationStatus.FAIL
            assert "Python 3.7 is too old" in result.message
    
    @patch('sounddevice.query_devices')
    def test_validate_blackhole_installation_found(self, mock_query_devices):
        """Test BlackHole validation when device is found."""
        mock_devices = [
            {'name': 'Built-in Microphone', 'max_input_channels': 1},
            {'name': 'BlackHole 2ch', 'max_input_channels': 2},
            {'name': 'Built-in Output', 'max_input_channels': 0}
        ]
        mock_query_devices.return_value = mock_devices
        
        self.validator._validate_blackhole_installation()
        
        result = self.validator.validation_results[-1]
        assert result.name == "blackhole_installation"
        assert result.status == ValidationStatus.PASS
        assert "BlackHole audio driver is installed" in result.message
    
    @patch('sounddevice.query_devices')
    def test_validate_blackhole_installation_not_found(self, mock_query_devices):
        """Test BlackHole validation when device is not found."""
        mock_devices = [
            {'name': 'Built-in Microphone', 'max_input_channels': 1},
            {'name': 'Built-in Output', 'max_input_channels': 0}
        ]
        mock_query_devices.return_value = mock_devices
        
        self.validator._validate_blackhole_installation()
        
        result = self.validator.validation_results[-1]
        assert result.name == "blackhole_installation"
        assert result.status == ValidationStatus.FAIL
        assert "BlackHole audio driver not found" in result.message
    
    @patch('sounddevice.query_devices')
    def test_validate_multi_output_device_found(self, mock_query_devices):
        """Test Multi-Output Device validation when configured."""
        mock_devices = [
            {'name': 'Built-in Microphone', 'max_input_channels': 1},
            {'name': 'Multi-Output Device', 'max_output_channels': 2},
            {'name': 'Built-in Output', 'max_output_channels': 2}
        ]
        mock_query_devices.return_value = mock_devices
        
        self.validator._validate_multi_output_device()
        
        result = self.validator.validation_results[-1]
        assert result.name == "multi_output_device"
        assert result.status == ValidationStatus.PASS
        assert "Multi-Output Device is configured" in result.message
    
    @patch('sounddevice.query_devices')
    def test_validate_multi_output_device_not_found(self, mock_query_devices):
        """Test Multi-Output Device validation when not configured."""
        mock_devices = [
            {'name': 'Built-in Microphone', 'max_input_channels': 1},
            {'name': 'Built-in Output', 'max_output_channels': 2}
        ]
        mock_query_devices.return_value = mock_devices
        
        self.validator._validate_multi_output_device()
        
        result = self.validator.validation_results[-1]
        assert result.name == "multi_output_device"
        assert result.status == ValidationStatus.WARNING
        assert "Multi-Output Device not found" in result.message
    
    @patch('subprocess.run')
    def test_validate_ollama_installation_found(self, mock_run):
        """Test Ollama validation when installed and running."""
        # Mock successful 'which ollama' command
        mock_run.return_value = Mock(returncode=0)
        
        # Mock successful Ollama API response
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": []}
            mock_get.return_value = mock_response
            
            self.validator._validate_ollama_installation()
            
            result = self.validator.validation_results[-1]
            assert result.name == "ollama_installation"
            assert result.status == ValidationStatus.PASS
            assert "Ollama is installed and running" in result.message
    
    @patch('subprocess.run')
    def test_validate_ollama_installation_not_found(self, mock_run):
        """Test Ollama validation when not installed."""
        # Mock failed 'which ollama' command
        mock_run.return_value = Mock(returncode=1)
        
        self.validator._validate_ollama_installation()
        
        result = self.validator.validation_results[-1]
        assert result.name == "ollama_installation"
        assert result.status == ValidationStatus.FAIL
        assert "Ollama is not installed" in result.message
    
    @patch('subprocess.run')
    def test_validate_ollama_installation_not_running(self, mock_run):
        """Test Ollama validation when installed but not running."""
        # Mock successful 'which ollama' command
        mock_run.return_value = Mock(returncode=0)
        
        # Mock failed Ollama API response
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Connection refused")
            
            self.validator._validate_ollama_installation()
            
            result = self.validator.validation_results[-1]
            assert result.name == "ollama_installation"
            assert result.status == ValidationStatus.WARNING
            assert "service is not accessible" in result.message
    
    def test_validate_ollama_model_available(self):
        """Test Ollama model validation when model is available."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama3.1:8b"},
                    {"name": "other-model"}
                ]
            }
            mock_get.return_value = mock_response
            
            self.validator._validate_ollama_model()
            
            result = self.validator.validation_results[-1]
            assert result.name == "ollama_model"
            assert result.status == ValidationStatus.PASS
            assert f"Model {config.llm.model_name} is available" in result.message
    
    def test_validate_ollama_model_not_available(self):
        """Test Ollama model validation when model is not available."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "other-model"}
                ]
            }
            mock_get.return_value = mock_response
            
            self.validator._validate_ollama_model()
            
            result = self.validator.validation_results[-1]
            assert result.name == "ollama_model"
            assert result.status == ValidationStatus.FAIL
            assert f"Model {config.llm.model_name} not found" in result.message
    
    def test_validate_disk_space_sufficient(self):
        """Test disk space validation with sufficient space."""
        with patch('os.statvfs') as mock_statvfs:
            # Mock 20GB free space
            mock_stat = Mock()
            mock_stat.f_bavail = 20 * 1024 * 1024 * 1024 // 4096  # 20GB in blocks
            mock_stat.f_frsize = 4096
            mock_statvfs.return_value = mock_stat
            
            self.validator._validate_disk_space()
            
            result = self.validator.validation_results[-1]
            assert result.name == "disk_space"
            assert result.status == ValidationStatus.PASS
            assert "Sufficient disk space available" in result.message
    
    def test_validate_disk_space_insufficient(self):
        """Test disk space validation with insufficient space."""
        with patch('os.statvfs') as mock_statvfs:
            # Mock 2GB free space
            mock_stat = Mock()
            mock_stat.f_bavail = 2 * 1024 * 1024 * 1024 // 4096  # 2GB in blocks
            mock_stat.f_frsize = 4096
            mock_statvfs.return_value = mock_stat
            
            self.validator._validate_disk_space()
            
            result = self.validator.validation_results[-1]
            assert result.name == "disk_space"
            assert result.status == ValidationStatus.FAIL
            assert "Insufficient disk space" in result.message
    
    def test_validate_temp_directory_accessible(self):
        """Test temp directory validation when accessible."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(config.files, 'temp_dir', Path(temp_dir)):
                self.validator._validate_temp_directory()
                
                result = self.validator.validation_results[-1]
                assert result.name == "temp_directory"
                assert result.status == ValidationStatus.PASS
                assert "Temp directory is accessible" in result.message
    
    def test_validate_temp_directory_not_accessible(self):
        """Test temp directory validation when not accessible."""
        with patch.object(config.files, 'temp_dir', Path("/nonexistent/path")):
            with patch('pathlib.Path.mkdir', side_effect=PermissionError("Access denied")):
                self.validator._validate_temp_directory()
                
                result = self.validator.validation_results[-1]
                assert result.name == "temp_directory"
                assert result.status == ValidationStatus.FAIL
                assert "Cannot access temp directory" in result.message
    
    def test_full_validation_all_pass(self):
        """Test full validation when all checks pass."""
        def mock_validation_methods():
            # Add mock passing results after clearing
            self.validator.validation_results.extend([
                ValidationResult("blackhole_installation", ValidationStatus.PASS, "Pass"),
                ValidationResult("ollama_installation", ValidationStatus.PASS, "Pass"),
                ValidationResult("ollama_model", ValidationStatus.PASS, "Pass"),
            ])
        
        with patch.object(self.validator, '_validate_operating_system', side_effect=mock_validation_methods), \
             patch.object(self.validator, '_validate_python_version'), \
             patch.object(self.validator, '_validate_dependencies'), \
             patch.object(self.validator, '_validate_blackhole_installation'), \
             patch.object(self.validator, '_validate_multi_output_device'), \
             patch.object(self.validator, '_validate_audio_permissions'), \
             patch.object(self.validator, '_validate_ollama_installation'), \
             patch.object(self.validator, '_validate_ollama_model'), \
             patch.object(self.validator, '_validate_disk_space'), \
             patch.object(self.validator, '_validate_temp_directory'):
            
            report = self.validator.run_full_validation()
            
            assert report.overall_status == ValidationStatus.PASS
            assert report.setup_complete
    
    def test_full_validation_with_failures(self):
        """Test full validation when some checks fail."""
        def mock_validation_methods():
            # Add mock results with failures after clearing
            self.validator.validation_results.extend([
                ValidationResult("blackhole_installation", ValidationStatus.FAIL, "Fail", 
                               fix_instructions="Install BlackHole"),
                ValidationResult("ollama_installation", ValidationStatus.PASS, "Pass"),
                ValidationResult("ollama_model", ValidationStatus.PASS, "Pass"),
            ])
        
        with patch.object(self.validator, '_validate_operating_system', side_effect=mock_validation_methods), \
             patch.object(self.validator, '_validate_python_version'), \
             patch.object(self.validator, '_validate_dependencies'), \
             patch.object(self.validator, '_validate_blackhole_installation'), \
             patch.object(self.validator, '_validate_multi_output_device'), \
             patch.object(self.validator, '_validate_audio_permissions'), \
             patch.object(self.validator, '_validate_ollama_installation'), \
             patch.object(self.validator, '_validate_ollama_model'), \
             patch.object(self.validator, '_validate_disk_space'), \
             patch.object(self.validator, '_validate_temp_directory'):
            
            report = self.validator.run_full_validation()
            
            assert report.overall_status == ValidationStatus.FAIL
            assert not report.setup_complete
            assert len(report.next_steps) > 0
            assert "Install BlackHole" in report.next_steps[0]
    
    def test_audio_setup_validation_only(self):
        """Test audio-only validation."""
        with patch.object(self.validator, '_validate_blackhole_installation'), \
             patch.object(self.validator, '_validate_multi_output_device'), \
             patch.object(self.validator, '_validate_audio_permissions'):
            
            self.validator.validation_results = [
                ValidationResult("blackhole_installation", ValidationStatus.PASS, "Pass"),
            ]
            
            report = self.validator.validate_audio_setup_only()
            
            assert isinstance(report, SetupValidationReport)
    
    def test_llm_setup_validation_only(self):
        """Test LLM-only validation."""
        with patch.object(self.validator, '_validate_ollama_installation'), \
             patch.object(self.validator, '_validate_ollama_model'):
            
            self.validator.validation_results = [
                ValidationResult("ollama_installation", ValidationStatus.PASS, "Pass"),
            ]
            
            report = self.validator.validate_llm_setup_only()
            
            assert isinstance(report, SetupValidationReport)
    
    def test_get_setup_wizard_steps(self):
        """Test setup wizard steps generation."""
        steps = self.validator.get_setup_wizard_steps()
        
        assert len(steps) == 6
        assert steps[0]["title"] == "Install BlackHole Audio Driver"
        assert steps[1]["title"] == "Create Multi-Output Device"
        assert steps[2]["title"] == "Grant Audio Permissions"
        assert steps[3]["title"] == "Install Ollama"
        assert steps[4]["title"] == "Download LLM Model"
        assert steps[5]["title"] == "Test Complete Setup"
        
        # Check that all steps have required fields
        for step in steps:
            assert "step" in step
            assert "title" in step
            assert "description" in step
            assert "instructions" in step
            assert "validation_method" in step
            assert "required" in step
    
    def test_setup_instructions_methods(self):
        """Test that all setup instruction methods return strings."""
        instructions = [
            self.validator._get_blackhole_installation_instructions(),
            self.validator._get_multi_output_setup_instructions(),
            self.validator._get_audio_permissions_instructions(),
            self.validator._get_ollama_installation_instructions(),
            self.validator._get_model_download_instructions(),
            self.validator._get_integration_test_instructions()
        ]
        
        for instruction in instructions:
            assert isinstance(instruction, str)
            assert len(instruction) > 0
            assert "Step" in instruction or "#" in instruction  # Should contain structured content


class TestSystemInfo:
    """Test system information gathering."""
    
    def test_get_system_info_structure(self):
        """Test that system info returns expected structure."""
        info = get_system_info()
        
        assert "platform" in info
        assert "python" in info
        
        # Platform info
        platform_info = info["platform"]
        assert "system" in platform_info
        assert "release" in platform_info
        assert "version" in platform_info
        assert "machine" in platform_info
        
        # Python info
        python_info = info["python"]
        assert "version" in python_info
        assert "executable" in python_info
        assert "path" in python_info
    
    @patch('platform.system')
    def test_get_system_info_macos_specific(self, mock_system):
        """Test macOS-specific system info."""
        mock_system.return_value = "Darwin"
        
        with patch('platform.mac_ver', return_value=('12.0', 'dev', 'arm64')):
            info = get_system_info()
            
            assert "mac_version" in info["platform"]
            assert "mac_dev_stage" in info["platform"]
            assert "mac_machine" in info["platform"]
    
    @patch('sounddevice.query_devices')
    def test_get_system_info_audio_devices(self, mock_query_devices):
        """Test audio device info in system info."""
        mock_devices = [
            {'name': 'Built-in Microphone', 'max_input_channels': 1},
            {'name': 'Built-in Output', 'max_output_channels': 2}
        ]
        mock_query_devices.return_value = mock_devices
        
        with patch('sounddevice.query_devices') as mock_query:
            mock_query.side_effect = [
                {'name': 'Built-in Microphone'},  # default input
                {'name': 'Built-in Output'},      # default output
                mock_devices                       # all devices
            ]
            
            info = get_system_info()
            
            assert "audio" in info
            assert "default_input" in info["audio"]
            assert "default_output" in info["audio"]
            assert "device_count" in info["audio"]
    
    @patch('sounddevice.query_devices')
    def test_get_system_info_audio_error(self, mock_query_devices):
        """Test system info when audio query fails."""
        mock_query_devices.side_effect = Exception("Audio error")
        
        info = get_system_info()
        
        assert "audio" in info
        assert "error" in info["audio"]


class TestSetupValidatorIntegration:
    """Integration tests for setup validator."""
    
    def test_validator_with_real_system_info(self):
        """Test validator with real system information (non-mocked)."""
        validator = SetupValidator()
        
        # This should not raise exceptions
        system_info = get_system_info()
        assert isinstance(system_info, dict)
        
        # Basic validation should work
        validator._validate_python_version()
        assert len(validator.validation_results) > 0
        
        # Should be able to generate wizard steps
        steps = validator.get_setup_wizard_steps()
        assert len(steps) == 6
    
    def test_validation_result_serialization(self):
        """Test that validation results can be serialized to JSON."""
        validator = SetupValidator()
        
        # Add some mock results
        validator.validation_results = [
            ValidationResult("test1", ValidationStatus.PASS, "Test 1"),
            ValidationResult("test2", ValidationStatus.FAIL, "Test 2", 
                           details="Details", fix_instructions="Fix it")
        ]
        
        report = validator._generate_report()
        
        # Should be serializable to JSON
        json_str = json.dumps(report.to_dict())
        assert isinstance(json_str, str)
        
        # Should be deserializable
        data = json.loads(json_str)
        assert data["overall_status"] == "fail"
        assert len(data["results"]) == 2


if __name__ == "__main__":
    pytest.main([__file__])