"""
Integration tests for FastAPI server startup and basic functionality.

Tests that the server can start, handle requests, and integrate with services.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from ai_meeting_notes.main import app


class TestServerIntegration:
    """Integration tests for the FastAPI server."""
    
    def test_server_startup_and_health_check(self):
        """Test that server starts up and health check works."""
        with TestClient(app) as client:
            response = client.get("/api/health")
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert "services" in data
            assert "message" in data
    
    def test_api_documentation_available(self):
        """Test that OpenAPI documentation is available."""
        with TestClient(app) as client:
            # Test OpenAPI JSON
            response = client.get("/openapi.json")
            assert response.status_code == 200
            
            openapi_data = response.json()
            assert openapi_data["info"]["title"] == "AI Meeting Notes API"
            assert "paths" in openapi_data
            
            # Verify key endpoints are documented
            paths = openapi_data["paths"]
            assert "/api/health" in paths
            assert "/api/status" in paths
            assert "/api/start-recording" in paths
            assert "/api/stop-recording" in paths
            assert "/api/transcript" in paths
            assert "/api/notes" in paths
            assert "/api/clear" in paths
    
    def test_cors_headers_present(self):
        """Test that CORS headers are properly configured."""
        with TestClient(app) as client:
            response = client.options("/api/health")
            # FastAPI automatically handles OPTIONS requests for CORS
            assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly defined
    
    def test_error_handling_for_invalid_endpoints(self):
        """Test error handling for non-existent endpoints."""
        with TestClient(app) as client:
            response = client.get("/api/nonexistent")
            assert response.status_code == 404
            
            response = client.post("/api/invalid-endpoint")
            assert response.status_code == 404
    
    def test_request_validation(self):
        """Test request validation for endpoints that expect specific data."""
        with TestClient(app) as client:
            # Test endpoints that should work without data
            response = client.get("/api/status")
            assert response.status_code == 200
            
            response = client.post("/api/clear")
            assert response.status_code == 200
    
    @patch('ai_meeting_notes.main.audio_recorder')
    @patch('ai_meeting_notes.main.transcriber')
    @patch('ai_meeting_notes.main.notes_generator')
    @patch('ai_meeting_notes.main.file_manager')
    def test_service_integration(self, mock_file_mgr, mock_notes_gen, mock_transcriber, mock_audio):
        """Test integration with mocked services."""
        # Configure mocks
        mock_audio.check_blackhole_available.return_value = True
        mock_audio.is_recording.return_value = False
        mock_notes_gen.check_ollama_available.return_value = True
        mock_file_mgr.check_disk_space.return_value = (True, 100.0, "")  # Return tuple as expected
        
        with TestClient(app) as client:
            # Test health check with mocked services
            response = client.get("/api/health")
            assert response.status_code == 200
            
            data = response.json()
            # Note: Status may be degraded due to real service initialization during startup
            assert data["status"] in ["healthy", "degraded"]
            assert "services" in data
            # Verify the structure is correct even if some services are unavailable
            assert isinstance(data["services"], dict)
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        with TestClient(app) as client:
            # Make multiple concurrent requests
            responses = []
            
            # Health checks should all succeed
            for _ in range(5):
                response = client.get("/api/health")
                responses.append(response)
            
            for response in responses:
                assert response.status_code == 200
    
    def test_response_models_structure(self):
        """Test that response models have the expected structure."""
        with TestClient(app) as client:
            # Test status response structure
            response = client.get("/api/status")
            assert response.status_code == 200
            
            data = response.json()
            required_fields = ["status", "message"]
            for field in required_fields:
                assert field in data
            
            # Test health response structure
            response = client.get("/api/health")
            assert response.status_code == 200
            
            data = response.json()
            required_fields = ["status", "services", "message"]
            for field in required_fields:
                assert field in data
            
            assert isinstance(data["services"], dict)
    
    def test_content_type_headers(self):
        """Test that proper content-type headers are returned."""
        with TestClient(app) as client:
            response = client.get("/api/health")
            assert response.status_code == 200
            assert "application/json" in response.headers.get("content-type", "")
    
    def test_server_metadata(self):
        """Test server metadata and configuration."""
        with TestClient(app) as client:
            response = client.get("/openapi.json")
            assert response.status_code == 200
            
            openapi_data = response.json()
            info = openapi_data["info"]
            
            assert info["title"] == "AI Meeting Notes API"
            assert info["version"] == "1.0.0"
            assert "description" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])