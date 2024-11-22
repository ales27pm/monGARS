
import sys
sys.path.insert(0, '/mnt/data/AutonomousAssistantProject/project')  # Add project path to sys.path

from fastapi.testclient import TestClient
from app.api.routes import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

def test_add_memory():
    """Test adding a memory to the database and embeddings index."""
    payload = {"content": "Test Memory", "metadata": "Test Metadata", "created_at": "2024-11-21"}
    response = client.post("/memories/", json=payload)
    assert response.status_code == 200
    assert response.json() == {"message": "Memory added successfully."}
