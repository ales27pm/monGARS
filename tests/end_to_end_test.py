
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_health_check():
    response = requests.get(f"{BASE_URL}/health/")
    if response.status_code == 200 and response.json() == {"status": "OK"}:
        print("Health check passed.")
    else:
        print("Health check failed.")

def test_add_memory():
    payload = {"content": "Test Memory", "metadata": "Test Metadata", "created_at": "2024-11-21"}
    response = requests.post(f"{BASE_URL}/memories/", json=payload)
    if response.status_code == 200:
        print("Memory addition passed.")
    else:
        print("Memory addition failed.")

def test_retrieve_memory():
    response = requests.get(f"{BASE_URL}/memories/1")
    if response.status_code == 200 and "memory" in response.json():
        print("Memory retrieval passed.")
    else:
        print("Memory retrieval failed.")

if __name__ == "__main__":
    test_health_check()
    test_add_memory()
    test_retrieve_memory()
