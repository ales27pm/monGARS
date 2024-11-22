
import sys
sys.path.insert(0, '/mnt/data/AutonomousAssistantProject/project')  # Add project path to sys.path for testing

try:
    from app.api.routes import app
    print("Import successful: app.api.routes")
except ModuleNotFoundError as e:
    print(f"Import error: {e}")
