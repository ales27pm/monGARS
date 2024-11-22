
import sys
sys.path.insert(0, '/mnt/data/AutonomousAssistantProject/project')  # Add project path to sys.path

from app.self_awareness.self_mapper import SelfMapper
from app.self_awareness.activity_logger import ActivityLogger
from app.self_awareness.self_optimizer import SelfOptimizer

# Test SelfMapper
def test_self_mapper():
    mapper = SelfMapper(base_path=".")
    file_map = mapper.map_files()
    assert len(file_map) > 0
    for file, path in file_map.items():
        assert file.endswith(".py")
        dependencies = mapper.analyze_dependencies(path)
        assert isinstance(dependencies, list)

# Test ActivityLogger
def test_activity_logger():
    logger = ActivityLogger(log_file="test_activity.log")
    logger.log_info("Testing info log.")
    logger.log_error("Testing error log.")
    logger.log_activity("TEST", "Testing activity logging.")

# Test SelfOptimizer
def test_self_optimizer():
    optimizer = SelfOptimizer()
    file_path = "test_example.py"  # Replace with actual file path for testing
    with open(file_path, "w") as file:
        file.write("import os\nimport sys\n")
    structure = optimizer.analyze_file(file_path)
    assert "Module" in structure
    optimized_imports = optimizer.optimize_imports(file_path)
    assert len(optimized_imports) > 0
