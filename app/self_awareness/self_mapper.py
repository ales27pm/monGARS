
import os
import importlib.util

class SelfMapper:
    """Analyzes the file structure and dependencies within the project."""

    def __init__(self, base_path):
        self.base_path = base_path

    def map_files(self):
        """Scan for Python files and map their paths."""
        file_map = {}
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.py'):
                    relative_path = os.path.relpath(os.path.join(root, file), self.base_path)
                    file_map[file] = relative_path
        return file_map

    def analyze_dependencies(self, file_path):
        """Analyze imports in a Python file to determine dependencies."""
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            imports = [line.strip() for line in lines if line.startswith('import') or line.startswith('from')]
            return imports
        except Exception as e:
            return [f"Error reading file {file_path}: {e}"]

if __name__ == "__main__":
    # Example usage
    mapper = SelfMapper(base_path='.')
    file_map = mapper.map_files()
    for file, path in file_map.items():
        print(f"File: {file}, Path: {path}")
        dependencies = mapper.analyze_dependencies(path)
        print("Dependencies:", dependencies)
