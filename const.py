from pathlib import Path

def get_project_root():
    current_path = Path.cwd()
    while True:
        if (current_path / '.git').exists() or \
                (current_path / '.project_root').exists() or \
                (current_path / '.gitignore').exists() or \
                (current_path / '.idea').exists():
            return current_path
        parent_path = current_path.parent
        if parent_path == current_path:
            raise Exception("Project root not found.")
        current_path = parent_path
PROJECT_ROOT = get_project_root()