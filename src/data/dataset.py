import json
from pathlib import Path
from typing import List, Dict

def load_prompts(file_path: str) -> List[Dict[str, str]]:
    """
    Loads a dataset of prompts from a JSON file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts
