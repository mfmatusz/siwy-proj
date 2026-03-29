import json
from pathlib import Path


def load_prompts(file_path: Path | str) -> list[dict[str, str]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
