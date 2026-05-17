import json
from pathlib import Path


def load_token_categories(file_path: Path | str) -> dict[str, dict]:
    """Load manual token category annotations from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Token categories file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def categorize_tokens(
    tokens: list[str],
    instruction_keywords: list[str],
    content_keywords: list[str],
) -> list[str]:
    """Assign each token to 'instruction', 'content', or 'functional'.

    Strips the SentencePiece word-boundary prefix (▁) before matching.
    Matching is case-insensitive. Tokens not matching any keyword are 'functional'.
    """
    instruction_set = {kw.lower() for kw in instruction_keywords}
    content_set = {kw.lower() for kw in content_keywords}
    result = []
    for token in tokens:
        clean = token.lstrip("▁").lower()
        if clean in instruction_set:
            result.append("instruction")
        elif clean in content_set:
            result.append("content")
        else:
            result.append("functional")
    return result
