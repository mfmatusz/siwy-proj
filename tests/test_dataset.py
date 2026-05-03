import json

import pytest

from src.data.dataset import load_prompts


def test_load_prompts_valid(tmp_path):
    data = [{"id": "test_01", "base_prompt": "A", "modified_prompt": "B"}]
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    result = load_prompts(p)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["id"] == "test_01"
    assert result[0]["base_prompt"] == "A"
    assert result[0]["modified_prompt"] == "B"


def test_load_prompts_multiple_entries(tmp_path):
    data = [{"id": f"p_{i}", "base_prompt": f"B{i}", "modified_prompt": f"M{i}"} for i in range(5)]
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    result = load_prompts(p)

    assert len(result) == 5
    assert result[4]["id"] == "p_4"


def test_load_prompts_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prompts(tmp_path / "nonexistent.json")


def test_load_prompts_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not valid json }", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_prompts(p)


def test_load_prompts_accepts_string_path(tmp_path):
    data = [{"id": "x", "base_prompt": "a", "modified_prompt": "b"}]
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    result = load_prompts(str(p))

    assert len(result) == 1
