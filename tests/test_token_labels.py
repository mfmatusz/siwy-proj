import json
from pathlib import Path

import pytest

from src.data.token_labels import categorize_tokens, load_token_categories


class TestLoadTokenCategories:
    def test_loads_valid_file(self, tmp_path: Path):
        data = {"p1": {"instruction_keywords": ["explain"], "content_keywords": ["gravity"]}}
        p = tmp_path / "cats.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        result = load_token_categories(p)

        assert result == data

    def test_accepts_string_path(self, tmp_path: Path):
        data = {"p1": {"instruction_keywords": [], "content_keywords": []}}
        p = tmp_path / "cats.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        result = load_token_categories(str(p))

        assert "p1" in result

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_token_categories(tmp_path / "missing.json")


class TestCategorizeTokens:
    def test_basic_instruction_and_content(self):
        tokens = ["▁explain", "▁gravity"]
        result = categorize_tokens(tokens, ["explain"], ["gravity"])
        assert result == ["instruction", "content"]

    def test_strips_sentencepiece_prefix(self):
        tokens = ["▁wyjaśnij"]
        result = categorize_tokens(tokens, ["wyjaśnij"], [])
        assert result[0] == "instruction"

    def test_case_insensitive_match(self):
        tokens = ["▁Wyjaśnij", "▁GRAWITACJĘ"]
        result = categorize_tokens(tokens, ["wyjaśnij"], ["grawitację"])
        assert result == ["instruction", "content"]

    def test_unknown_token_is_functional(self):
        tokens = ["<bos>", "▁i", "▁to"]
        result = categorize_tokens(tokens, ["wyjaśnij"], ["grawitację"])
        assert all(c == "functional" for c in result)

    def test_empty_token_list(self):
        result = categorize_tokens([], ["wyjaśnij"], ["grawitację"])
        assert result == []

    def test_output_length_matches_input(self):
        tokens = ["▁a", "▁b", "▁c", "▁d"]
        result = categorize_tokens(tokens, ["a"], ["b"])
        assert len(result) == len(tokens)

    def test_empty_keywords_all_functional(self):
        tokens = ["▁word1", "▁word2"]
        result = categorize_tokens(tokens, [], [])
        assert result == ["functional", "functional"]

    def test_instruction_takes_priority_over_content(self):
        tokens = ["▁overlap"]
        result = categorize_tokens(tokens, ["overlap"], ["overlap"])
        assert result == ["instruction"]
