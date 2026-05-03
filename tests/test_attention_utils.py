from pathlib import Path

import torch

from src.models.attention_utils import (
    aggregate_attention_by_type,
    gqa_aware_head_pooling,
    process_prompt_pair,
)

SEQ_LEN = 6
NUM_HEADS = 8
GROUP_SIZE = 2


def make_fake_attention(num_heads: int = NUM_HEADS, seq_len: int = SEQ_LEN) -> torch.Tensor:
    """Returns (num_heads, seq_len, seq_len) with rows summing to 1."""
    t = torch.rand(num_heads, seq_len, seq_len)
    return t / t.sum(dim=-1, keepdim=True)


def make_fake_attentions_tuple(num_layers: int = 34, num_heads: int = NUM_HEADS, seq_len: int = SEQ_LEN) -> tuple:
    """Returns tuple of num_layers tensors, each (1, num_heads, seq_len, seq_len)."""
    return tuple(make_fake_attention(num_heads, seq_len).unsqueeze(0) for _ in range(num_layers))


class TestGqaAwareHeadPooling:
    def test_output_shape(self):
        attn = make_fake_attention(NUM_HEADS, SEQ_LEN)
        result = gqa_aware_head_pooling(attn, group_size=GROUP_SIZE)
        assert result.shape == (SEQ_LEN, SEQ_LEN)

    def test_values_in_unit_range(self):
        attn = make_fake_attention(NUM_HEADS, SEQ_LEN)
        result = gqa_aware_head_pooling(attn, group_size=GROUP_SIZE)
        assert result.min().item() >= 0.0
        assert result.max().item() <= 1.0 + 1e-6

    def test_group_size_1_equals_head_mean(self):
        attn = make_fake_attention(4, SEQ_LEN)
        result = gqa_aware_head_pooling(attn, group_size=1)
        expected = attn.mean(dim=0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_different_seq_lengths(self):
        for seq_len in [1, 4, 16]:
            attn = make_fake_attention(NUM_HEADS, seq_len)
            result = gqa_aware_head_pooling(attn, group_size=GROUP_SIZE)
            assert result.shape == (seq_len, seq_len)


class TestAggregateAttentionByType:
    def test_returns_three_tensors(self):
        attentions = make_fake_attentions_tuple()
        result = aggregate_attention_by_type(attentions)
        assert len(result) == 3

    def test_output_shapes(self):
        attentions = make_fake_attentions_tuple()
        local_mean, global_mean, overall_mean = aggregate_attention_by_type(attentions)
        assert local_mean.shape == (SEQ_LEN, SEQ_LEN)
        assert global_mean.shape == (SEQ_LEN, SEQ_LEN)
        assert overall_mean.shape == (SEQ_LEN, SEQ_LEN)

    def test_overall_between_local_and_global(self):
        attentions = make_fake_attentions_tuple()
        local, global_, overall = aggregate_attention_by_type(attentions)
        lower = torch.minimum(local, global_)
        upper = torch.maximum(local, global_)
        assert (overall >= lower - 1e-6).all()
        assert (overall <= upper + 1e-6).all()


class TestProcessPromptPair:
    def test_saves_tensor_files(self, tmp_path: Path):
        attentions = make_fake_attentions_tuple()
        tokens = [f"tok{i}" for i in range(SEQ_LEN)]

        process_prompt_pair("test_pair", attentions, attentions, tokens, tokens, tmp_path)

        tensors_dir = tmp_path / "test_pair" / "tensors"
        for name in ["local_base", "global_base", "overall_base", "local_mod", "global_mod", "overall_mod"]:
            assert (tensors_dir / f"{name}.pt").exists(), f"Missing {name}.pt"

    def test_saves_heatmap_files(self, tmp_path: Path):
        attentions = make_fake_attentions_tuple()
        tokens = [f"tok{i}" for i in range(SEQ_LEN)]

        process_prompt_pair("test_pair", attentions, attentions, tokens, tokens, tmp_path)

        heatmaps_dir = tmp_path / "test_pair" / "heatmaps"
        png_files = list(heatmaps_dir.glob("*.png"))
        assert len(png_files) >= 6

    def test_returns_dict_of_paths(self, tmp_path: Path):
        attentions = make_fake_attentions_tuple()
        tokens = [f"tok{i}" for i in range(SEQ_LEN)]

        result = process_prompt_pair("p", attentions, attentions, tokens, tokens, tmp_path)

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, Path)
