import torch

from src.metrics import (
    attention_entropy,
    mean_attention_by_category,
    mean_attention_by_token_position,
    pairwise_attention_diff,
    sparsity_ratio,
)


def uniform_attention(seq_len: int) -> torch.Tensor:
    val = 1.0 / seq_len
    return torch.full((seq_len, seq_len), val)


def diagonal_attention(seq_len: int) -> torch.Tensor:
    return torch.eye(seq_len)


class TestAttentionEntropy:
    def test_output_shape(self):
        result = attention_entropy(uniform_attention(8))
        assert result.shape == (8,)

    def test_uniform_has_higher_entropy_than_diagonal(self):
        entropy_uniform = attention_entropy(uniform_attention(8))
        entropy_diag = attention_entropy(diagonal_attention(8))
        assert (entropy_uniform > entropy_diag).all()

    def test_diagonal_has_low_entropy(self):
        entropy = attention_entropy(diagonal_attention(8))
        assert (entropy < 0.01).all()

    def test_nonnegative(self):
        attn = torch.rand(6, 6)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        assert (attention_entropy(attn) >= 0).all()

    def test_single_token(self):
        result = attention_entropy(torch.tensor([[1.0]]))
        assert result.shape == (1,)


class TestSparsityRatio:
    def test_all_zeros_is_fully_sparse(self):
        assert sparsity_ratio(torch.zeros(4, 4), threshold=0.01) == 1.0

    def test_all_ones_is_not_sparse(self):
        assert sparsity_ratio(torch.ones(4, 4), threshold=0.01) == 0.0

    def test_half_sparse(self):
        attn = torch.zeros(4, 4)
        attn[:2, :] = 1.0
        result = sparsity_ratio(attn, threshold=0.5)
        assert abs(result - 0.5) < 1e-6

    def test_returns_float(self):
        assert isinstance(sparsity_ratio(torch.rand(3, 3)), float)

    def test_empty_tensor_returns_zero(self):
        assert sparsity_ratio(torch.zeros(0, 0)) == 0.0

    def test_higher_threshold_gives_higher_sparsity(self):
        attn = torch.rand(10, 10) * 0.5
        s1 = sparsity_ratio(attn, threshold=0.1)
        s2 = sparsity_ratio(attn, threshold=0.4)
        assert s2 >= s1


class TestPairwiseAttentionDiff:
    def test_same_shape_exact_diff(self):
        base = torch.ones(4, 4) * 0.3
        mod = torch.ones(4, 4) * 0.5
        diff = pairwise_attention_diff(base, mod)
        assert torch.allclose(diff, torch.full((4, 4), 0.2), atol=1e-6)

    def test_different_shapes_truncates_to_smaller(self):
        base = torch.ones(6, 6) * 0.3
        mod = torch.ones(4, 4) * 0.5
        diff = pairwise_attention_diff(base, mod)
        assert diff.shape == (4, 4)

    def test_zero_diff_for_identical(self):
        attn = torch.rand(5, 5)
        diff = pairwise_attention_diff(attn, attn)
        assert torch.allclose(diff, torch.zeros(5, 5), atol=1e-6)

    def test_output_dtype_is_float32(self):
        base = torch.rand(3, 3).to(torch.bfloat16)
        mod = torch.rand(3, 3).to(torch.bfloat16)
        diff = pairwise_attention_diff(base, mod)
        assert diff.dtype == torch.float32


class TestMeanAttentionByTokenPosition:
    def test_output_shape(self):
        result = mean_attention_by_token_position(torch.rand(8, 8))
        assert result.shape == (8,)

    def test_uniform_gives_constant_mean(self):
        seq_len = 6
        result = mean_attention_by_token_position(uniform_attention(seq_len))
        expected = torch.full((seq_len,), 1.0 / seq_len)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_equals_column_mean(self):
        attn = torch.rand(5, 5)
        result = mean_attention_by_token_position(attn)
        assert torch.allclose(result, attn.float().mean(dim=0), atol=1e-6)


class TestMeanAttentionByCategory:
    def test_output_has_all_category_keys(self):
        attn = uniform_attention(4)
        cats = ["instruction", "content", "functional", "functional"]
        result = mean_attention_by_category(attn, cats)
        assert set(result.keys()) == {"instruction", "content", "functional"}

    def test_empty_category_returns_zero(self):
        attn = uniform_attention(4)
        cats = ["functional", "functional", "functional", "functional"]
        result = mean_attention_by_category(attn, cats)
        assert result["instruction"] == 0.0
        assert result["content"] == 0.0

    def test_all_same_category_equals_overall_mean(self):
        seq_len = 4
        attn = uniform_attention(seq_len)
        cats = ["instruction"] * seq_len
        result = mean_attention_by_category(attn, cats)
        assert abs(result["instruction"] - 1.0 / seq_len) < 1e-6

    def test_values_are_floats(self):
        attn = torch.rand(5, 5)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        cats = ["instruction", "content", "functional", "functional", "instruction"]
        result = mean_attention_by_category(attn, cats)
        for v in result.values():
            assert isinstance(v, float)

    def test_single_token_per_category(self):
        attn = uniform_attention(3)
        cats = ["instruction", "content", "functional"]
        result = mean_attention_by_category(attn, cats)
        expected = 1.0 / 3
        for v in result.values():
            assert abs(v - expected) < 1e-6

    def test_nonnegative_values(self):
        attn = torch.rand(6, 6)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        cats = ["instruction", "content", "functional", "instruction", "content", "functional"]
        result = mean_attention_by_category(attn, cats)
        for v in result.values():
            assert v >= 0.0
