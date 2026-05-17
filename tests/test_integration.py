"""Integration tests: full attention analysis pipeline (no GPU or network required)."""

from pathlib import Path

import torch

from src.metrics import attention_entropy, pairwise_attention_diff, sparsity_ratio
from src.models.attention_utils import aggregate_attention_by_type, process_prompt_pair

NUM_HEADS = 8
SEQ_LEN = 6
NUM_LAYERS = 34


def make_attention_tuple(num_layers: int = NUM_LAYERS, num_heads: int = NUM_HEADS, seq_len: int = SEQ_LEN) -> tuple:
    t = torch.rand(num_heads, seq_len, seq_len)
    t = t / t.sum(dim=-1, keepdim=True)
    return tuple(t.unsqueeze(0) for _ in range(num_layers))


class TestFullAnalysisPipeline:
    def test_pipeline_produces_all_expected_outputs(self, tmp_path: Path):
        attentions_base = make_attention_tuple()
        attentions_mod = make_attention_tuple()
        tokens_base = [f"base_{i}" for i in range(SEQ_LEN)]
        tokens_mod = [f"mod_{i}" for i in range(SEQ_LEN)]

        saved_paths = process_prompt_pair(
            "integration_test", attentions_base, attentions_mod, tokens_base, tokens_mod, tmp_path
        )

        local_base, global_base, overall_base = aggregate_attention_by_type(attentions_base)
        _, _, overall_mod = aggregate_attention_by_type(attentions_mod)

        entropy_base = attention_entropy(overall_base)
        sparsity_base = sparsity_ratio(overall_base)
        diff = pairwise_attention_diff(overall_base, overall_mod)
        diff_l1 = diff.abs().mean().item()
        diff_l2 = diff.pow(2).mean().sqrt().item()

        assert len(saved_paths) >= 9
        for path in saved_paths.values():
            assert path.exists()
            assert path.stat().st_size > 0

        tensors_dir = tmp_path / "integration_test" / "tensors"
        for name in ["local_base", "global_base", "overall_base", "local_mod", "global_mod", "overall_mod"]:
            assert (tensors_dir / f"{name}.pt").exists()

        assert entropy_base.shape == (SEQ_LEN,)
        assert isinstance(sparsity_base, float)
        assert 0.0 <= sparsity_base <= 1.0
        assert isinstance(diff_l1, float) and diff_l1 >= 0.0
        assert isinstance(diff_l2, float) and diff_l2 >= 0.0

        assert local_base.shape == (SEQ_LEN, SEQ_LEN)
        assert global_base.shape == (SEQ_LEN, SEQ_LEN)
        assert overall_base.shape == (SEQ_LEN, SEQ_LEN)

    def test_pipeline_handles_different_sequence_lengths(self, tmp_path: Path):
        base_seq = 6
        mod_seq = 9

        def make_attn(seq: int) -> tuple:
            t = torch.rand(NUM_HEADS, seq, seq)
            t = t / t.sum(dim=-1, keepdim=True)
            return tuple(t.unsqueeze(0) for _ in range(NUM_LAYERS))

        attentions_base = make_attn(base_seq)
        attentions_mod = make_attn(mod_seq)
        tokens_base = [f"b{i}" for i in range(base_seq)]
        tokens_mod = [f"m{i}" for i in range(mod_seq)]

        saved_paths = process_prompt_pair(
            "len_mismatch", attentions_base, attentions_mod, tokens_base, tokens_mod, tmp_path
        )

        _, _, overall_base = aggregate_attention_by_type(attentions_base)
        _, _, overall_mod = aggregate_attention_by_type(attentions_mod)
        diff = pairwise_attention_diff(overall_base, overall_mod)

        assert diff.shape == (base_seq, base_seq)
        assert len(saved_paths) >= 9

    def test_identical_inputs_give_zero_diff(self, tmp_path: Path):
        attentions = make_attention_tuple()
        tokens = [f"tok_{i}" for i in range(SEQ_LEN)]

        process_prompt_pair("identical", attentions, attentions, tokens, tokens, tmp_path)

        _, _, overall = aggregate_attention_by_type(attentions)
        diff = pairwise_attention_diff(overall, overall)

        assert torch.allclose(diff, torch.zeros(SEQ_LEN, SEQ_LEN), atol=1e-6)

    def test_entropy_and_sparsity_across_multiple_prompts(self, tmp_path: Path):
        for i in range(3):
            attentions = make_attention_tuple()
            tokens = [f"t{j}" for j in range(SEQ_LEN)]
            process_prompt_pair(f"prompt_{i}", attentions, attentions, tokens, tokens, tmp_path)

            _, _, overall = aggregate_attention_by_type(attentions)
            entropy = attention_entropy(overall)
            sparsity = sparsity_ratio(overall)

            assert (entropy >= 0).all()
            assert 0.0 <= sparsity <= 1.0
