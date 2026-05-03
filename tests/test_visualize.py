from pathlib import Path

import torch

from src.visualization.visualize import plot_attention_heatmap

SEQ_LEN = 5


def make_attention(seq_len: int = SEQ_LEN) -> torch.Tensor:
    t = torch.rand(seq_len, seq_len)
    return t / t.sum(dim=-1, keepdim=True)


class TestPlotAttentionHeatmap:
    def test_creates_png_file(self, tmp_path: Path):
        attn = make_attention()
        tokens = [f"t{i}" for i in range(SEQ_LEN)]
        save_path = str(tmp_path / "heatmap.png")

        plot_attention_heatmap(attn, tokens, save_path)

        assert Path(save_path).exists()
        assert Path(save_path).stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path: Path):
        attn = make_attention()
        tokens = [f"t{i}" for i in range(SEQ_LEN)]
        save_path = str(tmp_path / "nested" / "deep" / "heatmap.png")

        plot_attention_heatmap(attn, tokens, save_path)

        assert Path(save_path).exists()

    def test_custom_title_accepted(self, tmp_path: Path):
        attn = make_attention()
        tokens = [f"t{i}" for i in range(SEQ_LEN)]
        save_path = str(tmp_path / "titled.png")

        plot_attention_heatmap(attn, tokens, save_path, title="Custom Title XYZ")

        assert Path(save_path).exists()

    def test_single_token(self, tmp_path: Path):
        attn = torch.tensor([[1.0]])
        tokens = ["<bos>"]
        save_path = str(tmp_path / "single.png")

        plot_attention_heatmap(attn, tokens, save_path)

        assert Path(save_path).exists()
