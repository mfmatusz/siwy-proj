from pathlib import Path

import torch

from src.visualization.visualize import plot_attention_heatmap, plot_category_attention_barchart

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


_SAMPLE_RECORDS = [
    {"prompt_category": "style", "condition": "base", "instruction": 0.30, "content": 0.50, "functional": 0.20},
    {"prompt_category": "style", "condition": "modified", "instruction": 0.40, "content": 0.45, "functional": 0.15},
    {"prompt_category": "tone", "condition": "base", "instruction": 0.25, "content": 0.55, "functional": 0.20},
    {"prompt_category": "tone", "condition": "modified", "instruction": 0.35, "content": 0.50, "functional": 0.15},
    {"prompt_category": "formality", "condition": "base", "instruction": 0.20, "content": 0.60, "functional": 0.20},
    {"prompt_category": "formality", "condition": "modified", "instruction": 0.30, "content": 0.55, "functional": 0.15},
]


class TestPlotCategoryAttentionBarchart:
    def test_creates_png_file(self, tmp_path: Path):
        save_path = str(tmp_path / "barchart.png")
        plot_category_attention_barchart(_SAMPLE_RECORDS, save_path)
        assert Path(save_path).exists()
        assert Path(save_path).stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path: Path):
        save_path = str(tmp_path / "deep" / "nested" / "chart.png")
        plot_category_attention_barchart(_SAMPLE_RECORDS, save_path)
        assert Path(save_path).exists()

    def test_single_category(self, tmp_path: Path):
        records = [
            {"prompt_category": "style", "condition": "base", "instruction": 0.3, "content": 0.5, "functional": 0.2},
        ]
        save_path = str(tmp_path / "single_cat.png")
        plot_category_attention_barchart(records, save_path)
        assert Path(save_path).exists()

    def test_custom_title_accepted(self, tmp_path: Path):
        save_path = str(tmp_path / "titled.png")
        plot_category_attention_barchart(_SAMPLE_RECORDS, save_path, title="Custom Chart Title")
        assert Path(save_path).exists()
