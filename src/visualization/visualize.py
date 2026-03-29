import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_attention_heatmap(
    attention_matrix: torch.Tensor,
    tokens: list[str],
    save_path: str,
    title: str = "Attention Heatmap",
):
    plt.figure(figsize=(24, 10))

    attn_np = attention_matrix.detach().cpu().to(torch.float32).numpy()

    ax = sns.heatmap(
        attn_np,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "Attention Weight"},
    )

    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
