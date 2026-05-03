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


def plot_diff_heatmap(
    diff_matrix: torch.Tensor,
    tokens_base: list[str],
    tokens_mod: list[str],
    save_path: str,
    title: str = "Attention Diff (Modified − Base)",
):
    diff_np = diff_matrix.detach().cpu().to(torch.float32).numpy()
    max_abs = max(abs(diff_np.min()), abs(diff_np.max()))
    if max_abs == 0:
        max_abs = 1.0
    n_rows, n_cols = diff_np.shape

    plt.figure(figsize=(24, 10))
    ax = sns.heatmap(
        diff_np,
        xticklabels=tokens_mod[:n_cols],
        yticklabels=tokens_base[:n_rows],
        cmap="RdBu_r",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        square=True,
        cbar_kws={"label": "Δ Attention Weight"},
    )
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
