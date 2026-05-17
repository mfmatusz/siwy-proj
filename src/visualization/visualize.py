import matplotlib

matplotlib.use("Agg")

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def plot_category_attention_barchart(
    records: list[dict],
    save_path: str,
    title: str = "Mean Attention by Token Category and Prompt Type",
) -> None:
    """Grouped bar chart: mean attention per token category across prompt categories.

    Two side-by-side subplots (instruction tokens, content tokens), each showing
    base vs modified bars for every prompt category on the X axis.

    Args:
        records: list of dicts with keys: prompt_category (str),
            condition ("base" | "modified"), instruction (float),
            content (float), functional (float).
        save_path: output PNG file path.
        title: chart title.
    """
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        pc = rec["prompt_category"]
        cond = rec["condition"]
        agg[pc][f"{cond}_instruction"].append(rec.get("instruction", 0.0))
        agg[pc][f"{cond}_content"].append(rec.get("content", 0.0))

    prompt_cats = sorted(agg.keys())
    x = np.arange(len(prompt_cats))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, token_cat in enumerate(("instruction", "content")):
        ax = axes[ax_idx]
        base_vals = [float(np.mean(agg[pc].get(f"base_{token_cat}", [0.0]))) for pc in prompt_cats]
        mod_vals = [float(np.mean(agg[pc].get(f"modified_{token_cat}", [0.0]))) for pc in prompt_cats]

        ax.bar(x - width / 2, base_vals, width, label="Base", color="steelblue", alpha=0.8)
        ax.bar(x + width / 2, mod_vals, width, label="Modified", color="darkorange", alpha=0.8)

        ax.set_xlabel("Prompt Category")
        ax.set_ylabel("Mean Attention Weight")
        ax.set_title(f"{token_cat.capitalize()} Tokens")
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_cats, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, None)

    fig.suptitle(title)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
