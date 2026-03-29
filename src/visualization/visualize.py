import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for environments without display (e.g., servers, CI/CD pipelines)

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from typing import List

def plot_attention_heatmap(
    attention_matrix: torch.Tensor, 
    tokens: List[str], 
    save_path: str, 
    title: str = "Attention Heatmap"
):
    """
    Plots and saves an attention heatmap for a given 2D attention matrix.
    
    Args:
        attention_matrix: A 2D tensor of shape (seq_len, seq_len) containing attention weights.
        tokens: A list of string tokens corresponding to the sequence.
        save_path: The file path where the plot should be saved.
        title: The title of the plot.
    """
    plt.figure(figsize=(24, 10))
    
    # Detach from GPU, convert to float32/fp16 numpy array for plotting
    attn_np = attention_matrix.detach().cpu().to(torch.float32).numpy()
    
    ax = sns.heatmap(
        attn_np, 
        xticklabels=tokens, 
        yticklabels=tokens,
        cmap="viridis",
        square=True,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
