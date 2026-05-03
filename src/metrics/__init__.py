"""Attention weight metrics. Pure functions — torch only, no I/O."""

import torch


def attention_entropy(attention_matrix: torch.Tensor) -> torch.Tensor:
    """Shannon entropy per token (row-wise). Returns tensor of shape (seq_len,).

    H(p) = -sum(p * log(p + eps)), eps=1e-9 prevents log(0).
    Higher entropy = attention more diffuse across tokens.
    """
    eps = 1e-9
    p = attention_matrix.float()
    return -(p * torch.log(p + eps)).sum(dim=-1)


def sparsity_ratio(attention_matrix: torch.Tensor, threshold: float = 0.01) -> float:
    """Fraction of attention weights below threshold. Returns float in [0.0, 1.0]."""
    total = attention_matrix.numel()
    if total == 0:
        return 0.0
    return (attention_matrix < threshold).sum().item() / total


def pairwise_attention_diff(base: torch.Tensor, modified: torch.Tensor) -> torch.Tensor:
    """Elementwise difference: modified - base.

    If shapes are identical, returns the full difference.
    If shapes differ (different tokenization lengths), truncates to the smaller common size.
    """
    if base.shape == modified.shape:
        return modified.float() - base.float()
    r = min(base.shape[0], modified.shape[0])
    c = min(base.shape[1], modified.shape[1])
    return modified.float()[:r, :c] - base.float()[:r, :c]


def mean_attention_by_token_position(attention_matrix: torch.Tensor) -> torch.Tensor:
    """Column-wise mean — how much attention each token position attracts. Returns tensor (seq_len,)."""
    return attention_matrix.float().mean(dim=0)
