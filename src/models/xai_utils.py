import torch


def mean_pooling_heads(attention_tensor: torch.Tensor) -> torch.Tensor:
    return attention_tensor.mean(dim=1)


def extract_sliding_window_mask(seq_len: int, window_size: int = 1024) -> torch.Tensor:
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
    mask = torch.tril(mask)
    mask = torch.triu(mask, diagonal=-window_size + 1)
    return mask
