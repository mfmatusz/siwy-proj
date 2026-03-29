import torch

def mean_pooling_heads(attention_tensor: torch.Tensor) -> torch.Tensor:
    """
    Aggregates model heads using the mean, resolving a tensor of shape:
    (batch_size, sequence_length, sequence_length)
    
    Useful especially with GQA architecture or for simpler interpretation.
    """
    # attention_tensor has shape: (batch, num_heads, seq_len, seq_len)
    return attention_tensor.mean(dim=1)

def extract_sliding_window_mask(seq_len: int, window_size: int = 1024) -> torch.Tensor:
    """
    Returns a (seq_len x seq_len) mask for sliding window local attention, 
    important for models based on such layer setups.
    """
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
    mask = torch.tril(mask) # causal mask
    mask = torch.triu(mask, diagonal=-window_size+1)
    return mask

