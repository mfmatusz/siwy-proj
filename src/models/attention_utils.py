from pathlib import Path

import torch

from src.config.model import GLOBAL_LAYER_INDICES, GQA_GROUP_SIZE, NUM_LAYERS
from src.visualization.visualize import plot_attention_heatmap


def gqa_aware_head_pooling(attention: torch.Tensor, group_size: int = GQA_GROUP_SIZE) -> torch.Tensor:
    num_heads = attention.shape[0]
    num_groups = num_heads // group_size
    grouped = attention.view(num_groups, group_size, *attention.shape[1:])
    return grouped.mean(dim=1).mean(dim=0)


def mean_pooling_heads(attention_tensor: torch.Tensor) -> torch.Tensor:
    return attention_tensor.mean(dim=1)


def extract_sliding_window_mask(seq_len: int, window_size: int = 1024) -> torch.Tensor:
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
    mask = torch.tril(mask)
    mask = torch.triu(mask, diagonal=-window_size + 1)
    return mask


def aggregate_attention_by_type(
    attentions: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_layers = [
        gqa_aware_head_pooling(attentions[i][0]) for i in range(NUM_LAYERS) if i not in GLOBAL_LAYER_INDICES
    ]
    global_layers = [gqa_aware_head_pooling(attentions[i][0]) for i in range(NUM_LAYERS) if i in GLOBAL_LAYER_INDICES]
    local_mean = torch.stack(local_layers).mean(dim=0)
    global_mean = torch.stack(global_layers).mean(dim=0)
    overall_mean = torch.stack(local_layers + global_layers).mean(dim=0)
    return local_mean, global_mean, overall_mean


def process_prompt_pair(
    prompt_id: str,
    attrs_base: tuple[torch.Tensor, ...],
    attrs_mod: tuple[torch.Tensor, ...],
    tokens_base: list[str],
    tokens_mod: list[str],
    experiment_dir: Path,
) -> dict[str, Path]:
    tensors_dir = experiment_dir / prompt_id / "tensors"
    heatmaps_dir = experiment_dir / prompt_id / "heatmaps"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    per_layer_base = {
        i: {
            "attention": gqa_aware_head_pooling(attrs_base[i][0]),
            "type": "global" if i in GLOBAL_LAYER_INDICES else "local",
        }
        for i in range(NUM_LAYERS)
    }
    per_layer_mod = {
        i: {
            "attention": gqa_aware_head_pooling(attrs_mod[i][0]),
            "type": "global" if i in GLOBAL_LAYER_INDICES else "local",
        }
        for i in range(NUM_LAYERS)
    }

    torch.save(per_layer_base, tensors_dir / "per_layer_base.pt")
    torch.save(per_layer_mod, tensors_dir / "per_layer_mod.pt")

    local_base, global_base, overall_base = aggregate_attention_by_type(attrs_base)
    local_mod, global_mod, overall_mod = aggregate_attention_by_type(attrs_mod)

    torch.save(local_base, tensors_dir / "local_base.pt")
    torch.save(global_base, tensors_dir / "global_base.pt")
    torch.save(overall_base, tensors_dir / "overall_base.pt")
    torch.save(local_mod, tensors_dir / "local_mod.pt")
    torch.save(global_mod, tensors_dir / "global_mod.pt")
    torch.save(overall_mod, tensors_dir / "overall_mod.pt")

    heatmap_configs = [
        (local_base, tokens_base, "local_base.png", f"Base Local: {prompt_id}"),
        (global_base, tokens_base, "global_base.png", f"Base Global: {prompt_id}"),
        (overall_base, tokens_base, "overall_base.png", f"Base Overall: {prompt_id}"),
        (local_mod, tokens_mod, "local_mod.png", f"Modified Local: {prompt_id}"),
        (global_mod, tokens_mod, "global_mod.png", f"Modified Global: {prompt_id}"),
        (overall_mod, tokens_mod, "overall_mod.png", f"Modified Overall: {prompt_id}"),
    ]

    saved_paths: dict[str, Path] = {}
    for attention, tokens, filename, title in heatmap_configs:
        path = heatmaps_dir / filename
        plot_attention_heatmap(attention, tokens, str(path), title=title)
        key = f"heatmaps/{prompt_id}/{filename.replace('.png', '')}"
        saved_paths[key] = path

    return saved_paths
