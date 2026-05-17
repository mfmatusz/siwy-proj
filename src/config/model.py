import os

from omegaconf import OmegaConf

# https://huggingface.co/docs/transformers/model_doc/gemma3
try:
    _config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conf", "config.yaml"))
    _app_config = OmegaConf.load(_config_path)
    MODEL_ID = _app_config.model.name
    GLOBAL_LAYER_INDICES: set[int] = set(_app_config.model.global_layer_indices)
    NUM_LAYERS: int = int(_app_config.model.num_layers)
    GQA_GROUP_SIZE: int = int(_app_config.model.gqa_group_size)
except Exception as e:
    print(f"Warning: Could not read config.yaml. Using defaults. ({e})")
    MODEL_ID = "google/gemma-3-4b-it"
    GLOBAL_LAYER_INDICES = {5, 11, 17, 23, 29}
    NUM_LAYERS = 34
    GQA_GROUP_SIZE = 2


def validate_model_config(cfg) -> None:
    """Validates model constants from Hydra config at startup. Raises ValueError on invalid values."""
    num_layers = int(cfg.model.num_layers)
    gqa_group_size = int(cfg.model.gqa_group_size)
    global_layer_indices = list(cfg.model.global_layer_indices)

    if num_layers <= 0:
        raise ValueError(f"model.num_layers must be positive, got {num_layers}")
    if gqa_group_size <= 0:
        raise ValueError(f"model.gqa_group_size must be positive, got {gqa_group_size}")
    if not global_layer_indices:
        raise ValueError("model.global_layer_indices must not be empty")
    if max(global_layer_indices) >= num_layers:
        raise ValueError(
            f"All global_layer_indices must be < num_layers ({num_layers}), "
            f"but got max index {max(global_layer_indices)}"
        )


def initialize_from_pretrained(model_id: str | None = None) -> None:
    """Optionally overrides module constants by querying the HF API.

    Call once in main() before the processing loop — NOT at import time.
    On failure, constants remain at their default values.
    """
    global GLOBAL_LAYER_INDICES, NUM_LAYERS, GQA_GROUP_SIZE
    from transformers import AutoConfig

    target = model_id or MODEL_ID
    try:
        config = AutoConfig.from_pretrained(target)
        text_config = getattr(config, "text_config", config)
        layer_types = getattr(text_config, "layer_types", [])
        if layer_types:
            GLOBAL_LAYER_INDICES = {
                i for i, ltype in enumerate(layer_types) if ltype in ("full_attention", "global_attention", "full")
            }
        NUM_LAYERS = getattr(text_config, "num_hidden_layers", NUM_LAYERS)
        num_heads = getattr(text_config, "num_attention_heads", 8)
        num_kv = getattr(text_config, "num_key_value_heads", 4)
        GQA_GROUP_SIZE = num_heads // num_kv
    except Exception as e:
        print(f"Warning: Could not load config dynamically. Using defaults. ({e})")
