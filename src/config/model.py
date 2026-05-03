import os

from omegaconf import OmegaConf

# https://huggingface.co/docs/transformers/model_doc/gemma3
try:
    _config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conf", "config.yaml"))
    _app_config = OmegaConf.load(_config_path)
    MODEL_ID = _app_config.model.name
except Exception as e:
    print(f"Warning: Could not read config.yaml. Using default MODEL_ID. ({e})")
    MODEL_ID = "google/gemma-3-4b-it"

# Hardcoded defaults for Gemma 3 4B IT — no network call required
GLOBAL_LAYER_INDICES: set[int] = {5, 11, 17, 23, 29}
NUM_LAYERS: int = 34
GQA_GROUP_SIZE: int = 2


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
