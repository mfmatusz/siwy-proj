import os
from transformers import AutoConfig
from omegaconf import OmegaConf

# https://huggingface.co/docs/transformers/model_doc/gemma3
try:
    _config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conf", "config.yaml"))
    _app_config = OmegaConf.load(_config_path)
    MODEL_ID = _app_config.model.name
except Exception as e:
    print(f"Warning: Could not read config.yaml. Using default MODEL_ID. ({e})")
    MODEL_ID = "google/gemma-3-4b-it"

try:
    _config = AutoConfig.from_pretrained(MODEL_ID)
    _text_config = getattr(_config, "text_config", _config)
    
    _layer_types = getattr(_text_config, "layer_types", [])
    if _layer_types:
        GLOBAL_LAYER_INDICES = {i for i, ltype in enumerate(_layer_types) if ltype in ("full_attention", "global_attention", "full")}
    else:
        GLOBAL_LAYER_INDICES = {5, 11, 17, 23, 29}

    NUM_LAYERS = getattr(_text_config, "num_hidden_layers", 34)
    
    _num_attention_heads = getattr(_text_config, "num_attention_heads", 8)
    _num_key_value_heads = getattr(_text_config, "num_key_value_heads", 4)
    GQA_GROUP_SIZE = _num_attention_heads // _num_key_value_heads

except Exception as e:
    print(f"Warning: Could not load config dynamically. Using defaults. ({e})")
    GLOBAL_LAYER_INDICES = {5, 11, 17, 23, 29}
    NUM_LAYERS = 34
    GQA_GROUP_SIZE = 2
