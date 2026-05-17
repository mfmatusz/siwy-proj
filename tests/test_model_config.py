import pytest
from omegaconf import OmegaConf

from src.config.model import validate_model_config


def make_cfg(**overrides) -> object:
    data = {
        "model": {
            "num_layers": 34,
            "gqa_group_size": 2,
            "global_layer_indices": [5, 11, 17, 23, 29],
        }
    }
    for dotkey, val in overrides.items():
        parts = dotkey.split(".")
        d = data
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = val
    return OmegaConf.create(data)


class TestValidateModelConfig:
    def test_valid_config_passes(self):
        validate_model_config(make_cfg())

    def test_zero_num_layers_raises(self):
        with pytest.raises(ValueError, match="num_layers"):
            validate_model_config(make_cfg(**{"model.num_layers": 0}))

    def test_negative_num_layers_raises(self):
        with pytest.raises(ValueError):
            validate_model_config(make_cfg(**{"model.num_layers": -1}))

    def test_zero_gqa_group_size_raises(self):
        with pytest.raises(ValueError, match="gqa_group_size"):
            validate_model_config(make_cfg(**{"model.gqa_group_size": 0}))

    def test_empty_global_layer_indices_raises(self):
        with pytest.raises(ValueError, match="global_layer_indices"):
            validate_model_config(make_cfg(**{"model.global_layer_indices": []}))

    def test_index_at_num_layers_raises(self):
        with pytest.raises(ValueError):
            validate_model_config(make_cfg(**{"model.global_layer_indices": [5, 34]}))

    def test_index_exceeding_num_layers_raises(self):
        with pytest.raises(ValueError):
            validate_model_config(make_cfg(**{"model.global_layer_indices": [5, 99]}))

    def test_single_valid_index(self):
        validate_model_config(make_cfg(**{"model.global_layer_indices": [0]}))

    def test_max_valid_index(self):
        validate_model_config(make_cfg(**{"model.global_layer_indices": [33]}))
