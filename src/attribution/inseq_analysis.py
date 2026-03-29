from pathlib import Path

import inseq
import torch
from loguru import logger
from transformers import AutoConfig


def load_inseq_model(model_name: str, attribution_method: str = "attention") -> inseq.AttributionModel:
    config = AutoConfig.from_pretrained(model_name)
    eos = config.eos_token_id
    eos_single = eos[0] if isinstance(eos, list) else eos

    if config.pad_token_id is None:
        config.pad_token_id = eos_single
    if config.bos_token_id is None:
        config.bos_token_id = 2

    return inseq.load_model(
        model_name,
        attribution_method,
        model_kwargs={"config": config, "torch_dtype": torch.bfloat16},
    )


def run_attribution(
    model: inseq.AttributionModel,
    prompt: str,
    generation_kwargs: dict | None = None,
) -> inseq.FeatureAttributionOutput:
    if generation_kwargs is None:
        generation_kwargs = {"max_new_tokens": 1}

    model.tokenizer.add_bos_token = False
    return model.attribute(prompt, generation_kwargs=generation_kwargs)


def save_attribution(output: inseq.FeatureAttributionOutput, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(str(path), overwrite=True)
    logger.info(f"Attribution saved to {path}")
