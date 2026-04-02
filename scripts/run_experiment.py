from pathlib import Path

import dotenv
import hydra

dotenv.load_dotenv()
import torch
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import wandb
from src.data.dataset import load_prompts
from src.models.attention_utils import process_prompt_pair
from src.models.extract_attention import load_model_and_tokenizer, run_inference_and_extract_attention


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Starting experiment: {cfg.experiment_name}")
    logger.info(f"Model: {cfg.model.name}, quantization: {cfg.model.quantization}")

    if cfg.wandb.get("enabled", True):
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.experiment_name,
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        wandb.init(mode="disabled")

    project_root = Path(get_original_cwd())
    experiment_dir = project_root / cfg.paths.data_processed / cfg.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(project_root / cfg.paths.data_raw / "prompts.json")
    logger.info(f"Loaded {len(prompts)} prompt pairs.")

    try:
        model, tokenizer = load_model_and_tokenizer(
            cfg.model.name,
            quantization=cfg.model.quantization,
            device=cfg.model.device,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("Continuing in dry-run mode.")
        model = None
        tokenizer = None

    for item in prompts:
        prompt_id = item["id"]
        base_prompt = item["base_prompt"]
        modified_prompt = item["modified_prompt"]

        logger.info(f"Processing pair: {prompt_id}")
        logger.info(f"  Base:     {base_prompt}")
        logger.info(f"  Modified: {modified_prompt}")

        if model is None or tokenizer is None:
            continue

        device = cfg.model.device
        attrs_base, tok_base = run_inference_and_extract_attention(model, tokenizer, base_prompt, device=device)
        attrs_mod, tok_mod = run_inference_and_extract_attention(model, tokenizer, modified_prompt, device=device)

        tokens_base = tokenizer.convert_ids_to_tokens(tok_base[0])
        tokens_mod = tokenizer.convert_ids_to_tokens(tok_mod[0])

        saved_paths = process_prompt_pair(prompt_id, attrs_base, attrs_mod, tokens_base, tokens_mod, experiment_dir)
        wandb.log({key: wandb.Image(str(path)) for key, path in saved_paths.items()})

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    wandb.finish()
    logger.info("Experiment finished.")


if __name__ == "__main__":
    main()
