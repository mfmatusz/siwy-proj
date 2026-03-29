import hydra
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.data.dataset import load_prompts
from src.models.extract_attention import load_model_and_tokenizer, run_inference_and_extract_attention
from src.visualization.visualize import plot_attention_heatmap


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Starting experiment: {cfg.experiment_name}")
    logger.info(f"Model: {cfg.model.name}, quantization: {cfg.model.quantization}")

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.experiment_name,
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    processed_dir = Path(cfg.paths.data_processed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(f"{cfg.paths.data_raw}/prompts.json")
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

        if model is None or tokenizer is None:
            continue

        attrs_base, tok_base = run_inference_and_extract_attention(model, tokenizer, base_prompt, device=cfg.model.device)
        attrs_mod, tok_mod = run_inference_and_extract_attention(model, tokenizer, modified_prompt, device=cfg.model.device)

        tokens_base = tokenizer.convert_ids_to_tokens(tok_base[0])
        tokens_mod = tokenizer.convert_ids_to_tokens(tok_mod[0])

        last_layer_base = attrs_base[-1][0].mean(dim=0)
        last_layer_mod = attrs_mod[-1][0].mean(dim=0)

        torch.save(last_layer_base, processed_dir / f"{prompt_id}_base.pt")
        torch.save(last_layer_mod, processed_dir / f"{prompt_id}_mod.pt")

        heatmap_base = str(processed_dir / f"{prompt_id}_heatmap_base.png")
        heatmap_mod = str(processed_dir / f"{prompt_id}_heatmap_mod.png")

        plot_attention_heatmap(last_layer_base, tokens_base, heatmap_base, title=f"Base: {prompt_id}")
        plot_attention_heatmap(last_layer_mod, tokens_mod, heatmap_mod, title=f"Modified: {prompt_id}")

        wandb.log({
            f"heatmaps/{prompt_id}_base": wandb.Image(heatmap_base),
            f"heatmaps/{prompt_id}_mod": wandb.Image(heatmap_mod),
        })

    wandb.finish()
    logger.info("Experiment finished.")


if __name__ == "__main__":
    main()
