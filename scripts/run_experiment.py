import dotenv

dotenv.load_dotenv()

from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import wandb
from src.config.model import validate_model_config
from src.data.dataset import load_prompts
from src.data.token_labels import categorize_tokens, load_token_categories
from src.metrics import attention_entropy, mean_attention_by_category, pairwise_attention_diff, sparsity_ratio
from src.models.attention_utils import aggregate_attention_by_type, process_prompt_pair
from src.models.extract_attention import load_model_and_tokenizer, run_inference_and_extract_attention
from src.visualization.visualize import plot_category_attention_barchart


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    validate_model_config(cfg)
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

    token_cats_file = project_root / cfg.paths.data_raw / "token_categories.json"
    token_cats_data = load_token_categories(token_cats_file) if token_cats_file.exists() else {}
    category_records: list[dict] = []

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

        _, _, overall_base = aggregate_attention_by_type(attrs_base)
        _, _, overall_mod = aggregate_attention_by_type(attrs_mod)
        diff = pairwise_attention_diff(overall_base, overall_mod)

        cats_data = token_cats_data.get(prompt_id, {})
        instruction_kws = cats_data.get("instruction_keywords", [])
        content_kws = cats_data.get("content_keywords", [])
        token_cats_base = categorize_tokens(tokens_base, instruction_kws, content_kws)
        token_cats_mod = categorize_tokens(tokens_mod, instruction_kws, content_kws)
        metrics_cat_base = mean_attention_by_category(overall_base, token_cats_base)
        metrics_cat_mod = mean_attention_by_category(overall_mod, token_cats_mod)
        category_records.append({"prompt_category": item["category"], "condition": "base", **metrics_cat_base})
        category_records.append({"prompt_category": item["category"], "condition": "modified", **metrics_cat_mod})

        wandb.log(
            {
                f"metrics/{prompt_id}/entropy_base": attention_entropy(overall_base).mean().item(),
                f"metrics/{prompt_id}/entropy_mod": attention_entropy(overall_mod).mean().item(),
                f"metrics/{prompt_id}/sparsity_base": sparsity_ratio(overall_base),
                f"metrics/{prompt_id}/sparsity_mod": sparsity_ratio(overall_mod),
                f"metrics/{prompt_id}/diff_l1": diff.abs().mean().item(),
                f"metrics/{prompt_id}/diff_l2": diff.pow(2).mean().sqrt().item(),
            }
        )
        wandb.log({key: wandb.Image(str(path)) for key, path in saved_paths.items()})

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if category_records:
        barchart_path = experiment_dir / "category_attention_barchart.png"
        plot_category_attention_barchart(
            category_records,
            str(barchart_path),
            title=f"Mean Attention by Token Category — {cfg.experiment_name}",
        )
        wandb.log({"category_attention_barchart": wandb.Image(str(barchart_path))})

    wandb.finish()
    logger.info("Experiment finished.")


if __name__ == "__main__":
    main()
