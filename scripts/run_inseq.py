from pathlib import Path

import dotenv
import hydra

dotenv.load_dotenv()
import torch
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import wandb
from src.attribution.inseq_analysis import load_inseq_model, run_attribution, save_attribution
from src.data.dataset import load_prompts

ATTRIBUTION_METHODS = ["attention"]


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Starting Inseq attribution: {cfg.experiment_name}")

    if cfg.wandb.get("enabled", True):
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.experiment_name}_inseq",
            tags=list(cfg.wandb.tags) + ["inseq"],
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        wandb.init(mode="disabled")

    project_root = Path(get_original_cwd())
    experiment_dir = project_root / cfg.paths.data_processed / f"{cfg.experiment_name}_inseq"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(project_root / cfg.paths.data_raw / "prompts.json")
    logger.info(f"Loaded {len(prompts)} prompt pairs.")

    for method in ATTRIBUTION_METHODS:
        logger.info(f"Loading model with method: {method}")
        model = load_inseq_model(cfg.model.name, method)

        for item in prompts:
            prompt_id = item["id"]
            base_prompt = item["base_prompt"]
            modified_prompt = item["modified_prompt"]

            logger.info(f"[{method}] Processing: {prompt_id}")
            logger.info(f"  Base:     {base_prompt}")
            logger.info(f"  Modified: {modified_prompt}")

            attr_base = run_attribution(model, base_prompt)
            attr_mod = run_attribution(model, modified_prompt)

            method_dir = experiment_dir / prompt_id / method
            save_attribution(attr_base, method_dir / "base.json")
            save_attribution(attr_mod, method_dir / "modified.json")

            base_html = attr_base.show(display=False, return_html=True, do_aggregation=True)
            mod_html = attr_mod.show(display=False, return_html=True, do_aggregation=True)

            wandb.log(
                {
                    f"inseq/{method}/{prompt_id}/base": wandb.Html(base_html),
                    f"inseq/{method}/{prompt_id}/modified": wandb.Html(mod_html),
                }
            )

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    wandb.finish()
    logger.info("Inseq attribution finished.")


if __name__ == "__main__":
    main()
