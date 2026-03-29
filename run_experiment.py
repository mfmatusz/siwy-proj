import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import sys
import torch
import wandb
from src.data.dataset import load_prompts
from src.models.extract_attention import load_model_and_tokenizer, run_inference_and_extract_attention
from src.models.xai_utils import mean_pooling_heads
from src.visualization.visualize import plot_attention_heatmap

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Starting experiment: {cfg.experiment_name}")
    log.info(f"Model: {cfg.model.name}, Quantization Config: {cfg.model.quantization}")
    
    # Initialize Weights & Biases
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.experiment_name,
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Make sure processed data dir exists
    processed_dir = Path(cfg.paths.data_processed)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the dataset (Prompts)
    prompts = load_prompts(f"{cfg.paths.data_raw}/prompts.json")
    log.info(f"Loaded {len(prompts)} prompt examples for analysis.")

    # 2. Load Model 
    log.info("Loading model... Please wait, this might take a while depending on VRAM / connection.")
    try:
        model, tokenizer = load_model_and_tokenizer(
            cfg.model.name, 
            quantization=cfg.model.quantization,
            device=cfg.model.device
        )
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        log.warning("Continuing in DRY-RUN mode without real model loaded.")
        model = None
        tokenizer = None

    for item in prompts:
        prompt_id = item["id"]
        b_prompt = item["base_prompt"]
        m_prompt = item["modified_prompt"]
        
        log.info(f"Testing pair iteratively: {prompt_id}")

        if model is not None and tokenizer is not None:
            # 3. Perform analysis for both base and modified prompts
            attrs_base, tok_base = run_inference_and_extract_attention(model, tokenizer, b_prompt, device=cfg.model.device)
            attrs_mod, tok_mod = run_inference_and_extract_attention(model, tokenizer, m_prompt, device=cfg.model.device)

            # Get the tokens strings for visualization
            string_tokens_base = tokenizer.convert_ids_to_tokens(tok_base[0])
            string_tokens_mod = tokenizer.convert_ids_to_tokens(tok_mod[0])

            # Select the last layer for demonstration (layer -1)
            last_layer_attn_base = attrs_base[-1][0] # shape: (num_heads, seq_len, seq_len)
            last_layer_attn_mod = attrs_mod[-1][0]
            
            # Mean pooling over heads
            mean_attn_base = last_layer_attn_base.mean(dim=0) # shape: (seq_len, seq_len)
            mean_attn_mod = last_layer_attn_mod.mean(dim=0)

            # 4. Save results to disk
            torch.save(mean_attn_base, processed_dir / f"{prompt_id}_base.pt")
            torch.save(mean_attn_mod, processed_dir / f"{prompt_id}_mod.pt")

            # 5. Generate and log heatmaps
            heatmap_path_base = str(processed_dir / f"{prompt_id}_heatmap_base.png")
            heatmap_path_mod = str(processed_dir / f"{prompt_id}_heatmap_mod.png")
            
            plot_attention_heatmap(mean_attn_base, string_tokens_base, heatmap_path_base, title=f"Base: {prompt_id}")
            plot_attention_heatmap(mean_attn_mod, string_tokens_mod, heatmap_path_mod, title=f"Modified: {prompt_id}")

            wandb.log({
                f"heatmaps/{prompt_id}_base": wandb.Image(heatmap_path_base),
                f"heatmaps/{prompt_id}_mod": wandb.Image(heatmap_path_mod)
            })

    wandb.finish()
    log.info("Experiment finished successfully.")

if __name__ == "__main__":
    main()
