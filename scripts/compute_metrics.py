import csv
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.data.dataset import load_prompts
from src.metrics import attention_entropy, pairwise_attention_diff, sparsity_ratio


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())
    experiment_dir = project_root / cfg.paths.data_processed / cfg.experiment_name

    prompts = load_prompts(project_root / cfg.paths.data_raw / "prompts.json")
    prompts_map = {p["id"]: p for p in prompts}

    rows: list[dict] = []

    for prompt_dir in sorted(experiment_dir.iterdir()):
        pid = prompt_dir.name
        tensors_dir = prompt_dir / "tensors"
        if not tensors_dir.exists():
            continue

        try:
            overall_base = torch.load(tensors_dir / "overall_base.pt", weights_only=True)
            overall_mod = torch.load(tensors_dir / "overall_mod.pt", weights_only=True)
            local_base = torch.load(tensors_dir / "local_base.pt", weights_only=True)
            local_mod = torch.load(tensors_dir / "local_mod.pt", weights_only=True)
            global_base = torch.load(tensors_dir / "global_base.pt", weights_only=True)
            global_mod = torch.load(tensors_dir / "global_mod.pt", weights_only=True)
        except FileNotFoundError as e:
            print(f"Skipping {pid}: {e}")
            continue

        diff = pairwise_attention_diff(overall_base, overall_mod)

        rows.append(
            {
                "id": pid,
                "category": prompts_map[pid]["category"],
                "entropy_base": round(attention_entropy(overall_base).mean().item(), 4),
                "entropy_mod": round(attention_entropy(overall_mod).mean().item(), 4),
                "entropy_delta": round(
                    attention_entropy(overall_mod).mean().item()
                    - attention_entropy(overall_base).mean().item(),
                    4,
                ),
                "sparsity_base": round(sparsity_ratio(overall_base), 4),
                "sparsity_mod": round(sparsity_ratio(overall_mod), 4),
                "diff_l1": round(diff.abs().mean().item(), 6),
                "diff_l2": round(diff.pow(2).mean().sqrt().item(), 6),
                "local_entropy_delta": round(
                    attention_entropy(local_mod).mean().item()
                    - attention_entropy(local_base).mean().item(),
                    4,
                ),
                "global_entropy_delta": round(
                    attention_entropy(global_mod).mean().item()
                    - attention_entropy(global_base).mean().item(),
                    4,
                ),
            }
        )

    if not rows:
        print(f"No tensor data found in {experiment_dir}. Run `make run` first.")
        return

    output_path = experiment_dir / "metrics.csv"
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")

    from collections import defaultdict

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)

    numeric_keys = [k for k in fieldnames if k not in ("id", "category")]

    cat_rows: list[dict] = []
    for cat in sorted(by_cat):
        items = by_cat[cat]
        avgs = {k: round(sum(r[k] for r in items) / len(items), 4) for k in numeric_keys}
        cat_rows.append({"category": cat, "n": len(items), **avgs})

    cat_output_path = experiment_dir / "metrics_by_category.csv"
    cat_fieldnames = ["category", "n"] + numeric_keys
    with cat_output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cat_fieldnames)
        writer.writeheader()
        writer.writerows(cat_rows)

    print(f"Saved {len(cat_rows)} rows to {cat_output_path}")

    print("\nPer-category averages:")
    header = f"{'category':<16}" + "".join(f"{k:>22}" for k in numeric_keys)
    print(header)
    for row in cat_rows:
        line = f"{row['category']:<16}" + "".join(f"{row[k]:>22.4f}" for k in numeric_keys)
        print(line)


if __name__ == "__main__":
    main()
