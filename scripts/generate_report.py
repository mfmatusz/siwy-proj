import webbrowser
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig

from src.visualization.report import generate_html_report


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    project_root = Path(get_original_cwd())
    processed = project_root / cfg.paths.data_processed
    experiment_dir = processed / cfg.experiment_name
    inseq_dir = processed / f"{cfg.experiment_name}_inseq"
    report_path = processed / f"{cfg.experiment_name}_report.html"

    logger.info(f"Building report for: {cfg.experiment_name}")

    html = generate_html_report(experiment_dir, inseq_dir, cfg.experiment_name)
    report_path.write_text(html, encoding="utf-8")

    logger.info(f"Report saved to {report_path}")
    webbrowser.open(f"file://{report_path.resolve()}")


if __name__ == "__main__":
    main()
