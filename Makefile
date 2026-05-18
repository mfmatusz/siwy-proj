.DEFAULT_GOAL := help

.PHONY: help install install-dev lint format test check run run-inseq report metrics clean

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup"
	@echo "  install      Install project dependencies"
	@echo "  install-dev  Install project + dev dependencies (pytest, ruff)"
	@echo "  clean        Remove .venv (use before reinstalling on a new platform)"
	@echo ""
	@echo "Development"
	@echo "  lint         Run ruff linter"
	@echo "  format       Run ruff formatter"
	@echo "  test         Run pytest"
	@echo "  check        Run lint + tests"
	@echo ""
	@echo "Experiment"
	@echo "  run          Extract attention weights (all prompt pairs)"
	@echo "  run-inseq    Run Inseq attribution analysis"
	@echo "  report       Generate HTML report"
	@echo "  metrics      Compute per-pair and per-category metrics from saved tensors → metrics.csv, metrics_by_category.csv"
	@echo ""
	@echo "Pass Hydra overrides via ARGS, e.g.:"
	@echo "  make run ARGS='model.device=cuda experiment_name=my_run'"

clean:
	rm -rf .venv

install:
	uv sync

install-dev:
	uv sync --extra dev

lint:
	uv run ruff check src/ scripts/ tests/

format:
	uv run ruff format src/ scripts/ tests/

test:
	uv sync --extra dev
	uv run python -m pytest tests/ -v

check: lint test

run:
	uv run python scripts/run_experiment.py $(ARGS)

run-inseq:
	uv run python scripts/run_inseq.py $(ARGS)

report:
	uv run python scripts/generate_report.py $(ARGS)

metrics:
	uv run python scripts/compute_metrics.py $(ARGS)
