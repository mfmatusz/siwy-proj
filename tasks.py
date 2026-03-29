from invoke import task

PROJECT_ROOT = "."


@task
def lint(c):
    c.run("ruff check src/ scripts/ tests/")


@task
def format(c):
    c.run("ruff format src/ scripts/ tests/")


@task
def test(c):
    c.run("pytest tests/ -v")


@task
def run(c, config_overrides=""):
    c.run(f"PYTHONPATH={PROJECT_ROOT} python scripts/run_experiment.py {config_overrides}")


@task
def run_inseq(c, config_overrides=""):
    c.run(f"PYTHONPATH={PROJECT_ROOT} python scripts/run_inseq.py {config_overrides}")


@task
def report(c, config_overrides=""):
    c.run(f"PYTHONPATH={PROJECT_ROOT} python scripts/generate_report.py {config_overrides}")


@task(pre=[lint, test])
def check(c):
    pass
