from invoke import task


@task
def lint(c):
    c.run("ruff check src/ tests/")


@task
def format(c):
    c.run("ruff format src/ tests/")


@task
def test(c):
    c.run("pytest tests/ -v")


@task
def run(c, config_overrides=""):
    c.run(f"python run_experiment.py {config_overrides}")


@task(pre=[lint, test])
def check(c):
    pass
