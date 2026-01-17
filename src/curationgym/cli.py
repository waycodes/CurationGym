"""CurationGym CLI - Data curation policy search."""

import click


@click.group()
@click.version_option()
def main() -> None:
    """CurationGym: Compute-budget-aware data curation policy search."""
    pass


@main.command()
@click.option("--policy", "-p", required=True, help="Path to policy config YAML")
@click.option("--output", "-o", required=True, help="Output directory for dataset")
@click.option("--dry-run", is_flag=True, help="Validate policy without full execution")
def curate(policy: str, output: str, dry_run: bool) -> None:
    """Execute a curation policy to produce a dataset."""
    click.echo(f"Curating with policy: {policy}")
    click.echo(f"Output: {output}")
    if dry_run:
        click.echo("(dry run mode)")
    raise NotImplementedError("Curation pipeline not yet implemented")


@main.command()
@click.option("--manifest", "-m", required=True, help="Dataset manifest path")
@click.option("--budget", "-b", default="small", help="Compute budget tier")
@click.option("--output", "-o", required=True, help="Output directory for checkpoints")
def train(manifest: str, budget: str, output: str) -> None:
    """Train a proxy model on a curated dataset."""
    click.echo(f"Training on: {manifest}")
    click.echo(f"Budget: {budget}")
    click.echo(f"Output: {output}")
    raise NotImplementedError("Training harness not yet implemented")


@main.command("eval")
@click.option("--checkpoint", "-c", required=True, help="Model checkpoint path")
@click.option("--suite", "-s", default="configs/evals/text_eval_suite.yaml", help="Eval suite config")
@click.option("--output", "-o", required=True, help="Output directory for results")
def evaluate(checkpoint: str, suite: str, output: str) -> None:
    """Evaluate a model on the benchmark suite."""
    click.echo(f"Evaluating: {checkpoint}")
    click.echo(f"Suite: {suite}")
    click.echo(f"Output: {output}")
    raise NotImplementedError("Evaluation harness not yet implemented")


@main.command()
@click.option("--search-space", "-s", required=True, help="Search space config")
@click.option("--budget", "-b", required=True, help="Budget config path")
@click.option("--output", "-o", required=True, help="Output directory for results")
@click.option("--method", "-m", default="random", help="Optimization method")
def optimize(search_space: str, budget: str, output: str, method: str) -> None:
    """Search for optimal curation policies."""
    click.echo(f"Optimizing with method: {method}")
    click.echo(f"Search space: {search_space}")
    click.echo(f"Budget: {budget}")
    click.echo(f"Output: {output}")
    raise NotImplementedError("Optimizer not yet implemented")


@main.command()
@click.option("--run-dir", "-r", required=True, help="Run directory to report on")
@click.option("--output", "-o", help="Output path for report")
def report(run_dir: str, output: str | None) -> None:
    """Generate experiment report with attribution."""
    click.echo(f"Generating report for: {run_dir}")
    if output:
        click.echo(f"Output: {output}")
    raise NotImplementedError("Report generator not yet implemented")


if __name__ == "__main__":
    main()
