"""lm-evaluation-harness adapter for text model evaluation."""

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from curationgym.eval.runner import EvalRunner, EvalResult


class LMEvalAdapter(EvalRunner):
    """Adapter for lm-evaluation-harness."""

    def __init__(self, harness_version: str = "0.4.0"):
        self.harness_version = harness_version

    @property
    def name(self) -> str:
        return "lm-evaluation-harness"

    def get_code_version(self) -> str:
        """Get lm-eval version."""
        try:
            import lm_eval
            return getattr(lm_eval, "__version__", self.harness_version)
        except ImportError:
            return self.harness_version

    def evaluate(
        self,
        checkpoint_path: str | Path,
        eval_suite_config: str | Path,
        output_dir: str | Path,
    ) -> EvalResult:
        """Run lm-evaluation-harness on checkpoint."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load eval suite config
        with open(eval_suite_config) as f:
            suite_config = yaml.safe_load(f)

        tasks = [t["name"] for t in suite_config.get("tasks", [])]
        task_weights = {t["name"]: t.get("weight", 1.0) for t in suite_config.get("tasks", [])}

        try:
            # Try using lm-eval Python API
            from lm_eval import evaluator
            from lm_eval.models.huggingface import HFLM

            model = HFLM(pretrained=str(checkpoint_path))

            results = evaluator.simple_evaluate(
                model=model,
                tasks=tasks,
                batch_size="auto",
                device="cuda" if self._has_cuda() else "cpu",
            )

            return self._parse_results(results, task_weights, suite_config)

        except ImportError:
            # Fall back to CLI
            return self._run_cli(checkpoint_path, tasks, task_weights, output_dir, suite_config)

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _run_cli(
        self,
        checkpoint_path: str | Path,
        tasks: list[str],
        task_weights: dict[str, float],
        output_dir: Path,
        suite_config: dict,
    ) -> EvalResult:
        """Run evaluation via CLI."""
        results_file = output_dir / "lm_eval_results.json"

        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={checkpoint_path}",
            "--tasks", ",".join(tasks),
            "--output_path", str(results_file),
            "--batch_size", "auto",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            with open(results_file) as f:
                results = json.load(f)
            return self._parse_results(results, task_weights, suite_config)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # Return empty result on failure
            return EvalResult(
                task_scores={},
                aggregate_score=0.0,
                eval_code_version=self.get_code_version(),
                raw_results={"error": str(e)},
            )

    def _parse_results(
        self,
        results: dict[str, Any],
        task_weights: dict[str, float],
        suite_config: dict,
    ) -> EvalResult:
        """Parse lm-eval results into EvalResult."""
        task_scores: dict[str, float] = {}
        confidence_intervals: dict[str, tuple[float, float]] = {}
        task_versions: dict[str, str] = {}

        results_dict = results.get("results", results)

        for task_name, task_results in results_dict.items():
            if isinstance(task_results, dict):
                # Get primary metric (usually acc or acc_norm)
                for metric in ["acc_norm", "acc", "exact_match", "f1"]:
                    if metric in task_results:
                        task_scores[task_name] = task_results[metric]
                        # Get stderr if available
                        stderr_key = f"{metric}_stderr"
                        if stderr_key in task_results:
                            stderr = task_results[stderr_key]
                            score = task_scores[task_name]
                            confidence_intervals[task_name] = (score - 1.96 * stderr, score + 1.96 * stderr)
                        break

                # Record task version if available
                if "version" in task_results:
                    task_versions[task_name] = str(task_results["version"])

        # Compute weighted aggregate
        total_weight = sum(task_weights.get(t, 1.0) for t in task_scores)
        if total_weight > 0:
            aggregate = sum(
                task_scores[t] * task_weights.get(t, 1.0)
                for t in task_scores
            ) / total_weight
        else:
            aggregate = 0.0

        return EvalResult(
            task_scores=task_scores,
            aggregate_score=aggregate,
            confidence_intervals=confidence_intervals,
            eval_code_version=self.get_code_version(),
            task_versions=task_versions,
            raw_results=results,
        )
