"""HuggingFace Transformers adapter for text proxy model training."""

import json
import time
from pathlib import Path
from typing import Any

from curationgym.train.adapters.base import TrainingAdapter, TrainingBudget, TrainingResult


# Model configs for different parameter counts (GPT-2 style)
MODEL_CONFIGS = {
    50_000_000: {  # ~50M
        "n_embd": 512,
        "n_layer": 8,
        "n_head": 8,
        "vocab_size": 50257,
    },
    150_000_000: {  # ~150M
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "vocab_size": 50257,
    },
    400_000_000: {  # ~400M
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16,
        "vocab_size": 50257,
    },
}


class HFTextAdapter(TrainingAdapter):
    """Training adapter using HuggingFace Transformers."""

    def __init__(self, model_type: str = "gpt2"):
        self.model_type = model_type

    @property
    def name(self) -> str:
        return f"hf_{self.model_type}"

    def get_model_config(self, params: int) -> dict[str, Any]:
        """Get closest model config for target params."""
        closest = min(MODEL_CONFIGS.keys(), key=lambda x: abs(x - params))
        return MODEL_CONFIGS[closest].copy()

    def train(
        self,
        dataset_manifest_path: str | Path,
        budget: TrainingBudget,
        output_dir: str | Path,
        seed: int = 42,
    ) -> TrainingResult:
        """Train GPT-2 style model on dataset."""
        try:
            from transformers import (
                GPT2Config,
                GPT2LMHeadModel,
                Trainer,
                TrainingArguments,
                AutoTokenizer,
            )
        except ImportError:
            raise ImportError("transformers required: pip install transformers")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # Load manifest
        manifest = json.loads(Path(dataset_manifest_path).read_text())

        # Create model config
        model_config_dict = self.get_model_config(budget.model_params)
        config = GPT2Config(**model_config_dict)
        model = GPT2LMHeadModel(config)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Calculate training steps
        tokens_per_step = budget.batch_size_tokens
        max_steps = budget.max_steps or (budget.max_tokens // tokens_per_step)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            max_steps=max_steps,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=tokens_per_step // (8 * 512),  # Approximate
            learning_rate=6e-4,
            warmup_steps=min(1000, max_steps // 10),
            logging_steps=100,
            save_steps=max_steps // 5,
            seed=seed,
            fp16=True,
            report_to=[],
        )

        # Create dataset from manifest shards
        from curationgym.train.dataloader import create_dataset_from_manifest

        train_dataset = create_dataset_from_manifest(
            manifest_path=dataset_manifest_path,
            tokenizer=tokenizer,
            max_length=512,
            seed=seed,
        )

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        train_result = trainer.train()

        # Save final checkpoint
        checkpoint_path = output_dir / "final_checkpoint"
        trainer.save_model(str(checkpoint_path))

        wall_time = time.time() - start_time

        # Collect metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", wall_time),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        }

        # Save metrics
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        return TrainingResult(
            checkpoint_path=str(checkpoint_path),
            final_loss=train_result.training_loss,
            tokens_trained=max_steps * tokens_per_step,
            steps_completed=train_result.global_step,
            wall_time_seconds=wall_time,
            metrics=metrics,
        )
