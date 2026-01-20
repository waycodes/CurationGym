# CurationGym

A reproducible, compute-budget-aware system for searching over data curation and mixing policies using proxy-model training and standard evaluation suites.

## Features

- **Policy-driven curation**: Define filtering, deduplication, decontamination, and mixing policies
- **Proxy model training**: Train small GPT-2 models (50M-400M) for fast iteration
- **Standard evaluation**: Integration with lm-evaluation-harness for benchmarking
- **Slice attribution**: Ridge regression and ablation-based attribution to understand which data slices improve benchmarks
- **Compute budgeting**: Multi-fidelity optimization with successive halving
- **Reproducibility**: Run stamping, manifest-based rebuild, dataset cards

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from curationgym.policy.schema import Policy
from curationgym.policy.execute import execute_policy

# Define a curation policy
policy = Policy(
    filters=[
        {"name": "length", "min_tokens": 50, "max_tokens": 10000},
        {"name": "quality", "threshold": 0.5},
    ],
    dedup={"method": "minhash", "threshold": 0.8},
    decontam={"mode": "ngram", "ngram_size": 13, "benchmarks": ["hellaswag", "arc"]},
    mixing={"domain:wiki": 0.4, "domain:news": 0.3, "domain:code": 0.3},
)

# Execute policy on input documents
manifest = execute_policy(
    policy=policy,
    input_docs=your_document_iterator,
    output_dir="./output",
)
```

## Architecture

```
src/curationgym/
├── core/           # Document model, manifest, artifact store
├── io/             # Data readers (HF, CommonCrawl)
├── operators/      # Filters, dedup, decontam operators
├── slices/         # Slice taxonomy and assignment
├── policy/         # Policy schema and execution
├── train/          # Proxy model training adapters
├── eval/           # Evaluation harness integration
├── optim/          # Search space, optimizers, multi-fidelity
├── attribution/    # Slice-to-benchmark attribution
├── release/        # Run stamping, dataset cards, HF export
├── report/         # Experiment reports and visualization
├── recovery.py     # Checkpoint/resume, graceful shutdown
└── profiling.py    # Performance profiling
```

## Modules

### Operators

- **Filters**: Length, language, quality, URL, PII masking
- **Deduplication**: Exact, MinHash, semantic, scoped
- **Decontamination**: N-gram based with remove/flag modes

### Optimization

- **Search spaces**: Categorical, continuous, integer parameters with constraints
- **Optimizers**: Random search, Optuna TPE with cost-aware objectives
- **Multi-fidelity**: Successive halving for compute-efficient search
- **Pareto tracking**: Multi-objective optimization support

### Attribution

- **Ridge regression**: Correlational attribution with bootstrapped CIs
- **Ablation**: Causal attribution by removing slices
- **Sanity checks**: Permutation tests, seed stability, negative controls

### Reproducibility

- **Run stamping**: Git commit, dependencies, hardware info
- **Manifest rebuild**: Reconstruct datasets from manifests
- **Dataset cards**: Auto-generated documentation
- **HF export**: Push to HuggingFace Hub

## Configuration

See `configs/` for example configurations:
- `configs/budgets/default.yaml` - Compute budget settings
- `configs/slices/default_taxonomy.yaml` - Slice definitions
- `configs/decontam/default.yaml` - Decontamination settings

## Testing

```bash
pytest tests/
```

## License

MIT
