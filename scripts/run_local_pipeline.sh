#!/bin/bash
# Example script for running a local pipeline
# Usage: ./scripts/run_local_pipeline.sh <config.yaml> <output_dir>

set -e

CONFIG=${1:-"configs/pipelines/example.yaml"}
OUTPUT=${2:-"runs/local_test"}

echo "Running local pipeline..."
echo "Config: $CONFIG"
echo "Output: $OUTPUT"

python -c "
from curationgym.pipeline import PipelineBuilder
from curationgym.pipeline.executors import LocalExecutor
from curationgym.core import Document

# Build pipeline from config
builder = PipelineBuilder()
# builder.load_config('$CONFIG')
# pipeline = builder.build()

# For demo: create simple pipeline
from curationgym.pipeline import DataTroveAdapter
pipeline = DataTroveAdapter()
pipeline.add_filter(lambda d: len(d.text) > 10)

# Create sample input
def sample_input():
    for i in range(100):
        yield Document(text=f'Sample document {i} with some content', id=f'doc_{i}')

# Execute
executor = LocalExecutor('$OUTPUT', num_workers=1)
state = executor.execute(pipeline, [sample_input], run_id='test_run')

print(f'Completed: {sum(1 for t in state.tasks.values() if t.status == \"completed\")} tasks')
print(f'Total docs: {sum(t.docs_processed for t in state.tasks.values())}')
"

echo "Done. Output in $OUTPUT"
