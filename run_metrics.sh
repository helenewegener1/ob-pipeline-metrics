#!/usr/bin/env bash
set -euo pipefail

# Run data_preprocessing.py with the requested parameters.
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
python_bin="${script_dir}/.venv/bin/python"
[ -x "$python_bin" ] || python_bin="python"

# "${python_bin}" "${script_dir}/flow_metrics.py" \
#   --name "dgcytof" \
#   --output_dir "${script_dir}/out/data/metrics/all/flow_metrics" \
#   --analysis.prediction "${script_dir}/out/data/analysis/default/dgcytof/dgcytof_predicted_labels.txt" \
#   --data.true_labels "${script_dir}/out/data/data_preprocessing/default/data_preprocessing_labels.txt" \
#   --metric "all"

"${python_bin}" "${script_dir}/flow_metrics.py" \
  --name "random" \
  --output_dir "${script_dir}/out/data/metrics/all/flow_metrics" \
  --analysis.prediction "${script_dir}/out/data/analysis/default/random/data_import_predicted_labels.txt" \
  --data.true_labels "${script_dir}/out/data/data_preprocessing/default/data_import.test.labels.gz" \
  --metric "all"
