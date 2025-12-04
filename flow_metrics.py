#!/usr/bin/env python

"""
Lightweight CLI for evaluating prediction-style outputs (e.g. cell type
classification) using the same inputs as `run_metrics.py`, but focused on
per-population prediction metrics instead of clustering comparison.

Inputs mirror run_metrics:
- --clustering.predicted_ks_range: csv-like file with header row of run IDs
  (e.g. k=...) and one column of predictions per run
- --data.true_labels: text file of ground-truth labels (1D)
- --metric: comma-separated list from VALID_METRICS (or "all")
- --output_dir/--name: where to write results; printed to stdout if omitted

Metrics implemented (selected via --metric):
- accuracy, precision, recall, f1 (per-population with macro averages)
- runtime: time spent computing the metrics for that run
- overlap: Jaccard overlap between predicted and true label sets (ignores 0)
- scalability: runtime normalized by number of evaluated samples

Note: this is a scaffolding for future refinement; runtime here measures the
metric computation itself, not the upstream model execution.
"""

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd

VALID_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_score",
    "runtime",
    "overlap",
    "scalability",
    "all",
}


CLASSIFICATION_METRICS = {"accuracy", "precision", "recall", "f1"}


def _read_first_line(path):
    """Read the first line of a (possibly gzipped) file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        return handle.readline()


def _has_header(first_line):
    """Heuristically decide whether the first line is a header row."""
    tokens = [tok for tok in first_line.replace(",", " ").split() if tok]
    if not tokens:
        return False
    for tok in tokens:
        try:
            float(tok)
        except ValueError:
            return True
    return False


def load_true_labels(data_file):
    """
    Load labels as 1D array; keeps missing labels as NaN (needed for
    semi-supervised handling in preprocessing).
    """
    first_line = _read_first_line(data_file)
    has_header = _has_header(first_line)

    opener = gzip.open if data_file.endswith(".gz") else open
    with opener(data_file, "rt") as handle:
        series = pd.read_csv(
            handle,
            header=0 if has_header else None,
            comment="#",
            na_values=["", '""', "nan", "NaN"],
            skip_blank_lines=False,
        ).iloc[:, 0]

    try:
        labels = pd.to_numeric(series, errors="coerce").to_numpy()
    except Exception as exc:
        raise ValueError("Invalid data structure, cannot parse labels.") from exc

    if labels.ndim != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")
    return labels


def load_predicted_labels(data_file):
    """
    Load predicted labels allowing for optional header rows and gzip input.
    Returns a tuple of (column_headers, predictions_matrix).
    """
    first_line = _read_first_line(data_file)
    has_header = _has_header(first_line)

    opener = gzip.open if data_file.endswith(".gz") else open
    parse_options = dict(
        header=0 if has_header else None,
        comment="#",
        na_values=["", '""', "nan", "NaN"],
        skip_blank_lines=False,
    )

    def _read_with_sep(sep):
        with opener(data_file, "rt") as handle:
            return pd.read_csv(
                handle,
                sep=sep,
                engine="python",
                **parse_options,
            )

    try:
        df = _read_with_sep(",")
    except pd.errors.ParserError:
        # Fallback for whitespace-delimited predictions
        df = _read_with_sep(r"\\s+")

    if df.empty:
        raise ValueError("Prediction file is empty.")

    try:
        values = df.apply(pd.to_numeric, errors="coerce").to_numpy()
    except Exception as exc:
        raise ValueError("Invalid data structure, cannot parse predictions.") from exc

    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("Invalid data structure, not a 2D matrix?")

    header = (
        [str(col) for col in df.columns]
        if has_header
        else [f"run{i}" for i in range(values.shape[1])]
    )
    return [np.array(header, dtype=str), values]


def parse_metric_argument(metric_arg):
    metrics = [m.strip().lower() for m in metric_arg.split(",") if m.strip()]
    if not metrics:
        raise ValueError("No metrics provided.")
    if "all" in metrics:
        metrics = sorted([m for m in VALID_METRICS if m != "all"])
    # Normalize aliases
    metrics = ["f1" if m == "f1_score" else m for m in metrics]
    metrics = list(dict.fromkeys(metrics))  # drop duplicates while preserving order
    invalid = [m for m in metrics if m not in VALID_METRICS]
    if invalid:
        raise ValueError(f"Invalid metric(s): {', '.join(invalid)}")
    return metrics


def _nan_safe_mean(values):
    vals = [v for v in values if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def strip_noise_labels(y_true, y_pred):
    y_true = np.array(y_true, ndmin=1)
    y_pred = np.array(y_pred, ndmin=1)
    mask = y_true > 0
    return y_true[mask], y_pred[mask]


def compute_per_population_stats(y_true, y_pred):
    per_population = {}
    labels = np.unique(y_true)
    for label in labels:
        pop_mask = y_true == label
        pop_size = pop_mask.sum()
        correct = (y_pred[pop_mask] == label).sum()
        tp = correct
        fp = ((y_true != label) & (y_pred == label)).sum()
        fn = ((y_true == label) & (y_pred != label)).sum()

        pop_accuracy = float(correct / pop_size) if pop_size else float("nan")
        pop_precision = float(tp / (tp + fp)) if (tp + fp) else float("nan")
        pop_recall = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        if (
            np.isnan(pop_precision)
            or np.isnan(pop_recall)
            or (pop_precision + pop_recall) == 0
        ):
            pop_f1 = float("nan")
        else:
            pop_f1 = float(
                2 * pop_precision * pop_recall / (pop_precision + pop_recall)
            )

        per_population[str(label)] = {
            "accuracy": pop_accuracy,
            "precision": pop_precision,
            "recall": pop_recall,
            "f1": pop_f1,
            "support": int(pop_size),
        }
    return per_population


def compute_macro_scores(per_population):
    macro_precision = _nan_safe_mean([v["precision"] for v in per_population.values()])
    macro_recall = _nan_safe_mean([v["recall"] for v in per_population.values()])
    macro_f1 = _nan_safe_mean([v["f1"] for v in per_population.values()])
    return macro_precision, macro_recall, macro_f1


def metric_accuracy(base_stats):
    return {"accuracy": base_stats["overall_accuracy"]}


def metric_precision(base_stats):
    return {"precision_macro": base_stats["macro_precision"]}


def metric_recall(base_stats):
    return {"recall_macro": base_stats["macro_recall"]}


def metric_f1(base_stats):
    return {"f1_macro": base_stats["macro_f1"]}


def metric_overlap(y_true, y_pred):
    true_labels = set(np.unique(y_true))
    pred_labels = set(np.unique(y_pred))
    true_labels.discard(0)
    pred_labels.discard(0)
    union = true_labels | pred_labels
    intersection = true_labels & pred_labels
    overlap = float(len(intersection) / len(union)) if union else float("nan")
    return {"overlap": overlap}


def metric_runtime(runtime_seconds):
    return {"runtime_seconds": runtime_seconds}


def metric_scalability(runtime_seconds, n_items):
    return {
        "scalability_seconds_per_item": (
            float(runtime_seconds / n_items) if n_items else float("nan")
        )
    }


def compute_prediction_metrics(y_true, y_pred, metrics_to_compute):
    """
    Computes per-population metrics and optional runtime/overlap/scalability
    for a single set of predictions.
    """
    start = time.perf_counter()

    y_true, y_pred = strip_noise_labels(y_true, y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Predicted labels and true labels must align in length.")

    results = {}

    # Base stats computed once for classification-style metrics
    if any(metric in CLASSIFICATION_METRICS for metric in metrics_to_compute):
        per_population = compute_per_population_stats(y_true, y_pred)
        macro_precision, macro_recall, macro_f1 = compute_macro_scores(per_population)
        base_stats = {
            "per_population": per_population,
            "overall_accuracy": (
                float((y_true == y_pred).mean()) if y_true.size else float("nan")
            ),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

        metric_dispatch = {
            "accuracy": lambda: metric_accuracy(base_stats),
            "precision": lambda: metric_precision(base_stats),
            "recall": lambda: metric_recall(base_stats),
            "f1": lambda: metric_f1(base_stats),
        }

        for metric_name, fn in metric_dispatch.items():
            if metric_name in metrics_to_compute:
                results.update(fn())

        results["per_population"] = per_population

    if "overlap" in metrics_to_compute:
        results.update(metric_overlap(y_true, y_pred))

    runtime_seconds = time.perf_counter() - start
    if "runtime" in metrics_to_compute:
        results.update(metric_runtime(runtime_seconds))
    if "scalability" in metrics_to_compute:
        results.update(metric_scalability(runtime_seconds, y_true.size))

    return results


def main():
    parser = argparse.ArgumentParser(description="Flow prediction metrics runner")

    parser.add_argument(
        "--analysis.prediction",
        type=str,
        required=True,
        help="csv text file with header row (k values) and columns of predictions",
    )
    parser.add_argument(
        "--labels_test",
        type=str,
        required=True,
        help="text file containing the true labels (1D)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory to store results (prints to stdout if omitted)",
    )
    parser.add_argument("--name", type=str, help="name of the dataset", default="flow")
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="comma-separated metrics to compute (or 'all')",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        parser.print_help()
        sys.exit(0)

    truth = load_true_labels(getattr(args, "data.true_labels"))
    ks, predicted = load_predicted_labels(getattr(args, "analysis.prediction"))
    metrics_to_compute = parse_metric_argument(args.metric)

    if predicted.shape[0] != truth.shape[0]:
        raise ValueError(
            f"Predicted labels rows ({predicted.shape[0]}) do not match true labels ({truth.shape[0]})."
        )

    results = {}
    for idx, k_label in enumerate(ks):
        metrics_for_k = compute_prediction_metrics(
            truth, predicted[:, idx], metrics_to_compute
        )
        results[str(k_label)] = metrics_for_k

    payload = {
        "name": args.name,
        "metrics_requested": metrics_to_compute,
        "results": results,
    }

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"{args.name}.flow_metrics.json.gz")
        with gzip.open(out_path, "wt") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved metrics to {out_path}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
