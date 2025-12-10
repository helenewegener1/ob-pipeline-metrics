#!/usr/bin/env python

"""
Lightweight CLI for evaluating prediction-style outputs (e.g. cell type
classification), using global confusion-matrix-based metrics
instead of per-population (per-label) metrics.

All classification metrics (accuracy, precision, recall, F1)
are now micro-averaged across all labels combined.
"""

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score
)

VALID_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_score",
    "runtime",
    "overlap",
    "scalability",
    "mcc",
    "popfreq_corr",
    "aucroc",
    "all",
}

CLASSIFICATION_METRICS = {"accuracy", "precision", "recall", "f1", "mcc", "aucroc"}


def _read_first_line(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        return handle.readline()


def _has_header(first_line):
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
    opener = gzip.open if data_file.endswith(".gz") else open
    with opener(data_file, "rt") as handle:
        series = pd.read_csv(
            handle,
            header=None,
            comment="#",
            na_values=["", '""', "nan", "NaN"],
            skip_blank_lines=True,
        ).iloc[:, 0]

    labels = pd.to_numeric(series, errors="coerce").to_numpy()
    if labels.ndim != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")
    return labels


def load_predicted_labels(data_file):
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
        df = _read_with_sep(r"\s+")

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
    metrics = ["f1" if m == "f1_score" else m for m in metrics]
    metrics = list(dict.fromkeys(metrics))
    invalid = [m for m in metrics if m not in VALID_METRICS]
    if invalid:
        raise ValueError(f"Invalid metric(s): {', '.join(invalid)}")
    return metrics


def strip_noise_labels(y_true, y_pred):
    y_true = np.array(y_true, ndmin=1)
    y_pred = np.array(y_pred, ndmin=1)
    mask = y_true > 0
    return y_true[mask], y_pred[mask]


# -------------------------------------------------------
# GLOBAL METRICS (REPLACES PER-POPULATION METRICS)
# -------------------------------------------------------

def compute_global_classification_stats(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tp = np.trace(cm)
    total = cm.sum()

    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tn = total - (fp + fn + np.diag(cm))

    precision = tp / (tp + fp.sum()) if (tp + fp.sum()) else float("nan")
    recall = tp / (tp + fn.sum()) if (tp + fn.sum()) else float("nan")

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    accuracy = tp / total if total else float("nan")

    return {
        "labels": labels.tolist(),
        "global_confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp_total": int(fp.sum()),
        "fn_total": int(fn.sum()),
        "tn_total": int(tn.sum()),
        "global_accuracy": float(accuracy),
        "global_precision": float(precision),
        "global_recall": float(recall),
        "global_f1": float(f1),
    }


# -------------------------------------------------------
# METRIC WRAPPERS
# -------------------------------------------------------

def metric_accuracy(base_stats):
    return {"accuracy": base_stats["global_accuracy"]}


def metric_precision(base_stats):
    return {"precision": base_stats["global_precision"]}


def metric_recall(base_stats):
    return {"recall": base_stats["global_recall"]}


def metric_f1(base_stats):
    return {"f1": base_stats["global_f1"]}


def metric_overlap(y_true, y_pred):
    true_labels = set(np.unique(y_true))
    pred_labels = set(np.unique(y_pred))
    true_labels.discard(0)
    pred_labels.discard(0)
    union = true_labels | pred_labels
    intersection = true_labels & pred_labels
    return {"overlap": float(len(intersection) / len(union)) if union else float("nan")}


def metric_runtime(runtime_seconds):
    return {"runtime_seconds": runtime_seconds}


def metric_scalability(runtime_seconds, n_items):
    return {
        "scalability_seconds_per_item": (
            float(runtime_seconds / n_items) if n_items else float("nan")
        )
    }


def metric_mcc(y_true, y_pred):
    try:
        return {"mcc": float(matthews_corrcoef(y_true, y_pred))}
    except Exception:
        return {"mcc": float("nan")}


def metric_population_frequency_correlation(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    true_freq = []
    pred_freq = []
    N = len(y_true)

    for label in labels:
        if label == 0:
            continue
        true_freq.append(np.sum(y_true == label) / N)
        pred_freq.append(np.sum(y_pred == label) / N)

    if len(true_freq) < 2:
        return {"popfreq_corr": float("nan")}

    r = np.corrcoef(true_freq, pred_freq)[0, 1]
    return {"popfreq_corr": float(r)}


def metric_aucroc(y_true, y_pred):
    uniq = np.unique(y_true)
    uniq = uniq[uniq != 0]
    try:
        y_true_bin = np.array([y_true == u for u in uniq]).T.astype(int)
        y_pred_bin = np.array([y_pred == u for u in uniq]).T.astype(int)
        auc = roc_auc_score(y_true_bin, y_pred_bin, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")
    return {"aucroc": float(auc)}


# -------------------------------------------------------
# MAIN METRIC COMPUTATION
# -------------------------------------------------------

def compute_prediction_metrics(y_true, y_pred, metrics_to_compute):
    start = time.perf_counter()

    y_true, y_pred = strip_noise_labels(y_true, y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Predicted labels and true labels must match in length.")

    results = {}

    if any(m in CLASSIFICATION_METRICS for m in metrics_to_compute):
        global_stats = compute_global_classification_stats(y_true, y_pred)
        base_stats = {
            "global_accuracy": global_stats["global_accuracy"],
            "global_precision": global_stats["global_precision"],
            "global_recall": global_stats["global_recall"],
            "global_f1": global_stats["global_f1"],
            "global_stats": global_stats,
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

        results["global_stats"] = global_stats

    if "overlap" in metrics_to_compute:
        results.update(metric_overlap(y_true, y_pred))

    if "mcc" in metrics_to_compute:
        results.update(metric_mcc(y_true, y_pred))

    if "popfreq_corr" in metrics_to_compute:
        results.update(metric_population_frequency_correlation(y_true, y_pred))

    if "aucroc" in metrics_to_compute:
        results.update(metric_aucroc(y_true, y_pred))

    runtime_seconds = time.perf_counter() - start

    if "runtime" in metrics_to_compute:
        results.update(metric_runtime(runtime_seconds))

    if "scalability" in metrics_to_compute:
        results.update(metric_scalability(runtime_seconds, y_true.size))

    return results


# -------------------------------------------------------
# MAIN CLI
# -------------------------------------------------------

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

    truth = load_true_labels(getattr(args, "labels_test"))
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
