"""
retune_threshold_allages.py
--------------------------------
Re-tunes the decision threshold for the already-trained all-ages model
(allages_run/federated_model_scaffold_v4_allages.pth) using a different
threshold-selection metric, WITHOUT retraining.

The original training run used --threshold-metric fbeta_precision (the
project default), which explicitly weights precision over recall -- fine
for the adult-only baseline, but it produced a very conservative pediatric
threshold (precision 99.8%, recall 73.4%): the model rarely raises a false
alarm on healthy kids, but misses over a quarter of actual pediatric
pneumonia cases. For a screening tool, a missed-disease error is worse than
a false alarm, so this re-tunes with `f1` (balanced) instead, using the
exact same pooled validation set the original training run used, and
reports both test sets again with the new threshold for direct comparison.

Usage:
    python retune_threshold_allages.py --metric f1
    python retune_threshold_allages.py --metric recall
"""

# Re-picks the decision threshold of the already-trained model using a different metric (no retraining).
import argparse
import json
from pathlib import Path

import torch

import train_scaffold_updated as ts


# Load the saved model, sweep thresholds on the pooled validation set, then score both test sets at the chosen threshold.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path,
                         default=Path("allages_run/federated_model_scaffold_v4_allages.pth"))
    parser.add_argument("--validation-root", type=Path, default=Path("dataset_v4/validation"))
    parser.add_argument("--test-root", type=Path, default=Path("dataset_v4/test_adult"))
    parser.add_argument("--test-root-2", type=Path, default=Path("dataset_v4/test_pediatric"))
    parser.add_argument("--metric", type=str, default="f1",
                         choices=["accuracy", "precision", "recall", "f1", "fbeta_precision"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-json", type=Path, default=None,
                         help="If set, writes the re-tuned results here in the same format as "
                              "scaffold_final_metrics.json (with a nested 'pediatric' block).")
    args = parser.parse_args()

    # Load the trained global model.
    model = ts.build_model(pretrained=False).to(ts.DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=ts.DEVICE))
    model.eval()

    _, test_tf = ts.build_transforms()
    validation_loader = ts.build_validation_loader_from_dir(args.validation_root, test_tf, args.batch_size)

    # Sweep thresholds on validation data using the metric given on the command line.
    best_threshold, val_metrics = ts.find_best_threshold(model, validation_loader, metric=args.metric)
    print(f"Re-tuned with metric='{args.metric}':")
    print(
        f"  best_threshold={best_threshold:.2f} "
        f"val_accuracy={val_metrics['accuracy']:.4f} "
        f"val_precision={val_metrics['precision']:.4f} "
        f"val_recall={val_metrics['recall']:.4f} "
        f"val_f1={val_metrics['f1']:.4f}"
    )

    # Evaluate on the adult and pediatric test sets separately.
    adult_acc, adult_prec, adult_rec, adult_f1 = ts.evaluate_on_test_set(
        args.model_path, args.test_root, batch_size=args.batch_size,
        threshold=best_threshold, label="adult"
    )
    ped_acc, ped_prec, ped_rec, ped_f1 = ts.evaluate_on_test_set(
        args.model_path, args.test_root_2, batch_size=args.batch_size,
        threshold=best_threshold, label="pediatric"
    )

    # Optionally save the results to JSON in the same format the app reads.
    if args.output_json is not None:
        result = {
            "accuracy": adult_acc,
            "precision": adult_prec,
            "recall": adult_rec,
            "f1_score": adult_f1,
            "model_path": str(args.model_path),
            "threshold": best_threshold,
            "threshold_metric": args.metric,
            "pediatric": {
                "accuracy": ped_acc,
                "precision": ped_prec,
                "recall": ped_rec,
                "f1_score": ped_f1,
                "test_root": str(args.test_root_2),
            },
        }
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
