"""
balanced_threshold_allages.py
--------------------------------
Finds a decision threshold for the already-trained all-ages model that
balances adult and pediatric performance, instead of optimizing one pooled
validation metric (which is what both prior attempts did):

  - fbeta_precision on pooled validation -> threshold 0.36: good adult
    precision, but pediatric recall collapsed to 73%.
  - f1 on pooled validation -> threshold 0.08: fixed pediatric (95% F1),
    but adult precision collapsed to 72% because the pooled metric let
    pediatric's steeper threshold-sensitivity dominate the choice.

This script instead computes ADULT-only and PEDIATRIC-only validation F1
separately at every threshold (using the original, unpooled validation
folders: dataset_v2/validation and pediatric_pool/validation), then picks
the threshold that maximizes the MINIMUM of the two -- i.e. the best
worst-case tradeoff, so neither group gets sacrificed for the other. No
retraining needed; reuses the already-trained checkpoint.

Usage:
    python balanced_threshold_allages.py
"""

# Finds the threshold that gives the best WORST-CASE F1 across adult and pediatric validation data (no retraining).
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import train_scaffold_updated as ts


# Run the model over a loader and return predicted pneumonia probabilities and true labels.
def collect_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(ts.DEVICE, non_blocking=True)
            logits = model(images).squeeze(1)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.asarray(all_probs), np.asarray(all_labels)


# Compute probabilities per group, sweep thresholds, pick the max-min F1 threshold, then evaluate both test sets.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path,
                         default=Path("allages_run/federated_model_scaffold_v4_allages.pth"))
    parser.add_argument("--adult-validation-root", type=Path, default=Path("dataset_v2/validation"))
    parser.add_argument("--pediatric-validation-root", type=Path, default=Path("pediatric_pool/validation"))
    parser.add_argument("--test-root", type=Path, default=Path("dataset_v4/test_adult"))
    parser.add_argument("--test-root-2", type=Path, default=Path("dataset_v4/test_pediatric"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    model = ts.build_model(pretrained=False).to(ts.DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=ts.DEVICE))
    model.eval()

    _, test_tf = ts.build_transforms()
    adult_val_loader = ts.build_validation_loader_from_dir(args.adult_validation_root, test_tf, args.batch_size)
    ped_val_loader = ts.build_validation_loader_from_dir(args.pediatric_validation_root, test_tf, args.batch_size)

    adult_probs, adult_labels = collect_probs(model, adult_val_loader)
    ped_probs, ped_labels = collect_probs(model, ped_val_loader)

    print(f"Adult validation: {len(adult_labels)} images | Pediatric validation: {len(ped_labels)} images\n")
    print(f"{'threshold':>10} {'adult_f1':>10} {'ped_f1':>10} {'min(f1)':>10} {'avg(f1)':>10}")

    best_threshold = 0.5
    best_min_f1 = -1.0
    rows = []
    # For each candidate threshold compute adult F1 and pediatric F1 separately; keep the best min(adult, pediatric).
    for threshold in np.linspace(0.05, 0.95, 91):
        adult_preds = (adult_probs >= threshold).astype(np.int64)
        ped_preds = (ped_probs >= threshold).astype(np.int64)
        adult_f1 = f1_score(adult_labels, adult_preds, zero_division=0)
        ped_f1 = f1_score(ped_labels, ped_preds, zero_division=0)
        min_f1 = min(adult_f1, ped_f1)
        rows.append((threshold, adult_f1, ped_f1, min_f1))
        if min_f1 > best_min_f1:
            best_min_f1 = min_f1
            best_threshold = threshold

    # Print every 5th row plus the winner for a compact view of the curve.
    for i, (threshold, adult_f1, ped_f1, min_f1) in enumerate(rows):
        if i % 5 == 0 or abs(threshold - best_threshold) < 1e-9:
            marker = "  <-- best" if abs(threshold - best_threshold) < 1e-9 else ""
            print(f"{threshold:10.2f} {adult_f1:10.4f} {ped_f1:10.4f} {min_f1:10.4f}{marker}")

    print(f"\nSelected balanced threshold: {best_threshold:.2f} (maximizes min(adult_f1, pediatric_f1))")

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
            "accuracy": adult_acc, "precision": adult_prec, "recall": adult_rec, "f1_score": adult_f1,
            "model_path": str(args.model_path),
            "threshold": best_threshold,
            "threshold_metric": "balanced (max-min of adult/pediatric validation F1)",
            "pediatric": {
                "accuracy": ped_acc, "precision": ped_prec, "recall": ped_rec, "f1_score": ped_f1,
                "test_root": str(args.test_root_2),
            },
        }
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
