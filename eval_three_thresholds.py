"""
eval_three_thresholds.py
------------------------
Runs the trained all-ages model once over the adult test set, pediatric test
set and pooled validation set, then scores the SAME predicted probabilities
at the three decision thresholds compared in the paper (0.36, 0.08, 0.18).

Usage:
    python eval_three_thresholds.py
"""

# Scores the trained model on validation + both test sets at the three compared thresholds (0.36, 0.08, 0.18).
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

import train_scaffold_updated as ts

# Model checkpoint, the datasets to score, and the thresholds to compare.
MODEL_PATH = Path("allages_run/federated_model_scaffold_v4_allages.pth")
SETS = {
    "Validation (pooled)": Path("dataset_v4/validation"),
    "Test - Adult": Path("dataset_v4/test_adult"),
    "Test - Pediatric": Path("dataset_v4/test_pediatric"),
}
THRESHOLDS = [
    (0.36, "precision-favoring (fbeta)"),
    (0.08, "pooled F1"),
    (0.18, "balanced (max-min F1)"),
]


# Run the model once and return predicted probabilities and true labels.
def collect(model, loader):
    probs, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            images = images.to(ts.DEVICE)
            probs.extend(torch.sigmoid(model(images).squeeze(1)).cpu().numpy())
            labels.extend(y.numpy())
    return np.asarray(probs), np.asarray(labels)


# Score each dataset once, then apply every threshold to the same probabilities and print the tables.
def main():
    model = ts.build_model(pretrained=False).to(ts.DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=ts.DEVICE))
    model.eval()
    _, test_tf = ts.build_transforms()

    results = {}
    for name, root in SETS.items():
        loader = ts.build_validation_loader_from_dir(root, test_tf, 16)
        probs, labels = collect(model, loader)
        print(f"{name}: {len(labels)} images scored", flush=True)
        for thr, _ in THRESHOLDS:
            preds = (probs >= thr).astype(int)
            results[(name, thr)] = {
                "acc": accuracy_score(labels, preds) * 100,
                "prec": precision_score(labels, preds, zero_division=0) * 100,
                "rec": recall_score(labels, preds, zero_division=0) * 100,
                "f1": f1_score(labels, preds, zero_division=0) * 100,
                "cm": confusion_matrix(labels, preds).tolist(),
            }

    print("\n" + "=" * 100)
    for thr, method in THRESHOLDS:
        print(f"\nTHRESHOLD {thr:.2f}  --  {method}")
        print(f"{'Dataset':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}   Confusion matrix")
        for name in SETS:
            r = results[(name, thr)]
            print(f"{name:<22}{r['acc']:>9.2f}%{r['prec']:>10.2f}%{r['rec']:>8.2f}%{r['f1']:>8.2f}%   {r['cm']}")
    print("\n" + "=" * 100)

    out = {f"{n} @ {t}": v for (n, t), v in results.items()}
    Path("allages_run/three_threshold_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved -> allages_run/three_threshold_results.json")


if __name__ == "__main__":
    main()
