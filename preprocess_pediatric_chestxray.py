"""
preprocess_pediatric_chestxray.py
--------------------------------
Converts the full Kermany pediatric chest X-ray dataset (chest_xray/, both
NORMAL and PNEUMONIA -- unlike preprocess_kaggle_pneumonia.py, which only
pulled PNEUMONIA to top up the adult set) into a standalone pediatric pool
with its own train/test split.

Kermany's own train/test/val split is not used, since it has documented
patient-level leakage (the project's own preprocess_kaggle_pneumonia.py
already notes this). Instead, all three original splits are pooled together
and re-split here at the PATIENT level, so a patient's images never appear
in both train and test.

Patient grouping:
  - PNEUMONIA filenames: "person1000_bacteria_2931.jpeg" -> group "1000"
  - NORMAL filenames:    "IM-0115-0001.jpeg"              -> group "0115"
  (the two numbering schemes are independent, so groups are namespaced by
  class to avoid an accidental collision.)

Output layout mirrors dataset_v2 so it can be merged in later:
    pediatric_pool/train/NORMAL/*.png
    pediatric_pool/train/PNEUMONIA/*.png
    pediatric_pool/validation/NORMAL/*.png
    pediatric_pool/validation/PNEUMONIA/*.png
    pediatric_pool/test/NORMAL/*.png
    pediatric_pool/test/PNEUMONIA/*.png

`train/` is meant to become the new pediatric federated client's data.
`validation/` is meant to be POOLED with the existing adult validation set
(dataset_v2/validation) for round-selection and threshold-tuning, so those
choices account for pediatric performance too, since a pediatric client is
now part of training.
`test/` is a held-out set, never trained on and never used for
round/threshold selection -- used only to report pediatric performance
separately from the adult test set.

Usage:
    python preprocess_pediatric_chestxray.py
"""

# Converts the Kermany pediatric dataset (NORMAL + PNEUMONIA) into 224x224 PNGs and splits it by PATIENT into train / validation / test.
import argparse
import re
import random
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Filename patterns used to recover the patient number (person123_... / IM-0123-...).
PNEUMONIA_PATTERN = re.compile(r"^person(\d+)_")
NORMAL_PATTERN = re.compile(r"^IM-(\d+)-")


# Returns a patient-group key so all images of one patient stay in the same split (prevents data leakage).
def group_key(class_name: str, filename: str) -> str:
    if class_name == "PNEUMONIA":
        m = PNEUMONIA_PATTERN.match(filename)
    else:
        m = NORMAL_PATTERN.match(filename)
    ident = m.group(1) if m else filename  # fall back to filename itself if pattern doesn't match
    return f"{class_name}_{ident}"


# Pool the images of one class from Kermany's original train/test/val folders (its own split is leaky).
def collect_files(source_root: Path, class_name: str) -> list:
    files = []
    for split in ["train", "test", "val"]:
        d = source_root / split / class_name
        if d.exists():
            files.extend(sorted(p for p in d.iterdir() if p.is_file()))
    return files


# Shuffle patients (fixed seed) and assign whole patients to test, then validation, then train.
def split_by_patient(files: list, test_fraction: float, val_fraction: float, seed: int, class_name: str) -> tuple:
    groups = {}
    for f in files:
        key = group_key(class_name, f.name)
        groups.setdefault(key, []).append(f)

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_test_groups = max(1, round(len(group_keys) * test_fraction))
    n_val_groups = max(1, round(len(group_keys) * val_fraction))
    test_groups = set(group_keys[:n_test_groups])
    val_groups = set(group_keys[n_test_groups:n_test_groups + n_val_groups])

    train_files, val_files, test_files = [], [], []
    for key, group_files in groups.items():
        if key in test_groups:
            test_files.extend(group_files)
        elif key in val_groups:
            val_files.extend(group_files)
        else:
            train_files.extend(group_files)

    return train_files, val_files, test_files, len(group_keys), len(test_groups), len(val_groups)


# Open each image, convert to grayscale, resize and save as PNG.
def convert_and_save(files: list, output_dir: Path, size: int) -> tuple:
    output_dir.mkdir(parents=True, exist_ok=True)
    converted, skipped = 0, 0
    for src_path in files:
        try:
            img = Image.open(src_path).convert("L")
        except (UnidentifiedImageError, OSError):
            skipped += 1
            continue
        img = img.resize((size, size), Image.Resampling.BILINEAR)
        out_name = f"{src_path.stem}.png"
        img.save(output_dir / out_name, format="PNG")
        converted += 1
    return converted, skipped


# Run the full pipeline for both classes and print the final split sizes.
def build(source_root: Path, output_root: Path, size: int, test_fraction: float, val_fraction: float, seed: int) -> None:
    print("=" * 80)
    print("PEDIATRIC CHEST X-RAY PREPROCESSING (Kermany dataset, full re-split)")
    print("=" * 80)

    for class_name in ["NORMAL", "PNEUMONIA"]:
        files = collect_files(source_root, class_name)
        print(f"\n{class_name}: {len(files)} source images pooled from train/test/val")

        train_files, val_files, test_files, n_groups, n_test_groups, n_val_groups = split_by_patient(
            files, test_fraction, val_fraction, seed, class_name
        )
        print(f"  Patients: {n_groups} total -> test {n_test_groups}, validation {n_val_groups}, "
              f"train {n_groups - n_test_groups - n_val_groups} "
              f"({len(train_files)} train images / {len(val_files)} validation images / {len(test_files)} test images)")

        train_converted, train_skipped = convert_and_save(
            train_files, output_root / "train" / class_name, size
        )
        val_converted, val_skipped = convert_and_save(
            val_files, output_root / "validation" / class_name, size
        )
        test_converted, test_skipped = convert_and_save(
            test_files, output_root / "test" / class_name, size
        )
        print(f"  Converted -> train: {train_converted} (skipped {train_skipped}), "
              f"validation: {val_converted} (skipped {val_skipped}), "
              f"test: {test_converted} (skipped {test_skipped})")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for split in ["train", "validation", "test"]:
        normal_n = len(list((output_root / split / "NORMAL").glob("*.png")))
        pneu_n = len(list((output_root / split / "PNEUMONIA").glob("*.png")))
        total = normal_n + pneu_n
        pct = pneu_n / total * 100 if total else 0
        print(f"  {split}: NORMAL={normal_n} PNEUMONIA={pneu_n} Total={total} (Pneumonia {pct:.1f}%)")
    print(f"\nOutput written to: {output_root}")


# Command-line options (source/output folders, size, test/validation fractions, seed).
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess the full Kermany pediatric dataset (both classes) into a standalone train/test pool."
    )
    parser.add_argument("--source-root", type=Path, default=Path("chest_xray"))
    parser.add_argument("--output-root", type=Path, default=Path("pediatric_pool"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--test-fraction", type=float, default=0.2,
                         help="Fraction of PATIENTS (not images) held out for the pediatric test set.")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                         help="Fraction of PATIENTS (not images) held out for the pediatric validation set.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


# Entry point.
def main() -> None:
    args = build_parser().parse_args()
    build(args.source_root, args.output_root, args.size, args.test_fraction, args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
