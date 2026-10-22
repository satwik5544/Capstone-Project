"""
merge_pediatric_client.py
---------------------------
Builds an all-ages federated dataset (dataset_v4) by adding the pediatric
Kermany data (pediatric_pool/, from preprocess_pediatric_chestxray.py) as a
new 4th federated client alongside the existing 3 adult RSNA clients from
dataset_v2.

Layout produced:
    dataset_v4/train/client_0/{NORMAL,PNEUMONIA}   <- copied unchanged from dataset_v2 (adult)
    dataset_v4/train/client_1/{NORMAL,PNEUMONIA}   <- copied unchanged from dataset_v2 (adult)
    dataset_v4/train/client_2/{NORMAL,PNEUMONIA}   <- copied unchanged from dataset_v2 (adult)
    dataset_v4/train/client_3/{NORMAL,PNEUMONIA}   <- NEW, pediatric_pool/train (pediatric)
    dataset_v4/validation/{NORMAL,PNEUMONIA}       <- dataset_v2/validation + pediatric_pool/validation, POOLED
    dataset_v4/test_adult/{NORMAL,PNEUMONIA}       <- copied unchanged from dataset_v2/test
    dataset_v4/test_pediatric/{NORMAL,PNEUMONIA}   <- copied unchanged from pediatric_pool/test

Validation is pooled (not kept separate) because client_3 is now part of
training -- round-selection and threshold-tuning should account for
pediatric performance too, not just adult. Test sets are kept SEPARATE
(not pooled) so final results can be reported per age group, which is the
actual evidence for an "all ages" claim -- see project discussion. A
combined test set is trivial to produce later (just concatenate the two)
if an overall pooled number is also wanted.

Usage:
    python merge_pediatric_client.py
"""

# Builds dataset_v4: the 3 adult clients + the pediatric data as a new 4th client, with pooled validation and two separate test sets.
import argparse
import shutil
from pathlib import Path

# Names of the existing adult client folders and the two classes.
ADULT_CLIENTS = ["client_0", "client_1", "client_2"]
CLASSES = ["NORMAL", "PNEUMONIA"]


# Copy a folder, replacing it if it already exists.
def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# Copy every file of a folder into another folder (used to pool validation images).
def _copy_files_into(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, dst_dir / f.name)
            n += 1
    return n


# Build dataset_v4: copy adult clients, add pediatric client_3, pool validation, keep test sets separate.
def build(v2_root: Path, pediatric_root: Path, output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    print("=" * 80)
    print("BUILDING ALL-AGES FEDERATED DATASET (dataset_v4)")
    print("=" * 80)

    # (Adult clients are copied unchanged from dataset_v2.)
    # --- adult clients 0-2: copied unchanged ---
    for client in ADULT_CLIENTS:
        _copy_tree(v2_root / "train" / client, output_root / "train" / client)
        normal_n = len(list((output_root / "train" / client / "NORMAL").glob("*.png")))
        pneu_n = len(list((output_root / "train" / client / "PNEUMONIA").glob("*.png")))
        print(f"  {client} (adult, unchanged): NORMAL={normal_n} PNEUMONIA={pneu_n}")

    # --- new pediatric client_3 ---
    _copy_tree(pediatric_root / "train", output_root / "train" / "client_3")
    normal_n = len(list((output_root / "train" / "client_3" / "NORMAL").glob("*.png")))
    pneu_n = len(list((output_root / "train" / "client_3" / "PNEUMONIA").glob("*.png")))
    print(f"  client_3 (pediatric, NEW): NORMAL={normal_n} PNEUMONIA={pneu_n}")

    # --- validation: pooled adult + pediatric ---
    for cls in CLASSES:
        dst = output_root / "validation" / cls
        n_adult = _copy_files_into(v2_root / "validation" / cls, dst)
        n_pediatric = _copy_files_into(pediatric_root / "validation" / cls, dst)
        print(f"  validation/{cls}: {n_adult} adult + {n_pediatric} pediatric = {n_adult + n_pediatric}")

    # --- test sets: kept SEPARATE for stratified reporting ---
    _copy_tree(v2_root / "test", output_root / "test_adult")
    _copy_tree(pediatric_root / "test", output_root / "test_pediatric")
    for split in ["test_adult", "test_pediatric"]:
        normal_n = len(list((output_root / split / "NORMAL").glob("*.png")))
        pneu_n = len(list((output_root / split / "PNEUMONIA").glob("*.png")))
        print(f"  {split}: NORMAL={normal_n} PNEUMONIA={pneu_n}")

    print("\n" + "=" * 80)
    print("FINAL PER-CLIENT TRAINING TOTALS")
    print("=" * 80)
    for client in ADULT_CLIENTS + ["client_3"]:
        normal_n = len(list((output_root / "train" / client / "NORMAL").glob("*.png")))
        pneu_n = len(list((output_root / "train" / client / "PNEUMONIA").glob("*.png")))
        total = normal_n + pneu_n
        print(f"  {client}: NORMAL={normal_n} PNEUMONIA={pneu_n} Total={total} "
              f"(Pneumonia {pneu_n/total:.1%})")

    print(f"\nOutput written to: {output_root}")


# Command-line options (input and output folders).
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge the pediatric pool into dataset_v2 as a new 4th federated client.")
    parser.add_argument("--v2-root", type=Path, default=Path("dataset_v2"))
    parser.add_argument("--pediatric-root", type=Path, default=Path("pediatric_pool"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset_v4"))
    return parser


# Entry point.
def main() -> None:
    args = build_parser().parse_args()
    build(args.v2_root, args.pediatric_root, args.output_root)


if __name__ == "__main__":
    main()
