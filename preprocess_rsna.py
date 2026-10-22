# Converts the RSNA adult DICOM X-rays into 224x224 grayscale PNGs sorted into NORMAL / PNEUMONIA folders.
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image


# Stretch pixel values to 0-255 and flip MONOCHROME1 scans so every image looks consistent.
def normalize_to_uint8(pixel_array: np.ndarray, invert: bool) -> np.ndarray:
    arr = pixel_array.astype(np.float32)
    arr -= arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr /= max_val
    if invert:
        arr = 1.0 - arr
    arr = (arr * 255.0).clip(0, 255)
    return arr.astype(np.uint8)


# Read the RSNA label CSV and build {patientId: 0/1} (1 = pneumonia / lung opacity).
def load_patient_targets(labels_csv_path: Path) -> dict:
    labels_df = pd.read_csv(labels_csv_path)
    if "patientId" not in labels_df.columns or "Target" not in labels_df.columns:
        raise ValueError("CSV must contain 'patientId' and 'Target' columns.")
    targets = labels_df.groupby("patientId", as_index=True)["Target"].max()
    return targets.to_dict()


# Main loop: read each DICOM, normalize, resize to 224x224 and save into the correct class folder.
def convert_dataset(images_dir: Path, labels_csv_path: Path, output_root: Path, size: int) -> None:
    patient_targets = load_patient_targets(labels_csv_path)
    normal_dir = output_root / "NORMAL"
    pneumonia_dir = output_root / "PNEUMONIA"
    normal_dir.mkdir(parents=True, exist_ok=True)
    pneumonia_dir.mkdir(parents=True, exist_ok=True)

    dcm_files = sorted(images_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in: {images_dir}")

    converted = 0
    skipped_unlabeled = 0

    for dcm_path in dcm_files:
        patient_id = dcm_path.stem
        target_value = int(patient_targets.get(patient_id, -1))
        if target_value == -1:
            skipped_unlabeled += 1
            continue

        ds = pydicom.dcmread(str(dcm_path))
        pixel_array = ds.pixel_array
        invert = getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1"

        img_uint8 = normalize_to_uint8(pixel_array, invert=invert)
        img = Image.fromarray(img_uint8).convert("L")
        img = img.resize((size, size), Image.Resampling.BILINEAR)

        class_dir = pneumonia_dir if target_value == 1 else normal_dir
        out_path = class_dir / f"{patient_id}.png"
        img.save(out_path, format="PNG")
        converted += 1

    print(f"Converted: {converted}")
    print(f"Skipped unlabeled: {skipped_unlabeled}")
    print(f"NORMAL count: {len(list(normal_dir.glob('*.png')))}")
    print(f"PNEUMONIA count: {len(list(pneumonia_dir.glob('*.png')))}")


# Command-line options (input folder, labels CSV, output folder, image size).
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert RSNA stage_2_train_images DICOM files to PNG and split by label."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("stage_2_train_images"),
        help="Directory containing RSNA .dcm files.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("stage_2_train_labels.csv") / "stage_2_train_labels.csv",
        help="Path to stage_2_train_labels.csv file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed_dataset"),
        help="Output folder that will contain NORMAL and PNEUMONIA subfolders.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Square resize dimension in pixels (default: 224).",
    )
    return parser


# Entry point.
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    convert_dataset(args.images_dir, args.labels_csv, args.output_root, args.size)


if __name__ == "__main__":
    main()