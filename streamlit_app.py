# ---- Streamlit web demo: upload a chest X-ray -> model prediction -> radiology-style PDF report ----
import io
import json
import re
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image, ImageOps

# Reuse the model class and constants from the training script so training and inference stay identical.
import train_scaffold_updated as ts


# Configure the browser tab title, icon and wide page layout.
st.set_page_config(
    page_title="Federated Learning for Pneumonia Detection",
    page_icon="🩺",
    layout="wide",
)


# Placeholder model performance values (requested).
# Fallback numbers shown only if the metrics JSON file cannot be read.
PLACEHOLDER_METRICS = {
    "accuracy": "77.42%",
    "precision": "78.07%",
    "recall": "76.25%",
    "f1": "77.15%",
    "pediatric_accuracy": "--",
    "pediatric_precision": "--",
    "pediatric_recall": "--",
    "pediatric_f1": "--",
}

# The trained global model file the app loads (change this to switch models).
MODEL_PATH = Path("allages_run/federated_model_scaffold_v4_allages.pth")
# JSON with the final metrics and the decision threshold used by the app.
FINAL_METRICS_PATH = Path("allages_run/scaffold_final_metrics_balanced.json")


# Returns the metrics file's modified time so cached values refresh when the file changes.
def _metrics_file_mtime() -> float:
    return FINAL_METRICS_PATH.stat().st_mtime if FINAL_METRICS_PATH.exists() else 0.0


# Reads accuracy / precision / recall / F1 (adult and pediatric) from the metrics JSON for display.
def load_display_metrics() -> dict:
    """Load model metrics for UI/report display, fallback to placeholders.

    Not cached: this is a tiny JSON read, and metrics must always reflect
    whatever scaffold_final_metrics.json currently holds -- a stale cache
    here previously kept showing a discarded experiment's numbers even
    after the file was fixed on disk."""
    metrics = PLACEHOLDER_METRICS.copy()
    if not FINAL_METRICS_PATH.exists():
        return metrics

    try:
        data = json.loads(FINAL_METRICS_PATH.read_text(encoding="utf-8"))
        metrics["accuracy"] = f"{float(data.get('accuracy', 0.0)) * 100:.2f}%"
        metrics["precision"] = f"{float(data.get('precision', 0.0)) * 100:.2f}%"
        metrics["recall"] = f"{float(data.get('recall', 0.0)) * 100:.2f}%"
        metrics["f1"] = f"{float(data.get('f1_score', 0.0)) * 100:.2f}%"

        pediatric = data.get("pediatric")
        if pediatric:
            metrics["pediatric_accuracy"] = f"{float(pediatric.get('accuracy', 0.0)) * 100:.2f}%"
            metrics["pediatric_precision"] = f"{float(pediatric.get('precision', 0.0)) * 100:.2f}%"
            metrics["pediatric_recall"] = f"{float(pediatric.get('recall', 0.0)) * 100:.2f}%"
            metrics["pediatric_f1"] = f"{float(pediatric.get('f1_score', 0.0)) * 100:.2f}%"
    except Exception:
        pass

    return metrics


# Load the display metrics once when the app starts.
DISPLAY_METRICS = load_display_metrics()
HAS_PEDIATRIC_METRICS = DISPLAY_METRICS["pediatric_accuracy"] != "--"


# Loads the trained DenseNet121 weights, the decision threshold and the preprocessing pipeline (cached so it is not reloaded on every click).
@st.cache_resource(show_spinner=False)
def load_inference_artifacts(_cache_key: tuple):
    """Load trained model and threshold. Cached on (model mtime, metrics
    mtime) so a checkpoint/metrics swap is picked up automatically instead
    of requiring a full server restart, while still avoiding a reload of
    the ~28MB checkpoint on every script rerun."""
    threshold = 0.5
    if FINAL_METRICS_PATH.exists():
        try:
            data = json.loads(FINAL_METRICS_PATH.read_text(encoding="utf-8"))
            threshold = float(data.get("threshold", 0.5))
        except Exception:
            threshold = 0.5

    model = ts.DenseNet121Custom(pretrained=False)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Inference preprocessing: must exactly match training (grayscale -> 224x224 -> tensor -> ImageNet normalize).
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((ts.INPUT_SIZE, ts.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, tf, threshold


# Deterministic test-time-augmentation views: averaging predictions over
# these measurably reduced test-set error versus a single forward pass
# (80.79% -> 81.08% accuracy) and makes any one prediction less sensitive
# to a single unlucky crop/orientation/contrast reading.
# Test-time augmentation: 5 slightly different views of the same image; their predictions are averaged.
_TTA_VIEWS = [
    lambda img: img,
    lambda img: TF.hflip(img),
    lambda img: TF.rotate(img, 5),
    lambda img: TF.rotate(img, -5),
    lambda img: ImageOps.autocontrast(img),
]

# Predictions with confidence below this value show a 'borderline' warning.
LOW_CONFIDENCE_THRESHOLD = 0.65


# Runs the model on one uploaded X-ray (5 TTA views) and returns label, confidence and class probabilities.
def predict_image(image_bytes: bytes) -> tuple[str, float, float, float]:
    """Run DenseNet121 inference with 5-view test-time augmentation and
    return label, confidence, normal_prob, pneumonia_prob."""
    cache_key = (
        MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0,
        _metrics_file_mtime(),
    )
    model, tf, threshold = load_inference_artifacts(cache_key)

    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    view_probs = []
    with torch.no_grad():
        for view_fn in _TTA_VIEWS:
            tensor = tf(view_fn(image)).unsqueeze(0)
            logits = model(tensor)
            view_probs.append(torch.sigmoid(logits).item())

    pneumonia_prob = float(np.mean(view_probs))
    normal_prob = 1.0 - pneumonia_prob

    # Apply the decision threshold to turn the pneumonia probability into Normal / Pneumonia.
    is_pneumonia = pneumonia_prob >= threshold
    label = "Pneumonia" if is_pneumonia else "Normal"
    confidence = pneumonia_prob if is_pneumonia else normal_prob
    return label, float(confidence), float(normal_prob), float(pneumonia_prob)


# Option lists and suggested report text used by the report form.
STANDARD_SYMPTOMS = [
    "Cough", "Fever", "Shortness of breath", "Chest pain",
    "Fatigue", "Sputum production", "Chills", "Loss of appetite",
]

TECHNIQUE_OPTIONS = [
    "Posteroanterior (PA) chest radiograph",
    "Anteroposterior (AP) chest radiograph",
    "Portable AP chest radiograph",
    "Lateral chest radiograph",
]

FINDINGS_SUGGESTED = {
    "Pneumonia": [
        "Patchy air-space opacities suggestive of consolidation",
        "Peribronchial thickening",
        "Possible infective infiltrate in the affected lung field(s)",
    ],
    "Normal": [
        "No focal air-space consolidation",
        "No pleural effusion",
        "No pneumothorax",
    ],
}

RECOMMENDATIONS_SUGGESTED = {
    "Pneumonia": [
        "Clinical correlation with vital signs, oxygen saturation, and laboratory parameters is recommended.",
        "Initiate appropriate medical management as per physician guidance.",
        "Follow-up chest imaging is advised to assess progression or resolution.",
        "Urgent clinical evaluation is recommended if symptoms worsen (e.g., increased breathlessness, "
        "persistent high fever, low oxygen saturation).",
    ],
    "Normal": [
        "Clinical correlation recommended based on patient symptoms.",
        "No immediate radiological follow-up required if asymptomatic.",
        "If symptoms (fever, cough, breathlessness) persist or worsen, consider repeat imaging.",
        "Routine medical consultation advised if clinically indicated.",
    ],
}

# Regular expressions used to validate the form fields.
PHONE_RE = re.compile(r"^[+0-9][0-9\-\s]{6,14}$")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]{1,79}$")
ID_RE = re.compile(r"^[0-9]{1,15}$")
LICENSE_RE = re.compile(r"^[A-Za-z0-9\-\/]{4,20}$")


# Validates the patient / study form fields and returns an error message for each invalid field.
def validate_patient_form(values: dict, today) -> dict:
    """Return {field_key: error_message} for every invalid required field.
    An empty dict means the form is ready for prediction/report generation."""
    errors: dict[str, str] = {}

    if not values["patient_id"].strip():
        errors["patient_id"] = "Patient ID is required."
    elif not ID_RE.match(values["patient_id"].strip()):
        errors["patient_id"] = "Enter a valid number."

    if not values["patient_name"].strip():
        errors["patient_name"] = "Patient name is required."
    elif not NAME_RE.match(values["patient_name"].strip()):
        errors["patient_name"] = "Enter a valid name (letters, spaces, - or ' only)."

    if values["patient_gender"] == "Select":
        errors["patient_gender"] = "Please select a gender."

    if not values["patient_phone"].strip():
        errors["patient_phone"] = "Phone number is required."
    elif not PHONE_RE.match(values["patient_phone"].strip()):
        errors["patient_phone"] = "Enter a valid phone number (7-15 digits)."

    if values["patient_dob"] >= today:
        errors["patient_dob"] = "Date of birth must be in the past."
    elif (today - values["patient_dob"]).days > 120 * 365:
        errors["patient_dob"] = "Date of birth is out of a plausible range."

    if values["patient_report_date"] > today:
        errors["patient_report_date"] = "Study date cannot be in the future."
    elif values["patient_report_date"] < values["patient_dob"]:
        errors["patient_report_date"] = "Study date cannot be before date of birth."

    if values.get("comparison_has_prior") and values.get("comparison_date") and values["comparison_date"] > today:
        errors["comparison_date"] = "Prior study date cannot be in the future."

    if not values["radiologist_name"].strip():
        errors["radiologist_name"] = "Radiologist name is required to sign off."
    elif not NAME_RE.match(values["radiologist_name"].strip()):
        errors["radiologist_name"] = "Enter a valid name (letters, spaces, - or ' only)."

    if not values["radiologist_license"].strip():
        errors["radiologist_license"] = "License number is required to sign off."
    elif not LICENSE_RE.match(values["radiologist_license"].strip()):
        errors["radiologist_license"] = "Use 4-20 letters, numbers, - or / only."

    return errors


# Escapes special characters so text can be written safely into the hand-built PDF.
def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


# Minimal PDF writer for a plain text report.
def build_simple_pdf(report_lines: list[str]) -> bytes:
    content_lines = ["BT", "/F1 12 Tf", "50 760 Td"]
    for idx, line in enumerate(report_lines):
        if idx > 0:
            content_lines.append("0 -18 Td")
        content_lines.append(f"({_escape_pdf_text(line)}) Tj")
    content_lines.append("ET")

    stream_data = "\n".join(content_lines).encode("latin-1", "replace")

    objects = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objects.append(
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\n"
        b"endobj\n"
    )
    objects.append(
        f"4 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode("ascii")
        + stream_data
        + b"\nendstream\nendobj\n"
    )
    objects.append(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    xref_start = len(pdf)
    count = len(objects) + 1

    pdf.extend(f"xref\n0 {count}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")

    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    pdf.extend(
        (
            f"trailer\n<< /Size {count} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF"
        ).encode("ascii")
    )

    return bytes(pdf)


def _pdf_text(x: float, y: float, text: str, size: int = 11, font: str = "F1") -> str:
    return f"0 0 0 rg BT /{font} {size} Tf 1 0 0 1 {x:.2f} {y:.2f} Tm ({_escape_pdf_text(text)}) Tj ET"


def _wrap_for_pdf(text: str, width: int = 95) -> list[str]:
    clean = " ".join((text or "").split())
    if not clean:
        return ["Not Provided"]
    return textwrap.wrap(clean, width=width)


# Letterhead text used at the top of the generated radiology report.
FACILITY_NAME = "RSNA FEDERATED RADIOLOGY NETWORK"
FACILITY_SUBTITLE = "Diagnostic Imaging & Radiology Reporting"
FACILITY_ADDRESS = "Federated Multi-Site Imaging Program  |  Report generated electronically"


# Builds the one-page radiology-style PDF report, with the AI output kept in a separate addendum.
def build_radiology_pdf(report: dict) -> bytes:
    """Lay out a single-page report in the same section order and tone as a
    real hospital radiology report (letterhead, demographics, EXAM title,
    Clinical History, Comparison, Technique, Findings, Impression,
    Recommendations, e-signature) with the AI system's own output kept in a
    clearly separated addendum at the end, rather than mixed into the
    clinical narrative as if a radiologist wrote it."""
    cmds: list[str] = []
    y = 780.0
    PAGE_LEFT, PAGE_RIGHT, PAGE_WIDTH = 36, 576, 540
    MIN_Y = 50.0

    def draw_filled_rect(x: float, y_bottom: float, w: float, h: float, r: float, g: float, b: float):
        cmds.append(f"{r:.3f} {g:.3f} {b:.3f} rg")
        cmds.append(f"{x:.2f} {y_bottom:.2f} {w:.2f} {h:.2f} re f")

    def draw_rect(x: float, y_bottom: float, w: float, h: float, r=0.760, g=0.820, b=0.900):
        cmds.append(f"{r:.3f} {g:.3f} {b:.3f} RG")
        cmds.append("1 w")
        cmds.append(f"{x:.2f} {y_bottom:.2f} {w:.2f} {h:.2f} re S")

    def hline(y_line: float, r=0.820, g=0.860, b=0.920):
        cmds.append(f"{r:.3f} {g:.3f} {b:.3f} RG")
        cmds.append("1 w")
        cmds.append(f"{PAGE_LEFT} {y_line:.2f} m {PAGE_RIGHT} {y_line:.2f} l S")

    def section_title(title: str, tint=(0.945, 0.965, 0.992)):
        nonlocal y
        draw_filled_rect(PAGE_LEFT, y - 16, PAGE_WIDTH, 16, *tint)
        draw_rect(PAGE_LEFT, y - 16, PAGE_WIDTH, 16)
        cmds.append(_pdf_text(44, y - 12, title, size=10, font="F2"))
        y -= 22

    def body_lines(text: str, width: int = 100, prefix1: str = "", prefix_n: str = ""):
        nonlocal y
        for idx, line in enumerate(_wrap_for_pdf(text, width)):
            prefix = prefix1 if idx == 0 else prefix_n
            if y < MIN_Y:
                return
            cmds.append(_pdf_text(44, y - 2, f"{prefix}{line}", size=9.5))
            y -= 13

    def numbered_list(items: list[str], width: int = 92):
        nonlocal y
        for idx, item in enumerate(items, start=1):
            for line_idx, line in enumerate(_wrap_for_pdf(item, width)):
                if y < MIN_Y:
                    return
                prefix = f"{idx}. " if line_idx == 0 else "    "
                cmds.append(_pdf_text(44, y - 2, f"{prefix}{line}", size=9.5))
                y -= 13

    def bulleted_list(items: list[str], width: int = 95):
        nonlocal y
        for item in items:
            for idx, line in enumerate(_wrap_for_pdf(item, width)):
                if y < MIN_Y:
                    return
                prefix = "-  " if idx == 0 else "   "
                cmds.append(_pdf_text(44, y - 2, f"{prefix}{line}", size=9.5))
                y -= 13

    # --- Letterhead -----------------------------------------------------
    draw_filled_rect(PAGE_LEFT, y - 46, PAGE_WIDTH, 46, 0.914, 0.949, 0.992)
    draw_rect(PAGE_LEFT, y - 46, PAGE_WIDTH, 46)
    cmds.append(_pdf_text(44, y - 16, FACILITY_NAME, size=14, font="F2"))
    cmds.append(_pdf_text(44, y - 30, FACILITY_SUBTITLE, size=9))
    cmds.append(_pdf_text(44, y - 42, FACILITY_ADDRESS, size=7.5))
    cmds.append(_pdf_text(400, y - 16, f"Report ID: {report['report_id']}", size=9, font="F2"))
    cmds.append(_pdf_text(400, y - 30, f"Status: {report['report_status']}", size=9))
    cmds.append(_pdf_text(400, y - 42, f"Study Date: {report['study_date']}", size=9))
    y -= 56

    # --- Patient demographics -------------------------------------------
    section_title("PATIENT")
    row_h = 15
    rows = [
        (f"Name: {report['name']}", f"DOB: {report['dob']}"),
        (f"Gender: {report['gender']}", f"MRN: {report['mrn']}"),
        (f"Phone: {report['phone']}", f"Referring Physician: {report['referring_physician']}"),
    ]
    table_h = row_h * len(rows)
    bottom = y - table_h
    draw_rect(PAGE_LEFT, bottom, PAGE_WIDTH, table_h)
    cmds.append("0.820 0.860 0.920 RG")
    cmds.append("1 w")
    cmds.append(f"306 {bottom:.2f} m 306 {y:.2f} l S")
    for idx in range(1, len(rows)):
        hline(y - (idx * row_h))
    for idx, (left, right) in enumerate(rows):
        y_text = y - 11 - (idx * row_h)
        cmds.append(_pdf_text(44, y_text, left, size=9))
        cmds.append(_pdf_text(314, y_text, right, size=9))
    y = bottom - 8

    # --- Exam title -------------------------------------------------------
    cmds.append(_pdf_text(PAGE_LEFT, y, f"EXAM: {report['technique_title']}", size=10.5, font="F2"))
    y -= 18

    section_title("CLINICAL HISTORY / INDICATION")
    body_lines(report["symptoms"], 100)
    y -= 6

    section_title("COMPARISON")
    body_lines(report["comparison"], 100)
    y -= 6

    section_title("TECHNIQUE")
    body_lines(report["technique"], 100)
    y -= 6

    section_title("FINDINGS")
    bulleted_list(report["findings"], 95)
    y -= 6

    section_title("IMPRESSION", tint=(0.929, 0.961, 0.933))
    numbered_list(report["impression"], 92)
    if report.get("priority"):
        y -= 2
        cmds.append(_pdf_text(44, y - 2, report["priority"], size=9.5, font="F2"))
        y -= 14
    y -= 6

    section_title("RECOMMENDATIONS")
    numbered_list(report["recommendations"], 92)
    y -= 10

    # --- Electronic signature --------------------------------------------
    sign_h = 34
    sign_bottom = y - sign_h
    draw_rect(PAGE_LEFT, sign_bottom, PAGE_WIDTH, sign_h)
    cmds.append(_pdf_text(44, y - 13, f"Electronically signed by: {report['radiologist_name']}", size=9.5, font="F2"))
    cmds.append(_pdf_text(44, y - 27, f"License No: {report['license']}    |    Signed: {report['signed_at']}", size=9))
    y = sign_bottom - 14

    # --- AI-assisted analysis addendum (kept visually separate from the
    # clinical narrative above, the way CAD/AI-assisted reads are commonly
    # disclosed in practice, instead of blended into Findings/Impression) --
    section_title("AI-ASSISTED ANALYSIS ADDENDUM", tint=(0.965, 0.955, 0.90))
    addendum_lines = [
        f"System output: {report['label_upper']} (confidence {report['confidence']}%) "
        f"-- Normal {report['normal_prob']}% / Pneumonia {report['pneumonia_prob']}%",
        f"Model: {report['model']}  |  Reported test-set accuracy: {report['accuracy']}",
        f"Interpretation guidance: {report['interpretation']}",
    ]
    for line in addendum_lines:
        body_lines(line, 100)
    y -= 4
    bulleted_list(report["limitations"], 95)
    y -= 2
    body_lines(report["disclaimer"], 100, prefix1="", prefix_n="")

    stream_data = "\n".join(cmds).encode("latin-1", "replace")

    objects = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objects.append(
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>\n"
        b"endobj\n"
    )
    objects.append(
        f"4 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode("ascii")
        + stream_data
        + b"\nendstream\nendobj\n"
    )
    objects.append(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    objects.append(b"6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    xref_start = len(pdf)
    count = len(objects) + 1

    pdf.extend(f"xref\n0 {count}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")

    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    pdf.extend(
        (
            f"trailer\n<< /Size {count} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF"
        ).encode("ascii")
    )

    return bytes(pdf)


# Inject custom CSS styling for the whole app (colors, cards, report layout). Purely visual.
st.markdown(
    """
    <style>
    :root {
        --blue: #0b5ed7;
        --blue-dark: #084db2;
        --blue-soft: #eaf3ff;
        --bg: #f4f8fd;
        --card: #ffffff;
        --text: #1f2a37;
        --muted: #5f6c7b;
        --line: #d7e4f2;
        --ok: #2e8b57;
        --danger: #d63031;
    }

    .main {
        background:
            radial-gradient(1200px 450px at -10% -10%, rgba(30, 115, 245, 0.15), rgba(30, 115, 245, 0) 55%),
            radial-gradient(1200px 450px at 110% -20%, rgba(11, 94, 215, 0.15), rgba(11, 94, 215, 0) 55%),
            linear-gradient(155deg, #f9fcff 0%, #f2f7fd 55%, #e8f0fa 100%);
        animation: bgShift 12s ease-in-out infinite alternate;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--blue);
        margin: 0 0 0.2rem 0;
        letter-spacing: 0.2px;
        animation: fadeInUp 0.55s ease-out;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 1rem;
        margin: 0 0 1.1rem 0;
        animation: fadeInUp 0.7s ease-out;
    }

    .card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 10px 26px rgba(15, 65, 135, 0.10);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
        animation: fadeInUp 0.5s ease-out;
        color: var(--text);
    }

    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 30px rgba(15, 65, 135, 0.17);
    }

    .upload-frame {
        border: 1px dashed #bfd2ea;
        background: #fbfdff;
        border-radius: 16px;
        padding: 0.7rem;
        margin-top: 0.5rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .upload-frame:hover {
        border-color: #8db5e8;
        box-shadow: 0 8px 18px rgba(24, 87, 170, 0.14);
    }

    .result-normal {
        border-left: 8px solid var(--ok);
        background: linear-gradient(140deg, #ffffff 0%, #eaf8f0 100%);
    }

    .result-pneumonia {
        border-left: 8px solid var(--danger);
        background: linear-gradient(140deg, #ffffff 0%, #ffecee 100%);
    }

    .confidence-track {
        width: 100%;
        height: 12px;
        border-radius: 999px;
        background: #ffe1e3;
        overflow: hidden;
        margin: 0.6rem 0 0.3rem 0;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #2e8b57, #f4c542, #d63031);
    }

    .low-confidence-banner {
        border-left: 8px solid #e0a800;
        background: linear-gradient(140deg, #ffffff 0%, #fff8e6 100%);
        border-radius: 20px;
        padding: 0.9rem 1rem;
        margin-top: 0.7rem;
        color: #6b4e00;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 10px 26px rgba(15, 65, 135, 0.10);
    }

    .pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: var(--blue-soft);
        color: #165dc7;
        font-size: 0.84rem;
        font-weight: 700;
        margin: 0.3rem 0.4rem 0 0;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(130px, 1fr));
        gap: 0.75rem;
        margin-top: 0.7rem;
    }

    .metric-box {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.7rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: fadeInUp 0.65s ease-out;
    }

    .metric-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(15, 65, 135, 0.14);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.9rem;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    .section-title {
        color: #0f2f5a;
        font-size: 1.08rem;
        font-weight: 800;
        margin-bottom: 0.55rem;
    }

    .report-card {
        background: linear-gradient(140deg, #ffffff 0%, #f5faff 100%);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 8px 20px rgba(15, 65, 135, 0.09);
        color: #1f2a37;
        line-height: 1.65;
        animation: fadeInUp 0.55s ease-out;
    }

    .report-card b {
        color: #0f2f5a;
        font-weight: 700;
    }

    .report-card span {
        color: #1f2a37;
    }

    .radiology-report {
        background: #ffffff;
        border: 1px solid #cfd9e6;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 10px 24px rgba(16, 55, 110, 0.12);
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
        overflow-x: hidden;
    }

    .rr-header {
        background: linear-gradient(120deg, #f6f9fe 0%, #eaf2ff 100%);
        border-bottom: 1px solid #d3dfef;
        padding: 0.9rem 1rem;
        text-align: center;
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f2f5a;
        letter-spacing: 0.4px;
    }

    .rr-section-title {
        background: #f2f5fb;
        border-top: 1px solid #dbe4f1;
        border-bottom: 1px solid #dbe4f1;
        padding: 0.5rem 0.8rem;
        font-size: 1rem;
        font-weight: 800;
        color: #1a365d;
    }

    .rr-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        width: 100%;
    }

    .rr-cell {
        border-right: 1px solid #e2e9f3;
        border-bottom: 1px solid #e2e9f3;
        padding: 0.55rem 0.75rem;
        color: #25364d;
        background: #ffffff;
        overflow-wrap: anywhere;
        word-break: break-word;
        white-space: normal;
        min-width: 0;
        box-sizing: border-box;
    }

    .rr-cell:nth-child(even) {
        border-right: none;
    }

    .rr-block {
        padding: 0.72rem 0.8rem;
        border-bottom: 1px solid #e2e9f3;
        color: #223247;
        line-height: 1.55;
        background: #ffffff;
        overflow-wrap: anywhere;
        word-break: break-word;
    }

    .rr-letterhead {
        background: linear-gradient(120deg, #f6f9fe 0%, #eaf2ff 100%);
        border-bottom: 2px solid #17324d;
        padding: 1rem 1.1rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .rr-facility-name {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f2f5a;
        letter-spacing: 0.3px;
    }

    .rr-facility-sub {
        color: #2b6f8a;
        font-size: 0.85rem;
        font-weight: 700;
        margin-top: 0.1rem;
    }

    .rr-facility-addr {
        color: #5f6c7b;
        font-size: 0.72rem;
        margin-top: 0.2rem;
    }

    .rr-letterhead-meta {
        text-align: right;
        color: #1a365d;
        font-size: 0.82rem;
        line-height: 1.5;
        white-space: nowrap;
    }

    .rr-exam-title {
        padding: 0.6rem 0.8rem;
        font-weight: 800;
        color: #0f2f5a;
        background: #fbfdff;
        border-bottom: 1px solid #e2e9f3;
        letter-spacing: 0.3px;
    }

    .rr-section-impression {
        background: #e9f5eb !important;
        color: #1a4d2e !important;
    }

    .rr-signature {
        padding: 0.8rem 1rem;
        border-top: 2px solid #17324d;
        border-bottom: 1px solid #e2e9f3;
        background: #fbfdff;
        color: #1a365d;
        font-size: 0.86rem;
        line-height: 1.6;
    }

    .rr-addendum {
        background: #fffdf5;
        border-top: 1px dashed #d8c98a;
    }

    .rr-addendum-title {
        padding: 0.5rem 0.8rem;
        font-weight: 800;
        font-size: 0.88rem;
        color: #6b5900;
        background: #fbf3d9;
        border-bottom: 1px solid #ecdfa8;
    }

    .rr-addendum .rr-block {
        font-size: 0.86rem;
        color: #4a4230;
    }

    .rr-foot-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        width: 100%;
    }

    .rr-foot-cell {
        padding: 0.6rem 0.75rem;
        border-right: 1px solid #e2e9f3;
        border-bottom: 1px solid #e2e9f3;
        color: #25364d;
        overflow-wrap: anywhere;
        word-break: break-word;
        white-space: normal;
        min-width: 0;
        box-sizing: border-box;
        line-height: 1.45;
    }

    .rr-foot-cell:nth-child(even) {
        border-right: none;
    }

    .sidebar-card {
        background: linear-gradient(140deg, #f8fbff 0%, #ebf3ff 100%);
        border: 1px solid #bfd3ec;
        border-radius: 14px;
        padding: 0.8rem;
        margin-bottom: 0.9rem;
        animation: fadeInUp 0.55s ease-out;
        color: #16345f;
    }

    .sidebar-caption {
        color: #31598d;
        font-size: 0.95rem;
        font-weight: 700;
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        background: linear-gradient(120deg, var(--blue) 0%, #1d73f5 100%);
        color: #ffffff;
        font-weight: 700;
        padding: 0.62rem 1rem;
        box-shadow: 0 10px 20px rgba(11, 94, 215, 0.28);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 26px rgba(11, 94, 215, 0.36);
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes bgShift {
        from { background-position: 0% 0%, 100% 0%, 0% 0%; }
        to { background-position: -8% 6%, 108% -6%, 0% 0%; }
    }

    .field-error {
        color: var(--danger);
        font-size: 0.82rem;
        margin: -0.55rem 0 0.5rem 0.1rem;
        font-weight: 600;
    }

    .required-mark {
        color: var(--danger);
        font-weight: 800;
    }

    .section-divider {
        border: none;
        border-top: 1px solid var(--line);
        margin: 0.9rem 0;
    }

    /* Native bordered containers (st.container(border=True)) styled to
       match the custom .card look instead of Streamlit's plain gray box. */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card);
        border: 1px solid var(--line) !important;
        border-radius: 18px !important;
        box-shadow: 0 10px 26px rgba(15, 65, 135, 0.08);
        padding: 0.2rem 0.2rem 0.6rem 0.2rem;
        margin-bottom: 1.1rem;
    }

    div[data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
    }

    /* Section headings inside cards (### markdown) */
    div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        color: #0f2f5a;
        font-size: 1.15rem;
        font-weight: 800;
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.55rem;
        margin-bottom: 0.9rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] strong {
        color: #1a365d;
    }

    /* Multiselect tags (Findings / Recommendations / Symptoms chips) */
    span[data-baseweb="tag"] {
        background: linear-gradient(120deg, var(--blue) 0%, #1d73f5 100%) !important;
        border-radius: 999px !important;
    }

    /* File uploader drop zone */
    section[data-testid="stFileUploaderDropzone"] {
        background: #fbfdff;
        border: 1px dashed #bfd2ea !important;
        border-radius: 16px;
    }

    /* Text inputs, textareas, selects, date inputs -- consistent rounded look */
    div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="textarea"],
    div[data-baseweb="datepicker"] input {
        border-radius: 10px !important;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-baseweb="select"]:focus-within,
    textarea:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 2px rgba(11, 94, 215, 0.15) !important;
    }

    div[data-testid="stButton"] > button:disabled {
        background: #d3dde8 !important;
        color: #93a3b8 !important;
        box-shadow: none !important;
        cursor: not-allowed;
    }

    /* Sidebar radio nav items */
    div[data-testid="stSidebar"] label {
        font-size: 0.92rem;
    }

    @media (max-width: 900px) {
        .hero-title { font-size: 1.7rem; }
        .metric-grid { grid-template-columns: 1fr; }
        .rr-grid { grid-template-columns: 1fr; }
        .rr-foot-grid { grid-template-columns: 1fr; }
        .rr-cell:nth-child(even), .rr-foot-cell:nth-child(even) { border-right: 1px solid #e2e9f3; }
        .rr-letterhead { flex-direction: column; }
        .rr-letterhead-meta { text-align: left; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sidebar: project info, team members and page navigation.
with st.sidebar:
    st.markdown("### 🫀 Project Dashboard")
    st.image("assets/project_banner.png", use_container_width=True)
    st.markdown(
        """
        <div class="sidebar-card">
        <b>Federated Learning Based Pneumonia Detection System</b><br>
        <span class="sidebar-caption">Capstone Project</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Project Details")
    dataset_line = (
        "- Dataset: RSNA (adult) + Kermany (pediatric) chest X-rays"
        if HAS_PEDIATRIC_METRICS
        else "- Dataset: Chest X-ray Pneumonia Dataset"
    )
    st.markdown(
        "- Federated Learning Algorithm: SCAFFOLD\n"
        "- Deep Learning Model: DenseNet121\n"
        f"{dataset_line}"
    )

    st.markdown("#### Team Members")
    st.markdown("- Veerana Gouda\n- Satwik K\n- Satwik Hegde")

    nav = st.radio("Navigation", ["Home", "About Project", "Model Performance", "Contact"], index=0)


# Session state remembers the last prediction between button clicks.
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "show_report" not in st.session_state:
    st.session_state.show_report = False


# Page routing: About Project / Model Performance / Contact / Home (upload and predict).
if nav == "About Project":
    st.markdown('<div class="hero-title">About Project</div>', unsafe_allow_html=True)
    st.markdown(
        "This capstone demonstrates privacy-preserving pneumonia detection from chest X-rays. "
        "SCAFFOLD-based federated learning is used to train DenseNet121 without centralizing raw patient data."
    )

elif nav == "Model Performance":
    st.markdown('<div class="hero-title">📊 Model Performance</div>', unsafe_allow_html=True)
    adult_label = "Adult Test Set" if HAS_PEDIATRIC_METRICS else "Test Set"
    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">{adult_label}</div>
            <div class="metric-grid">
                <div class="metric-box"><div class="metric-label">Accuracy</div><div class="metric-value">{DISPLAY_METRICS['accuracy']}</div></div>
                <div class="metric-box"><div class="metric-label">Precision</div><div class="metric-value">{DISPLAY_METRICS['precision']}</div></div>
                <div class="metric-box"><div class="metric-label">Recall</div><div class="metric-value">{DISPLAY_METRICS['recall']}</div></div>
                <div class="metric-box"><div class="metric-label">F1 Score</div><div class="metric-value">{DISPLAY_METRICS['f1']}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if HAS_PEDIATRIC_METRICS:
        st.markdown(
            f"""
            <div class="card">
                <div class="section-title">Pediatric Test Set</div>
                <div class="metric-grid">
                    <div class="metric-box"><div class="metric-label">Accuracy</div><div class="metric-value">{DISPLAY_METRICS['pediatric_accuracy']}</div></div>
                    <div class="metric-box"><div class="metric-label">Precision</div><div class="metric-value">{DISPLAY_METRICS['pediatric_precision']}</div></div>
                    <div class="metric-box"><div class="metric-label">Recall</div><div class="metric-value">{DISPLAY_METRICS['pediatric_recall']}</div></div>
                    <div class="metric-box"><div class="metric-label">F1 Score</div><div class="metric-value">{DISPLAY_METRICS['pediatric_f1']}</div></div>
                </div>
                <div style="color:#5f6c7b; font-size:0.82rem; margin-top:0.7rem; line-height:1.5;">
                    Reported separately from the adult test set (rather than one pooled number) so
                    performance on each age group is visible on its own, since averaging the two
                    together could hide a group-specific weakness.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif nav == "Contact":
    st.markdown('<div class="hero-title">Contact</div>', unsafe_allow_html=True)
    st.markdown("For demo inquiries, please contact the capstone team coordinator.")

else:
    st.markdown('<div class="hero-title">Federated Learning for Pneumonia Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">A privacy-preserving pneumonia detection system using SCAFFOLD and DenseNet121.</div>',
        unsafe_allow_html=True,
    )

    today = datetime.now().date()
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        with st.container(border=True):
            st.markdown("### 👤 Patient & Study Information")
            st.caption("Fields marked with <span class='required-mark'>*</span> are required.", unsafe_allow_html=True)

            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                patient_id = st.text_input("Patient / MRN ID *", key="patient_id", placeholder="e.g. 2026001")
                patient_name = st.text_input("Patient Name *", key="patient_name", placeholder="e.g. Asha Rao")
                patient_dob = st.date_input(
                    "Date of Birth *", value=datetime(2000, 1, 1).date(),
                    min_value=datetime(1900, 1, 1).date(), max_value=today, key="patient_dob",
                )
                age_years = max(0, (today - patient_dob).days // 365)
                st.caption(f"Age: **{age_years} years** (auto-calculated)")
            with detail_col2:
                patient_phone = st.text_input("Phone Number *", key="patient_phone", placeholder="e.g. +91 98765 43210")
                patient_gender = st.selectbox("Gender *", ["Select", "Male", "Female", "Other"], key="patient_gender")
                patient_report_date = st.date_input(
                    "Date of Study *", value=today, max_value=today, key="patient_report_date",
                )
                referring_physician = st.text_input("Referring Physician", key="referring_physician", placeholder="Optional")

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("**Clinical History** — select symptoms instead of typing them out")
            symptoms_selected = st.multiselect(
                "Reported symptoms", STANDARD_SYMPTOMS, key="symptoms_selected",
                placeholder="Choose one or more symptoms",
            )
            symptoms_notes = st.text_input(
                "Additional clinical notes (optional)", key="symptoms_notes",
                placeholder="e.g. Symptoms for 5 days, no travel history",
            )

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("**Examination Technique**")
            tech_col1, tech_col2 = st.columns([1, 1])
            with tech_col1:
                technique_choice = st.selectbox("Technique", TECHNIQUE_OPTIONS, key="technique_choice")
            with tech_col2:
                comparison_has_prior = st.checkbox("Prior study available for comparison?", key="comparison_has_prior")
                comparison_date = None
                if comparison_has_prior:
                    comparison_date = st.date_input("Prior study date", max_value=today, key="comparison_date")

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("**Sign-off**")
            sign_col1, sign_col2 = st.columns(2)
            with sign_col1:
                radiologist_name = st.text_input("Radiologist Name *", key="radiologist_name", placeholder="e.g. Dr. Meera Nair")
            with sign_col2:
                radiologist_license = st.text_input("License Number *", key="radiologist_license", placeholder="e.g. RAD-2024-1187")

            form_values = {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "patient_gender": patient_gender,
                "patient_phone": patient_phone,
                "patient_dob": patient_dob,
                "patient_report_date": patient_report_date,
                "comparison_has_prior": comparison_has_prior,
                "comparison_date": comparison_date,
                "radiologist_name": radiologist_name,
                "radiologist_license": radiologist_license,
            }
            form_errors = validate_patient_form(form_values, today)

            field_labels = {
                "patient_id": "Patient / MRN ID", "patient_name": "Patient Name",
                "patient_gender": "Gender", "patient_phone": "Phone Number",
                "patient_dob": "Date of Birth", "patient_report_date": "Date of Study",
                "comparison_date": "Prior Study Date", "radiologist_name": "Radiologist Name",
                "radiologist_license": "License Number",
            }
            if form_errors:
                for field_key, message in form_errors.items():
                    st.markdown(
                        f'<div class="field-error">⚠ {field_labels.get(field_key, field_key)}: {message}</div>',
                        unsafe_allow_html=True,
                    )

        patient_info_complete = not form_errors

        with st.container(border=True):
            st.markdown("### 📤 Upload Chest X-ray")

            uploaded_file = None
            uploaded_bytes = None
            upload_is_valid = False

            if not patient_info_complete:
                st.info(
                    "Complete the required patient & study details above (see any ⚠ messages) "
                    "before uploading the X-ray image."
                )
                st.file_uploader(
                    "Choose a chest X-ray image", type=["jpg", "jpeg", "png"],
                    accept_multiple_files=False, disabled=True, key="xray_uploader_disabled",
                )
            else:
                uploaded_file = st.file_uploader(
                    "Choose a chest X-ray image",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=False,
                    key="xray_uploader",
                )

                if uploaded_file is not None:
                    raw_bytes = uploaded_file.getvalue()
                    try:
                        preview_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                        st.markdown('<div class="upload-frame">', unsafe_allow_html=True)
                        st.image(preview_img, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        uploaded_bytes = raw_bytes
                        upload_is_valid = True
                    except Exception:
                        st.error(
                            f"'{uploaded_file.name}' could not be read as an image. "
                            "The file may be corrupted or is not actually a JPG/PNG — please upload a valid chest X-ray image."
                        )

            # Predict button: enabled only after the patient form is valid and a readable image is uploaded.
            predict_clicked = st.button(
                "🩺 Predict", disabled=not upload_is_valid,
                help="Upload a valid chest X-ray image first." if not upload_is_valid else None,
            )

    # Run inference and store the result; show an error instead of a fake result if it fails.
    if predict_clicked and uploaded_file is not None and uploaded_bytes is not None:
        try:
            label, confidence, normal_prob, pneumonia_prob = predict_image(uploaded_bytes)
            st.session_state.prediction = {
                "label": label,
                "confidence": confidence,
                "normal_prob": normal_prob,
                "pneumonia_prob": pneumonia_prob,
                "image_name": uploaded_file.name,
                "timestamp": datetime.now(),
            }
        except Exception as exc:
            # Never fabricate a result on failure -- a silently-substituted
            # "Normal, 50% confidence" reading is indistinguishable from a
            # real low-confidence prediction and is unacceptable for a
            # medical screening tool. Surface the failure instead.
            st.session_state.prediction = None
            st.error(
                f"Inference failed for this image, so no prediction was made: {exc}\n\n"
                "Please verify the file is a valid, uncorrupted chest X-ray image and try again."
            )
        st.session_state.show_report = False

    # Right column: prediction result, performance cards and report preview / download.
    with right_col:
        with st.container(border=True):
            st.markdown("### 🩺 Prediction Result")

            pred = st.session_state.prediction
            if pred is None:
                st.info("Upload an image and click Predict to view result.")
            else:
                is_pneumonia = pred["label"] == "Pneumonia"
                result_class = "result-pneumonia" if is_pneumonia else "result-normal"
                result_color = "#d63031" if is_pneumonia else "#2e8b57"

                pneumonia_pct = pred["pneumonia_prob"] * 100
                st.markdown(
                    f"""
                    <div class="card {result_class}">
                        <div style="font-size:1.15rem; font-weight:800; color:{result_color};">Prediction: {pred['label']}</div>
                        <div style="color:#374151; margin-top:0.3rem;">Confidence Score: {pred['confidence'] * 100:.2f}%</div>
                        <div class="confidence-track"><div class="confidence-fill" style="width:{pneumonia_pct:.2f}%;"></div></div>
                        <div style="color:#374151; font-size:0.85rem;">
                            <span>Normal: {pred['normal_prob'] * 100:.2f}%</span>
                            &nbsp;|&nbsp;
                            <span>Pneumonia: {pneumonia_pct:.2f}%</span>
                        </div>
                        <div style="margin-top:0.5rem;"><span class="pill">Federated Algorithm: SCAFFOLD</span><span class="pill">Model Used: DenseNet121</span><span class="pill">5-view TTA</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if pred["confidence"] < LOW_CONFIDENCE_THRESHOLD:
                    st.markdown(
                        f"""
                        <div class="low-confidence-banner">
                            ⚠️ Borderline confidence ({pred['confidence']*100:.1f}%) — this reading is close to the
                            model's decision boundary. Do not rely on this result alone; manual radiologist review
                            of the image is strongly recommended.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                adult_card_label = "📊 Model Performance (Adult Test Set)" if HAS_PEDIATRIC_METRICS else "📊 Model Performance"
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="section-title">{adult_card_label}</div>
                        <div class="metric-grid">
                            <div class="metric-box"><div class="metric-label">Accuracy</div><div class="metric-value">{DISPLAY_METRICS['accuracy']}</div></div>
                            <div class="metric-box"><div class="metric-label">Precision</div><div class="metric-value">{DISPLAY_METRICS['precision']}</div></div>
                            <div class="metric-box"><div class="metric-label">Recall</div><div class="metric-value">{DISPLAY_METRICS['recall']}</div></div>
                            <div class="metric-box"><div class="metric-label">F1 Score</div><div class="metric-value">{DISPLAY_METRICS['f1']}</div></div>
                        </div>
                        <div style="color:#5f6c7b; font-size:0.82rem; margin-top:0.7rem; line-height:1.5;">
                            Recall of {DISPLAY_METRICS['recall']} means some real PNEUMONIA cases can be
                            missed on a single read, and precision of {DISPLAY_METRICS['precision']} means some
                            NORMAL cases can be flagged as PNEUMONIA. A "Normal" result does not rule out pneumonia
                            with certainty &mdash; this tool supports, but does not replace, radiologist judgment.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if HAS_PEDIATRIC_METRICS:
                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="section-title">📊 Model Performance (Pediatric Test Set)</div>
                            <div class="metric-grid">
                                <div class="metric-box"><div class="metric-label">Accuracy</div><div class="metric-value">{DISPLAY_METRICS['pediatric_accuracy']}</div></div>
                                <div class="metric-box"><div class="metric-label">Precision</div><div class="metric-value">{DISPLAY_METRICS['pediatric_precision']}</div></div>
                                <div class="metric-box"><div class="metric-label">Recall</div><div class="metric-value">{DISPLAY_METRICS['pediatric_recall']}</div></div>
                                <div class="metric-box"><div class="metric-label">F1 Score</div><div class="metric-value">{DISPLAY_METRICS['pediatric_f1']}</div></div>
                            </div>
                            <div style="color:#5f6c7b; font-size:0.82rem; margin-top:0.7rem; line-height:1.5;">
                                This model was trained on both adult (RSNA) and pediatric (Kermany) chest X-rays as
                                separate federated clients, so it can be used for either age group. Adult and
                                pediatric performance are reported separately, not averaged together.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                suggested_findings = FINDINGS_SUGGESTED[pred["label"]]
                suggested_impression = (
                    "Findings are suggestive of infective consolidation, likely pneumonia. Clinical correlation is advised."
                    if is_pneumonia
                    else "No radiographic evidence of acute cardiopulmonary abnormality; no imaging features suggestive of "
                         "active pneumonia."
                )
                suggested_recs = RECOMMENDATIONS_SUGGESTED[pred["label"]]

                findings = list(suggested_findings)
                impression_items = [suggested_impression]
                recommendations = list(suggested_recs)

                report_id_seed = "".join(ch for ch in patient_id if ch.isdigit())
                report_num = int(report_id_seed) if report_id_seed else 101
                report_id = f"FL-PNEU-{datetime.now().year}-{report_num:06d}"
                interpretation = (
                    "Radiographic features are suggestive of pneumonia. Correlate with clinical and laboratory findings."
                    if is_pneumonia
                    else "No strong radiographic evidence of pneumonia detected. Correlate clinically."
                )

                priority_note = "PRIORITY: Urgent review recommended." if is_pneumonia else ""
                limitations = [
                    "This is an AI-generated screening aid based on imaging data only, with no access to prior "
                    "history, labs, or clinical context beyond what was entered above.",
                    "Reported test-set performance: "
                    f"accuracy {DISPLAY_METRICS['accuracy']}, precision {DISPLAY_METRICS['precision']}, "
                    f"recall {DISPLAY_METRICS['recall']}. Early or subtle abnormalities may not be detected, "
                    "and some normal studies may be flagged as abnormal.",
                ]
                report_status = "FINAL"
                mandatory_radiologist = radiologist_name.strip() or "____________________________"
                signoff_disclaimer = (
                    "This report was produced with AI-assisted image analysis (see addendum below). "
                    "Final interpretation is the responsibility of the signing radiologist."
                )
                technique_title = f"{technique_choice.upper()}"
                technique_line = f"{technique_choice} acquired under standard exposure conditions."
                comparison_line = (
                    f"Compared with prior study dated {comparison_date.strftime('%Y-%m-%d')}."
                    if comparison_has_prior and comparison_date
                    else "No prior imaging available for comparison."
                )
                symptoms_line = ", ".join(symptoms_selected) if symptoms_selected else "No symptoms selected"
                if symptoms_notes.strip():
                    symptoms_line += f". {symptoms_notes.strip()}"

                patient = {
                    "id": patient_id.strip() or "Not Provided",
                    "name": patient_name.strip() or "Not Provided",
                    "phone": patient_phone.strip() or "Not Provided",
                    "dob": patient_dob.strftime("%Y-%m-%d"),
                    "gender": patient_gender if patient_gender != "Select" else "Not Provided",
                    "report_date": patient_report_date.strftime("%Y-%m-%d"),
                    "referring_physician": referring_physician.strip() or "Not Provided",
                    "clinical_history": symptoms_line,
                    "radiologist_license": radiologist_license.strip() or "Not Provided",
                }

                ts_value = pred["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                findings_html = "<br>".join(f"&bull; {item}" for item in findings) or "&bull; Not Provided"
                impression_html = (
                    "<br>".join(f"{idx}. {item}" for idx, item in enumerate(impression_items, start=1))
                    or "Not Provided"
                )
                if priority_note:
                    impression_html += f"<br><b style='color:#d63031;'>{priority_note}</b>"
                recommendations_html = (
                    "<br>".join(f"{idx}. {item}" for idx, item in enumerate(recommendations, start=1))
                    or "Not Provided"
                )
                limitations_html = "<br>".join(f"&bull; {item}" for item in limitations)

                report_html = textwrap.dedent(
                    f"""
                    <div class="radiology-report">
                        <div class="rr-letterhead">
                            <div class="rr-letterhead-main">
                                <div class="rr-facility-name">{FACILITY_NAME}</div>
                                <div class="rr-facility-sub">{FACILITY_SUBTITLE}</div>
                                <div class="rr-facility-addr">{FACILITY_ADDRESS}</div>
                            </div>
                            <div class="rr-letterhead-meta">
                                <div><b>Report ID:</b> {report_id}</div>
                                <div><b>Status:</b> {report_status}</div>
                                <div><b>Study Date:</b> {patient['report_date']}</div>
                            </div>
                        </div>

                        <div class="rr-section-title">Patient</div>
                        <div class="rr-grid">
                            <div class="rr-cell"><b>Name:</b> {patient['name']}</div>
                            <div class="rr-cell"><b>Date of Birth:</b> {patient['dob']}</div>
                            <div class="rr-cell"><b>Gender:</b> {patient['gender']}</div>
                            <div class="rr-cell"><b>MRN:</b> {patient['id']}</div>
                            <div class="rr-cell"><b>Phone:</b> {patient['phone']}</div>
                            <div class="rr-cell"><b>Referring Physician:</b> {patient['referring_physician']}</div>
                        </div>

                        <div class="rr-exam-title">EXAM: {technique_title}</div>

                        <div class="rr-section-title">Clinical History / Indication</div>
                        <div class="rr-block">{patient['clinical_history']}</div>

                        <div class="rr-section-title">Comparison</div>
                        <div class="rr-block">{comparison_line}</div>

                        <div class="rr-section-title">Technique</div>
                        <div class="rr-block">{technique_line}</div>

                        <div class="rr-section-title">Findings</div>
                        <div class="rr-block">{findings_html}</div>

                        <div class="rr-section-title rr-section-impression">Impression</div>
                        <div class="rr-block">{impression_html}</div>

                        <div class="rr-section-title">Recommendations</div>
                        <div class="rr-block">{recommendations_html}</div>

                        <div class="rr-signature">
                            <div><b>Electronically signed by:</b> {mandatory_radiologist}</div>
                            <div><b>License No:</b> {patient['radiologist_license']} &nbsp;|&nbsp; <b>Signed:</b> {ts_value}</div>
                        </div>

                        <div class="rr-addendum">
                            <div class="rr-addendum-title">AI-Assisted Analysis Addendum</div>
                            <div class="rr-block">
                                System output: <b>{pred['label'].upper()}</b> (confidence {pred['confidence']*100:.2f}%)
                                &mdash; Normal {pred['normal_prob'] * 100:.2f}% / Pneumonia {pred['pneumonia_prob'] * 100:.2f}%<br>
                                Model: DenseNet121, federated training (SCAFFOLD) &nbsp;|&nbsp;
                                Reported test-set accuracy: {DISPLAY_METRICS['accuracy']}<br>
                                Interpretation guidance: {interpretation}
                            </div>
                            <div class="rr-block">{limitations_html}</div>
                            <div class="rr-block" style="font-style:italic;">{signoff_disclaimer}</div>
                        </div>
                    </div>
                    """,
                )
                report_html = "\n".join(line.lstrip() for line in report_html.splitlines()).strip()

                report_payload = {
                    "name": patient["name"],
                    "dob": patient["dob"],
                    "gender": patient["gender"],
                    "mrn": patient["id"],
                    "report_id": report_id,
                    "report_status": report_status,
                    "phone": patient["phone"],
                    "study_date": patient["report_date"],
                    "referring_physician": patient["referring_physician"],
                    "image_name": pred["image_name"],
                    "symptoms": patient["clinical_history"],
                    "technique_title": technique_title,
                    "technique": technique_line,
                    "comparison": comparison_line,
                    "findings": findings or ["Not Provided"],
                    "impression": impression_items or ["Not Provided"],
                    "priority": priority_note,
                    "recommendations": recommendations or ["Not Provided"],
                    "normal_prob": f"{pred['normal_prob'] * 100:.2f}",
                    "pneumonia_prob": f"{pred['pneumonia_prob'] * 100:.2f}",
                    "label_upper": pred["label"].upper(),
                    "confidence": f"{pred['confidence'] * 100:.2f}",
                    "model": "DenseNet121, federated training (SCAFFOLD)",
                    "accuracy": DISPLAY_METRICS["accuracy"],
                    "interpretation": interpretation,
                    "limitations": limitations,
                    "radiologist_name": mandatory_radiologist,
                    "license": patient["radiologist_license"],
                    "signed_at": ts_value,
                    "disclaimer": signoff_disclaimer,
                }
                # Build the downloadable PDF report.
                pdf_bytes = build_radiology_pdf(report_payload)

                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    view_btn_label = "🙈 Hide Report" if st.session_state.show_report else "👁️ View Report"
                    if st.button(view_btn_label, key="toggle_report_btn"):
                        st.session_state.show_report = not st.session_state.show_report
                with action_col2:
                    st.download_button(
                        label="⬇️ Download Report PDF",
                        data=pdf_bytes,
                        file_name=f"radiology_report_{patient['id']}_{report_id}.pdf",
                        mime="application/pdf",
                    )

                if st.session_state.show_report:
                    st.markdown("### 📄 Report Preview")
                    st.markdown(report_html, unsafe_allow_html=True)
  