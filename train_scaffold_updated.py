"""
train_scaffold_updated.py
-------------------------
Federated learning with SCAFFOLD algorithm for pneumonia detection.

Architecture : DenseNet121 (pretrained ImageNet)
               - Unfreeze last 40 layers
               - Custom classification head with GlobalAveragePooling2D
Model output : Binary classification (NORMAL=0, PNEUMONIA=1)
Dataset      : dataset/train/client_{i}/ and dataset/test/
Framework    : Flower + PyTorch + TensorFlow/Keras layers

Features:
- SCAFFOLD federated learning with 15 rounds
- Dynamic learning rate scheduling per round
- Class-weighted loss (0: 1.0, 1: 1.5)
- Prediction threshold at 0.4
- Per-round metrics: Accuracy, Precision, Recall, F1-score
- Final test set evaluation with confusion matrix and classification report

Usage:
    python train_scaffold_updated.py --seed 42
"""

# ---- Imports: standard library, Flower (federated learning), PyTorch (deep learning), scikit-learn (metrics) ----
import argparse
import copy
import csv
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from collections import OrderedDict
from pathlib import Path
from typing import Any, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset, Subset
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
# Use the GPU when available, otherwise fall back to the CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Class labels: index 0 = NORMAL, index 1 = PNEUMONIA (binary classification).
CLASSES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = 2
# Every X-ray is resized to 224x224 pixels, the input size DenseNet121 expects.
INPUT_SIZE = 224
# Default probability cutoff used only for per-round progress logging (the final threshold is tuned later).
PREDICTION_THRESHOLD = 0.4

# DataLoader workers: with num_workers=0, image decode/resize/augment runs
# synchronously on the main thread between GPU steps, leaving the GPU mostly
# idle (observed ~1-2% utilization despite the model living on CUDA).
# Worker processes would overlap loading with compute, but this simulation
# runs each client inside a Ray actor process, and on Windows Ray's actor
# processes don't preserve a stable __main__ module identity -- PyTorch's
# spawned DataLoader workers can't pickle classes back to this script from
# inside that actor, so num_workers>0 crashes here. Keeping 0 for
# reliability; pin_memory is still a free win since it needs no subprocess.
# Number of background data-loading workers (0 = load in the main process; safest with Ray on Windows).
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

# Which backbone build_model()/build_transforms() produce. Set once from
# args.backbone at the start of main(), before any client/model is built.
# "torchvision": DenseNet121 with generic ImageNet1K pretraining (original).
# "xrv": DenseNet121 pretrained on ~100k real chest X-rays (TorchXRayVision,
# pooled from NIH/CheXpert/MIMIC/PadChest/etc.) -- domain-specific features.
# Which pretrained backbone to build: 'torchvision' (ImageNet DenseNet121) or 'xrv' (chest-X-ray pretrained).
BACKBONE = "torchvision"


# BLOCK: Dataset that finds every image under a folder and reads its label from the parent folder name (NORMAL / PNEUMONIA).
class RecursiveLabelDataset(Dataset):
    """Dataset that finds images recursively and infers labels from folder names."""

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = CLASSES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = self._scan_samples()

        if not self.samples:
            raise FileNotFoundError(
                f"No labeled image files found under {self.root}. Expected folders named {self.classes}."
            )

    # Walk the folder tree and collect (image path, label) pairs.
    def _scan_samples(self):
        samples = []
        valid_extensions = {ext.lower() for ext in IMG_EXTENSIONS}

        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in valid_extensions:
                continue

            label_name = self._find_label_name(path)
            if label_name is None:
                continue

            samples.append((path, self.class_to_idx[label_name]))

        return samples

    # Look at the parent folders of an image to find its class name.
    def _find_label_name(self, path: Path):
        for parent in path.parents:
            if parent.name in self.class_to_idx:
                return parent.name
        return None

    def __len__(self):
        return len(self.samples)

    # Load one image, apply the preprocessing transform, and return (image tensor, label).
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = default_loader(str(path))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# BLOCK: Builds the validation DataLoader from a dedicated held-out folder (never used for training).
def build_validation_loader_from_dir(data_root: Path, transform, batch_size: int):
    """Build a validation loader from a dedicated, fully held-out folder
    (never seen by any client during training) — no subsetting needed."""
    dataset = RecursiveLabelDataset(data_root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


# BLOCK: Fallback validation loader that carves a stratified slice out of the training data.
def build_validation_loader(data_root: Path, transform, batch_size: int, validation_ratio: float, seed: int):
    dataset = RecursiveLabelDataset(data_root, transform=transform)
    labels = [label for _, label in dataset.samples]
    indices = np.arange(len(labels))
    _, validation_indices = train_test_split(
        indices,
        test_size=validation_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    validation_subset = Subset(dataset, validation_indices.tolist())
    return DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


# BLOCK: Sweeps 91 decision thresholds (0.05-0.95) on the validation set and returns the one that maximizes the chosen metric.
def find_best_threshold(model: nn.Module, loader: DataLoader, metric: str = "fbeta_precision"):
    """Sweep decision thresholds and pick the one maximizing `metric` on `loader`.

    `fbeta_precision` uses F-beta with beta=0.5, weighting precision higher than
    recall, which is the right target when the goal is fewer false alarms
    (fewer NORMAL cases misclassified as PNEUMONIA).
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images).squeeze(1)
            outputs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(outputs)
            all_labels.extend(labels.numpy())

    all_probs = np.asarray(all_probs)
    all_labels = np.asarray(all_labels)

    best_threshold = 0.5
    best_score = -1.0
    best_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "fbeta_precision": 0.0}

    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = (all_probs >= threshold).astype(np.int64)
        accuracy = accuracy_score(all_labels, predictions)
        precision = precision_score(all_labels, predictions, zero_division=0)
        recall = recall_score(all_labels, predictions, zero_division=0)
        f1 = f1_score(all_labels, predictions, zero_division=0)
        fbeta = fbeta_score(all_labels, predictions, beta=0.5, zero_division=0)

        scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fbeta_precision": fbeta,
        }
        current_score = scores[metric]
        if current_score > best_score:
            best_score = current_score
            best_threshold = float(threshold)
            best_metrics = dict(scores)

    return best_threshold, best_metrics


# BLOCK: Normalization used by the optional chest-X-ray backbone (a class instead of a lambda so it can be pickled on Windows).
class _XRVNormalize:
    """Picklable stand-in for `transforms.Lambda(lambda t: ...)`. A local lambda
    can't be pickled, which breaks DataLoader worker processes (num_workers>0)
    on Windows, where multiprocessing uses spawn instead of fork."""

    def __call__(self, t):
        return (2.0 * t - 1.0) * 1024.0


# BLOCK: Image preprocessing. Train = grayscale -> resize -> augmentation -> tensor -> normalize; test = same without augmentation.
def build_transforms(backbone: str = None):
    """Build train and test transforms. `backbone` must be passed explicitly
    by any caller that may run in a separate process (e.g. a Ray actor),
    since the module-level BACKBONE global set in main() does not propagate
    across process boundaries -- it only helps same-process callers."""
    backbone = backbone or BACKBONE
    if backbone == "xrv":
        # TorchXRayVision models expect single-channel input scaled to
        # roughly [-1024, 1024], not 3-channel ImageNet mean/std.
        num_channels = 1
        normalize = _XRVNormalize()
    else:
        num_channels = 3
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    # Training pipeline: random flips/rotations/shifts/contrast (data augmentation) help the model generalize.
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=num_channels),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(7),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.RandomAutocontrast(p=0.3),
        transforms.ToTensor(),
        normalize,
    ])

    # Test/validation pipeline: deterministic only (no augmentation). The Streamlit app must use the same steps.
    test_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=num_channels),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, test_tf


# BLOCK: ImageNet-pretrained DenseNet121 with only the deeper blocks trainable and a small custom classification head.
class DenseNet121Custom(nn.Module):
    """DenseNet121 with custom classification head and selective unfreezing."""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained DenseNet121
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.densenet121(weights=weights)
        
        # Freeze all layers, then unfreeze the deeper feature blocks by name
        # (denseblock3 onward). Early blocks capture generic low-level texture
        # that transfers fine from ImageNet; fine-tuning only the deeper,
        # more task-specific blocks gives cleaner adaptation to chest X-rays
        # than an arbitrary "last 40 parameter tensors" slice, which cuts
        # across layer boundaries inconsistently.
        # Only these deeper blocks are fine-tuned; early layers keep their generic pretrained features (frozen).
        UNFREEZE_BLOCKS = ("denseblock3", "transition3", "denseblock4", "norm5")
        for name, param in base_model.named_parameters():
            param.requires_grad = any(block in name for block in UNFREEZE_BLOCKS)
        
        # Extract features (remove classifier)
        self.features = base_model.features
        
        # Custom classification head
        # Classification head: pool features -> 128-unit layer -> dropout -> 1 output logit (pneumonia score).
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(1024, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, 1)
    
    # Forward pass: image -> DenseNet features -> pooled -> head -> raw logit (sigmoid is applied later).
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x  # logits with shape: [batch_size, 1]


# BLOCK: Optional alternative backbone pretrained on ~100k chest X-rays (TorchXRayVision). Not used for the final model.
class DenseNet121XRV(nn.Module):
    """DenseNet121 pretrained on real chest X-rays (TorchXRayVision), instead
    of generic ImageNet, with the same custom head and selective unfreezing
    scheme as DenseNet121Custom. XRV pools ~100k CXRs across NIH ChestX-ray14,
    CheXpert, MIMIC-CXR, PadChest, Google, OpenI and Kaggle -- features here
    are already pneumonia/opacity-relevant, unlike ImageNet's natural-image
    features."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchxrayvision as xrv

        xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all" if pretrained else None)

        UNFREEZE_BLOCKS = ("denseblock3", "transition3", "denseblock4", "norm5")
        for name, param in xrv_model.features.named_parameters():
            param.requires_grad = any(block in name for block in UNFREEZE_BLOCKS)

        self.features = xrv_model.features  # expects 1-channel input, ~[-1024, 1024]

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(1024, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x  # logits with shape: [batch_size, 1]


# BLOCK: Factory that returns the chosen model (default: torchvision DenseNet121).
def build_model(pretrained: bool = True, backbone: str = None) -> nn.Module:
    """Construct the model. `backbone` must be passed explicitly by any
    caller that may run in a separate process (e.g. a Ray actor) -- see
    build_transforms() docstring."""
    backbone = backbone or BACKBONE
    if backbone == "xrv":
        return DenseNet121XRV(pretrained=pretrained)
    return DenseNet121Custom(pretrained=pretrained)


# BLOCK: Converts weights received from the server (numpy arrays) back into a PyTorch model's state_dict.
def load_state_dict_from_ndarrays(model: nn.Module, ndarrays) -> None:
    """Load model weights from numpy arrays or Flower Parameters."""
    if isinstance(ndarrays, fl.common.Parameters):
        ndarrays = fl.common.parameters_to_ndarrays(ndarrays)

    state_dict = model.state_dict()
    updated_state = OrderedDict()
    for (key, value), array in zip(state_dict.items(), ndarrays):
        updated_state[key] = torch.tensor(array, dtype=value.dtype, device=value.device)
    model.load_state_dict(updated_state, strict=True)


# BLOCK: One federated client (one 'hospital'). Holds its own private data, trains locally and returns ONLY model weights. Implements SCAFFOLD.
class ScaffoldClient(fl.client.NumPyClient):
    """Federated learning client with SCAFFOLD algorithm."""
    
    # Class variable to track current round
    current_round = 1
    strategy_ref = None
    
    def __init__(
        self,
        client_id: int,
        data_root: Path,
        batch_size: int = 16,
        local_epochs: int = 2,
        seed: int = 42,
        class1_weight: float = 1.0,
        backbone: str = "torchvision",
    ):
        self.client_id = client_id
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.class1_weight = class1_weight
        self.backbone = backbone

        # Load datasets
        train_tf, test_tf = build_transforms(self.backbone)
        train_dir = data_root / f"client_{client_id}"
        
        # Training dataset with class weights
        # Load this client's own images only (folder client_<id>); labels come from the NORMAL/PNEUMONIA subfolders.
        train_dataset = datasets.ImageFolder(str(train_dir), transform=train_tf)
        targets = torch.tensor(train_dataset.targets, dtype=torch.long)
        class_counts = torch.bincount(targets, minlength=NUM_CLASSES).float()

        # A negative class1_weight means "auto": balance sampling to this
        # client's own class ratio instead of a fixed global multiplier.
        # Clients can have very different local imbalance when the training
        # pool isn't artificially downsampled to be class-balanced.
        if self.class1_weight < 0:
            effective_class1_weight = float(class_counts[0] / max(class_counts[1], 1.0))
        else:
            effective_class1_weight = float(self.class1_weight)

        # Lower class-1 weight can reduce false positives and improve precision.
        class_weights = torch.tensor([1.0, effective_class1_weight], dtype=torch.float32)
        sample_weights = class_weights[targets].double()
        
        # Class-balanced sampling controlled by class1_weight (1.0 = plain sampling).
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        print(
            f"  [client_{client_id}] NORMAL={int(class_counts[0])} PNEUMONIA={int(class_counts[1])} "
            f"effective_class1_weight={effective_class1_weight:.3f}"
        )
        
        self.trainloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        
        # Loader used for the quick local accuracy check in evaluate().
        self.evalloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        
        # Build model
        # Each client keeps its own local copy of the model.
        self.model = build_model(pretrained=True, backbone=self.backbone).to(DEVICE)
        
        # Binary cross-entropy on logits is numerically more stable.
        # Binary cross-entropy loss on raw logits (numerically stable).
        self.criterion = nn.BCEWithLogitsLoss()
        
        # SCAFFOLD control variates
        # SCAFFOLD control variates: c_local = this client's drift estimate, c_global = the server's average. Both start at zero.
        self.c_local = [torch.zeros_like(p) for p in self.model.parameters()]
        self.c_global = [torch.zeros_like(p) for p in self.model.parameters()]
        
        # Store class counts for metrics
        self.train_counts = class_counts
    
    # Extracts ONLY the model weights as numpy arrays. This is the only thing that leaves the client; images never do.
    def get_parameters(self, config):
        return [tensor.detach().cpu().numpy() for tensor in self.model.state_dict().values()]
    
    # Loads the global weights sent by the server into this client's local model.
    def set_parameters(self, parameters):
        load_state_dict_from_ndarrays(self.model, parameters)
    
    # Local training for one federated round: train on private data, apply the SCAFFOLD correction, return updated weights.
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Get current round from class variable
        server_round = ScaffoldClient.current_round
        lr = get_dynamic_lr(server_round)
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        # AdamW optimizer on the trainable layers only, using this round's learning rate.
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        
        self.model.train()
        
        # Save initial parameters for SCAFFOLD correction
        # Remember the weights before training, to measure how far the model moves (needed for the control-variate update).
        x_before = [p.data.clone() for p in self.model.parameters()]
        
        total_loss = 0.0
        total_samples = 0
        
        # Local training loop: several passes (epochs) over this client's private images.
        for epoch in range(self.local_epochs):
            for images, labels in self.trainloader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True).float().unsqueeze(1)  # [batch_size] -> [batch_size, 1]
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(images)  # shape: [batch_size, 1]
                loss = self.criterion(logits, labels)  # both [batch_size, 1]
                
                # SCAFFOLD gradient correction
                loss.backward()
                
                # SCAFFOLD correction: add (global control - local control) to each gradient to cancel client drift.
                for p, c_l, c_g in zip(self.model.parameters(), self.c_local, self.c_global):
                    if p.grad is not None:
                        p.grad.data.add_(c_g - c_l)
                
                optimizer.step()
                
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
        
        # Update local control variate (SCAFFOLD Option II)
        # Update the local control variate (SCAFFOLD Option II) using the distance the weights moved.
        steps = self.local_epochs * len(self.trainloader)
        for c_l, c_g, x_b, p in zip(self.c_local, self.c_global, x_before, self.model.parameters()):
            c_l.data = c_l.data - c_g + (x_b - p.data) / (steps * lr + 1e-8)
        
        avg_loss = total_loss / max(total_samples, 1)
        print(f"  [client_{self.client_id}] round={server_round} loss={avg_loss:.4f} lr={lr:.6f}")
        
        # Send back ONLY the updated weights, the number of training samples and the average loss.
        return self.get_parameters(config={}), total_samples, {"loss": avg_loss}
    
    # Quick local accuracy check on this client's own data (progress logging only).
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.evalloader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                
                logits = self.model(images)  # shape: [batch_size, 1]
                probs = torch.sigmoid(logits)
                preds = (probs >= PREDICTION_THRESHOLD).long().squeeze(1)  # [batch_size, 1] -> [batch_size]
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / max(total, 1)
        return 0.0, total, {"accuracy": accuracy}


# BLOCK: Learning-rate schedule: 3e-4 for rounds 1-5, 2e-4 for rounds 6-10, 1e-4 afterwards.
def get_dynamic_lr(round_num: int) -> float:
    """Get learning rate based on round number."""
    if round_num <= 5:
        return 0.0003  # 3e-4
    elif round_num <= 10:
        return 0.0002  # 2e-4
    else:
        return 0.0001  # 1e-4


# BLOCK: The server / 'global model'. Extends FedAvg to average client weights, then tracks per-round metrics and remembers the best round.
class CustomStrategy(fl.server.strategy.FedAvg):
    """Custom strategy that tracks final parameters and evaluates per round."""
    
    def __init__(self, test_model=None, test_loader=None, validation_loader=None, num_rounds=15,
                 selection_metric="accuracy", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None
        self.test_model = test_model
        self.test_loader = test_loader
        self.validation_loader = validation_loader
        self.num_rounds = num_rounds
        self.round_metrics = []
        # Best-round tracking: the last round is not always the best one
        # (later rounds can drift/overfit), so we keep whichever round scored
        # highest on the held-out validation split and use that as the final
        # model instead of always taking the last round's weights.
        # `selection_metric` picks which of (accuracy, precision, recall, f1)
        # from evaluate_round_metrics() drives this choice.
        self.selection_metric = selection_metric
        self.best_val_score = -1.0
        self.best_round = None
        self.best_state_dict = None

    # Called after every round: averages all clients' weights (the global model), logs test progress and saves the best round by validation score.
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate training results and evaluate on test set."""
        # Update the client class variable with current round
        ScaffoldClient.current_round = server_round

        # FedAvg step: weighted average of all clients' weights = the new global model.
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters

            # Evaluate on test set if available
            if self.test_model is not None and self.test_loader is not None:
                try:
                    load_state_dict_from_ndarrays(self.test_model, aggregated_parameters)
                    acc, prec, rec, f1 = evaluate_round_metrics(
                        self.test_model, self.test_loader, server_round
                    )
                    self.round_metrics.append({
                        "round": server_round,
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1_score": f1,
                    })
                    # Persist per-round metrics so progress is not lost if terminal output is truncated.
                    with open("scaffold_round_metrics.json", "w", encoding="utf-8") as f:
                        json.dump(self.round_metrics, f, indent=2)
                except Exception as e:
                    print(f"  Warning: Per-round evaluation failed for round {server_round}: {e}")

            # Track the best round using the held-out validation split (not
            # the test set, so model selection doesn't leak from the test data).
            if self.test_model is not None and self.validation_loader is not None:
                try:
                    load_state_dict_from_ndarrays(self.test_model, aggregated_parameters)
                    val_acc, val_prec, val_rec, val_f1 = evaluate_round_metrics(
                        self.test_model, self.validation_loader, server_round
                    )
                    val_scores = {"accuracy": val_acc, "precision": val_prec, "recall": val_rec, "f1": val_f1}
                    val_score = val_scores[self.selection_metric]
                    if val_score > self.best_val_score:
                        self.best_val_score = val_score
                        self.best_round = server_round
                        self.best_state_dict = copy.deepcopy(self.test_model.state_dict())
                        print(f"  [Round {server_round}] New best validation {self.selection_metric}={val_score:.4f} — checkpoint saved")
                except Exception as e:
                    print(f"  Warning: Validation-based best-round tracking failed for round {server_round}: {e}")

        return aggregated_parameters, aggregated_metrics


# BLOCK: Final evaluation of the saved model on a held-out test set: accuracy, precision, recall, F1, confusion matrix and goal check.
def evaluate_on_test_set(model_path: Path, test_data_root: Path, batch_size: int = 16, threshold: float = PREDICTION_THRESHOLD, label: str = ""):
    """Evaluate model on test set and print metrics."""
    print("\n" + "=" * 80)
    print(f"FINAL TEST SET EVALUATION{f' ({label})' if label else ''}")
    print("=" * 80)
    
    # Build model and load weights
    model = build_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Load test dataset
    _, test_tf = build_transforms()
    test_dataset = RecursiveLabelDataset(test_data_root, transform=test_tf)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            
            logits = model(images)  # shape: [batch_size, 1]
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long().squeeze(1).cpu().numpy()  # [batch_size, 1] -> [batch_size]
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)
    
    # Print results
    print(f"\nFinal Accuracy:  {accuracy:.4f}")
    print(f"Final Precision: {precision:.4f}")
    print(f"Final Recall:    {recall:.4f}")
    print(f"Final F1-score:  {f1:.4f}")
    print(f"Decision Threshold: {threshold:.2f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(report)
    
    # Print goal comparison
    print("\n" + "=" * 80)
    print("GOALS vs ACHIEVED")
    print("=" * 80)
    goals = {
        "Accuracy": (0.90, accuracy),
        "Precision": (0.85, precision),
        "Recall": (0.85, recall),
        "F1-score": (0.85, f1),
    }
    
    for metric, (goal, achieved) in goals.items():
        status = "✓ PASSED" if achieved >= goal else "✗ FAILED"
        print(f"{metric:12} Goal: {goal:.2%}  Achieved: {achieved:.2%}  {status}")
    
    print("=" * 80)
    
    return accuracy, precision, recall, f1


# BLOCK: Scores a model on a data loader (used each round for test progress and validation-based round selection).
def evaluate_round_metrics(model, test_loader, round_num):
    """Evaluate model on test set and return per-round metrics."""
    try:
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE, non_blocking=True)
                logits = model(images)  # shape: [batch_size, 1]
                probs = torch.sigmoid(logits)
                preds = (probs >= PREDICTION_THRESHOLD).long().squeeze(1).cpu().numpy()  # [batch_size, 1] -> [batch_size]
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        if len(all_preds) == 0 or len(all_labels) == 0:
            print(f"  [Round {round_num}] No predictions or labels")
            return 0.0, 0.0, 0.0, 0.0
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print(f"  [Round {round_num}] Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        return accuracy, precision, recall, f1
    except Exception as e:
        print(f"  [Round {round_num}] Evaluation error: {e}")
        return 0.0, 0.0, 0.0, 0.0


# BLOCK: Entry point. Parses options, builds data loaders, runs the federated simulation, saves the best model, picks the threshold and evaluates on the test set(s).
def main():
    # Command-line options: clients, rounds, epochs, data/test/validation folders, thresholds, etc.
    parser = argparse.ArgumentParser(description="SCAFFOLD Federated Learning - Pneumonia Detection")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of federated clients")
    parser.add_argument("--rounds", type=int, default=20, help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=2, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--data-root", type=Path, default=Path("dataset") / "train", help="Training data root")
    parser.add_argument("--test-root", type=Path, default=Path("dataset") / "test", help="Test data root")
    parser.add_argument(
        "--test-root-2",
        type=Path,
        default=None,
        help="Optional second, disjoint test folder evaluated separately from --test-root and "
             "reported as its own set of metrics (e.g. a pediatric test set alongside an adult "
             "one), so subgroup performance isn't hidden inside one pooled number.",
    )
    parser.add_argument(
        "--test-root-2-label",
        type=str,
        default="secondary",
        help="Label used in output/logging for --test-root-2 (e.g. 'pediatric').",
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=None,
        help="Dedicated held-out validation folder (never trained on). If omitted, falls back to "
             "carving a subset out of --data-root (legacy behavior, has train/validation overlap).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-model",
        type=Path,
        default=Path("federated_model_scaffold.pth"),
        help="Path to save final model"
    )
    parser.add_argument(
        "--gpu-per-client",
        type=float,
        default=1.0,
        help="GPU fraction per client"
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of training data reserved for threshold tuning.",
    )
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="fbeta_precision",
        choices=["accuracy", "precision", "recall", "f1", "fbeta_precision"],
        help="Validation metric used to choose the decision threshold. "
             "'fbeta_precision' (F-beta, beta=0.5) weights precision higher than recall, "
             "which reduces false alarms.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="If set, skips the validation threshold sweep entirely and uses this fixed "
             "decision threshold for final test evaluation instead.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "precision", "recall", "f1"],
        help="Validation metric used to pick which round's weights become the final model.",
    )
    parser.add_argument(
        "--class1-weight",
        type=float,
        default=1.0,
        help="Sampling weight for class 1 (PNEUMONIA). Lower values usually increase precision.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="torchvision",
        choices=["torchvision", "xrv"],
        help="'torchvision': DenseNet121 with generic ImageNet pretraining (original). "
             "'xrv': DenseNet121 pretrained on ~100k real chest X-rays (TorchXRayVision) "
             "for domain-specific features.",
    )

    args = parser.parse_args()

    global BACKBONE
    BACKBONE = args.backbone

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING - PNEUMONIA DETECTION (SCAFFOLD)")
    print("=" * 80)
    print(f"Device           : {DEVICE}")
    print(f"Num Clients      : {args.num_clients}")
    print(f"Rounds           : {args.rounds}")
    print(f"Local Epochs     : {args.local_epochs}")
    print(f"Batch Size       : {args.batch_size}")
    print(f"Input Size       : {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Threshold        : {PREDICTION_THRESHOLD}")
    print(f"Class Weights    : {{0: 1.0, 1: {args.class1_weight}}}")
    print("=" * 80 + "\n")

    _, test_tf = build_transforms()
    if args.validation_root is not None:
        validation_loader = build_validation_loader_from_dir(
            args.validation_root,
            transform=test_tf,
            batch_size=args.batch_size,
        )
    else:
        validation_loader = build_validation_loader(
            args.data_root,
            transform=test_tf,
            batch_size=args.batch_size,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
    
    # Prepare test data for per-round evaluation
    _, test_tf = build_transforms()
    test_dataset = RecursiveLabelDataset(args.test_root, transform=test_tf)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_model = build_model(pretrained=False).to(DEVICE)

    # Factory Flower calls to create client number <cid> (reads folder client_<cid>).
    def client_fn(cid: str) -> fl.client.NumPyClient:
        return ScaffoldClient(
            client_id=int(cid),
            data_root=args.data_root,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            seed=args.seed,
            class1_weight=args.class1_weight,
            backbone=args.backbone,
        ).to_client()
    
    # Create the server strategy (global-model aggregator).
    strategy = CustomStrategy(
        test_model=test_model,
        test_loader=test_loader,
        validation_loader=validation_loader,
        num_rounds=args.rounds,
        selection_metric=args.selection_metric,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=0,
        min_available_clients=args.num_clients,
    )
    
    client_resources = {
        "num_cpus": 2,
    }
    if torch.cuda.is_available():
        client_resources["num_gpus"] = args.gpu_per_client
    
    # Run federated training: simulate all clients for the requested number of rounds.
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    if strategy.round_metrics:
        with open("scaffold_round_metrics.json", "w", encoding="utf-8") as f:
            json.dump(strategy.round_metrics, f, indent=2)

        with open("scaffold_round_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["round", "accuracy", "precision", "recall", "f1_score"],
            )
            writer.writeheader()
            writer.writerows(strategy.round_metrics)
        print("Per-round metrics saved -> scaffold_round_metrics.json")
        print("Per-round metrics saved -> scaffold_round_metrics.csv")
    
    # Save final model — prefer the best round on the held-out validation
    # split over the last round, since later rounds are not guaranteed to be
    # better (client drift / overfitting can make a mid-training round best).
    if strategy.final_parameters is None:
        raise RuntimeError("No final aggregated parameters captured.")

    # Load the best round's weights (by validation score) as the final global model.
    final_model = build_model(pretrained=False)
    if strategy.best_state_dict is not None:
        print(
            f"Using round {strategy.best_round} (best validation {strategy.selection_metric}={strategy.best_val_score:.4f}) "
            f"instead of the last round ({args.rounds})."
        )
        final_model.load_state_dict(strategy.best_state_dict)
    else:
        print("No best-round checkpoint found; falling back to the last round's weights.")
        load_state_dict_from_ndarrays(final_model, strategy.final_parameters)

    print(f"Saving final model to: {args.save_model}")
    # Save the final global model to disk (.pth file).
    torch.save(final_model.state_dict(), args.save_model)
    print(f"Model saved -> {args.save_model}\n")

    # Choose the decision threshold: use the fixed value if given, otherwise sweep on validation data.
    if args.fixed_threshold is not None:
        best_threshold = args.fixed_threshold
        print(f"Using fixed decision threshold: {best_threshold:.2f} (validation sweep skipped)")
    else:
        best_threshold, val_metrics = find_best_threshold(final_model.to(DEVICE), validation_loader, metric=args.threshold_metric)
        print("Validation threshold tuning:")
        print(
            f"  best_threshold={best_threshold:.2f} "
            f"val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )
    
    # Evaluate on test set(s). A second, disjoint test set (--test-root-2) is
    # evaluated separately -- not pooled with the first -- so subgroup
    # performance (e.g. pediatric vs adult) is reported on its own rather
    # than hidden inside one averaged number.
    # Evaluate on the primary test set.
    final_acc, final_prec, final_rec, final_f1 = evaluate_on_test_set(
        args.save_model, args.test_root, batch_size=args.batch_size, threshold=best_threshold
    )

    final_metrics = {
        "accuracy": final_acc,
        "precision": final_prec,
        "recall": final_rec,
        "f1_score": final_f1,
        "model_path": str(args.save_model),
        "rounds": args.rounds,
        "batch_size": args.batch_size,
        "local_epochs": args.local_epochs,
        "threshold": best_threshold,
    }

    # Optionally evaluate on a second test set (e.g. pediatric) and report it separately.
    if args.test_root_2 is not None:
        acc2, prec2, rec2, f1_2 = evaluate_on_test_set(
            args.save_model, args.test_root_2, batch_size=args.batch_size,
            threshold=best_threshold, label=args.test_root_2_label,
        )
        final_metrics[args.test_root_2_label] = {
            "accuracy": acc2,
            "precision": prec2,
            "recall": rec2,
            "f1_score": f1_2,
            "test_root": str(args.test_root_2),
        }

    # Save final metrics to JSON (read later by the Streamlit app).
    with open("scaffold_final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print("Final metrics saved -> scaffold_final_metrics.json")


if __name__ == "__main__":
    main()
