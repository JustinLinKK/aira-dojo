"""Deterministic generated-solution workload for MLE-Bench aerial cactus.

This file is intentionally self-contained because it is executed as generated
solution code through MLEBenchTask.step_task. Runtime knobs are read from
environment variables so the same script can serve smoke, calibration,
acceptance, stress, and repeatability runs.
"""

import glob
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path("./data")
OUTPUT_PATH = Path("submission.csv")


def getenv_int(name, default):
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def getenv_float(name, default):
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


PARALLEL_MODELS = getenv_int("AERIAL_CACTUS_PARALLEL_MODELS", 4)
EPOCHS = getenv_int("AERIAL_CACTUS_EPOCHS", 8)
BATCH_SIZE = getenv_int("AERIAL_CACTUS_BATCH_SIZE", 256)
IMAGE_SIZE = getenv_int("AERIAL_CACTUS_IMAGE_SIZE", 64)
DATALOADER_WORKERS = getenv_int("AERIAL_CACTUS_DATALOADER_WORKERS", 2)
BASE_SEED = getenv_int("AERIAL_CACTUS_BASE_SEED", 20260415)
VALID_FRACTION = getenv_float("AERIAL_CACTUS_VALID_FRACTION", 0.2)
TRAIN_REPEAT = getenv_int("AERIAL_CACTUS_TRAIN_REPEAT", 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_first(patterns):
    for pattern in patterns:
        matches = sorted(DATA_DIR.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find any of: {patterns}")


def image_index():
    paths = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(Path(path) for path in glob.glob(str(DATA_DIR / "**" / suffix), recursive=True))
    by_name = {}
    for path in paths:
        by_name[path.name] = path
        by_name[path.stem] = path
    return by_name


def resolve_image_path(image_id, by_name):
    image_id = str(image_id)
    candidates = [image_id, Path(image_id).name, Path(image_id).stem, f"{image_id}.jpg", f"{Path(image_id).stem}.jpg"]
    for candidate in candidates:
        if candidate in by_name:
            return by_name[candidate]
    raise FileNotFoundError(f"Could not resolve image id {image_id}")


def load_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


class CactusDataset(Dataset):
    def __init__(self, paths, labels=None, augment=False):
        self.paths = list(paths)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = load_image(self.paths[index])
        if self.augment:
            if random.random() < 0.5:
                array = array[:, :, ::-1].copy()
            if random.random() < 0.5:
                array = array[:, ::-1, :].copy()
        tensor = torch.tensor(array, dtype=torch.float32)
        if self.labels is None:
            return tensor
        return tensor, torch.tensor([self.labels[index]], dtype=torch.float32)


class SmallCactusNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.15), nn.Linear(192, 1))

    def forward(self, x):
        return self.classifier(self.features(x))


def split_indices(labels, seed):
    indices = np.arange(len(labels))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    valid_size = max(1, int(len(indices) * VALID_FRACTION))
    valid_idx = indices[:valid_size]
    train_idx = indices[valid_size:]
    if len(train_idx) == 0:
        train_idx = valid_idx
    return train_idx, valid_idx


def autocast_context(device):
    if device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=torch.float16)


def make_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=DATALOADER_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=DATALOADER_WORKERS > 0,
        generator=generator,
    )


def train_one_model(worker_id, train_paths, labels, test_paths):
    seed = BASE_SEED + worker_id * 9973
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(device)

    train_idx, valid_idx = split_indices(labels, seed)
    expanded_train_idx = np.tile(train_idx, TRAIN_REPEAT)
    train_dataset = CactusDataset(
        [train_paths[i] for i in expanded_train_idx],
        labels[expanded_train_idx],
        augment=True,
    )
    valid_dataset = CactusDataset([train_paths[i] for i in valid_idx], labels[valid_idx], augment=False)
    test_dataset = CactusDataset(test_paths, augment=False)

    train_loader = make_loader(train_dataset, BATCH_SIZE, True, seed)
    valid_loader = make_loader(valid_dataset, BATCH_SIZE * 2, False, seed)
    test_loader = make_loader(test_dataset, BATCH_SIZE * 2, False, seed)

    model = SmallCactusNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
        print(f"worker={worker_id} epoch={epoch + 1}/{EPOCHS} loss={running_loss / max(1, len(train_loader)):.5f}")

    model.eval()
    valid_predictions = []
    valid_targets = []
    test_predictions = []
    with torch.no_grad():
        for images, targets in valid_loader:
            images = images.to(device, non_blocking=True)
            with autocast_context(device):
                logits = model(images)
            valid_predictions.extend(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).tolist())
            valid_targets.extend(targets.numpy().reshape(-1).tolist())
        for images in test_loader:
            images = images.to(device, non_blocking=True)
            with autocast_context(device):
                logits = model(images)
            test_predictions.extend(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).tolist())

    valid_predictions_np = np.asarray(valid_predictions)
    valid_targets_np = np.asarray(valid_targets)
    accuracy = float(((valid_predictions_np >= 0.5) == (valid_targets_np >= 0.5)).mean())
    elapsed = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    print(f"worker={worker_id} done seconds={elapsed:.2f} valid_accuracy={accuracy:.5f} peak_cuda_mb={peak_mb:.1f}")
    return np.asarray(test_predictions, dtype=np.float32), {
        "accuracy": accuracy,
        "seconds": elapsed,
        "peak_cuda_mb": peak_mb,
    }


def load_problem():
    train_csv = find_first(["**/train.csv"])
    sample_submission = find_first(["**/sample_submission.csv", "**/sample_submission*.csv"])
    train_df = pd.read_csv(train_csv)
    sample_df = pd.read_csv(sample_submission)

    id_col = "id" if "id" in train_df.columns else train_df.columns[0]
    label_candidates = [col for col in train_df.columns if col != id_col]
    if not label_candidates:
        raise ValueError(f"No label column found in {train_csv}")
    label_col = "has_cactus" if "has_cactus" in label_candidates else label_candidates[0]

    sample_id_col = "id" if "id" in sample_df.columns else sample_df.columns[0]
    pred_candidates = [col for col in sample_df.columns if col != sample_id_col]
    if not pred_candidates:
        raise ValueError(f"No prediction column found in {sample_submission}")
    pred_col = "has_cactus" if "has_cactus" in pred_candidates else pred_candidates[0]

    by_name = image_index()
    train_paths = [resolve_image_path(image_id, by_name) for image_id in train_df[id_col].tolist()]
    test_paths = [resolve_image_path(image_id, by_name) for image_id in sample_df[sample_id_col].tolist()]
    labels = train_df[label_col].astype(float).to_numpy()
    return sample_df, sample_id_col, pred_col, train_paths, labels, test_paths


def main():
    print(
        "aerial-cactus workload "
        f"parallel_models={PARALLEL_MODELS} epochs={EPOCHS} batch_size={BATCH_SIZE} "
        f"image_size={IMAGE_SIZE} dataloader_workers={DATALOADER_WORKERS} train_repeat={TRAIN_REPEAT}"
    )
    sample_df, sample_id_col, pred_col, train_paths, labels, test_paths = load_problem()
    print(f"loaded train_images={len(train_paths)} test_images={len(test_paths)} positive_rate={labels.mean():.5f}")

    predictions = []
    metrics = []
    with ThreadPoolExecutor(max_workers=PARALLEL_MODELS) as executor:
        futures = [
            executor.submit(train_one_model, worker_id, train_paths, labels, test_paths)
            for worker_id in range(PARALLEL_MODELS)
        ]
        for future in as_completed(futures):
            worker_predictions, worker_metrics = future.result()
            predictions.append(worker_predictions)
            metrics.append(worker_metrics)

    ensemble_predictions = np.mean(np.stack(predictions, axis=0), axis=0)
    submission = sample_df[[sample_id_col]].copy()
    submission[pred_col] = ensemble_predictions
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH} rows={len(submission)}")
    print(f"worker_metrics={metrics}")


main()
