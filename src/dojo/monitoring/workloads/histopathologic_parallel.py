"""Deterministic generated-solution workload for histopathologic cancer detection.

This file is executed as generated solution code through MLEBenchTask.step_task.
It intentionally trains several independent GPU models at once so the resource
monitor can measure a long, realistic generated-code training window.
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


def workload_int(name, default):
    return getenv_int(f"HISTOPATHOLOGIC_{name}", getenv_int(f"MLEBENCH_WORKLOAD_{name}", default))


def workload_float(name, default):
    return getenv_float(f"HISTOPATHOLOGIC_{name}", getenv_float(f"MLEBENCH_WORKLOAD_{name}", default))


PARALLEL_MODELS = workload_int("PARALLEL_MODELS", 4)
EPOCHS = workload_int("EPOCHS", 3)
BATCH_SIZE = workload_int("BATCH_SIZE", 512)
IMAGE_SIZE = workload_int("IMAGE_SIZE", 96)
REQUESTED_DATALOADER_WORKERS = workload_int("DATALOADER_WORKERS", 0)
BASE_SEED = workload_int("BASE_SEED", 20260415)
VALID_FRACTION = workload_float("VALID_FRACTION", 0.08)
TRAIN_REPEAT = workload_int("TRAIN_REPEAT", 1)


def available_shm_mb():
    try:
        stat = os.statvfs("/dev/shm")
    except OSError:
        return 0.0
    return float(stat.f_bavail * stat.f_frsize / (1024 * 1024))


SHM_AVAILABLE_MB = available_shm_mb()
DATALOADER_WORKERS = 0 if SHM_AVAILABLE_MB < 1024 else REQUESTED_DATALOADER_WORKERS


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
    for suffix in ("*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"):
        paths.extend(Path(path) for path in glob.glob(str(DATA_DIR / "**" / suffix), recursive=True))
    by_name = {}
    for path in paths:
        by_name[path.name] = path
        by_name[path.stem] = path
    return by_name


def resolve_image_path(image_id, by_name):
    image_id = str(image_id)
    stem = Path(image_id).stem
    candidates = [
        image_id,
        Path(image_id).name,
        stem,
        f"{image_id}.tif",
        f"{stem}.tif",
        f"{image_id}.jpg",
        f"{stem}.jpg",
    ]
    for candidate in candidates:
        if candidate in by_name:
            return by_name[candidate]
    raise FileNotFoundError(f"Could not resolve image id {image_id}")


def load_image(path, augment):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.asarray(image, dtype=np.float32) / 255.0

    if augment:
        if random.random() < 0.5:
            array = array[:, ::-1, :].copy()
        if random.random() < 0.5:
            array = array[::-1, :, :].copy()
        if random.random() < 0.5:
            k = random.randint(1, 3)
            array = np.rot90(array, k=k).copy()
        if random.random() < 0.35:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            mean = array.mean(axis=(0, 1), keepdims=True)
            array = np.clip((array - mean) * contrast + mean, 0.0, 1.0)
            array = np.clip(array * brightness, 0.0, 1.0)

    return np.transpose(array, (2, 0, 1))


class HistologyDataset(Dataset):
    def __init__(self, paths, labels=None, augment=False):
        self.paths = list(paths)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = load_image(self.paths[index], self.augment)
        tensor = torch.tensor(array, dtype=torch.float32)
        if self.labels is None:
            return tensor
        return tensor, torch.tensor([self.labels[index]], dtype=torch.float32)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.net(x)


class HistologyNet(nn.Module):
    def __init__(self, width=64, depth=4, dropout=0.15):
        super().__init__()
        channels = [width * (2**i) for i in range(depth)]
        layers = []
        in_channels = 3
        for index, out_channels in enumerate(channels):
            layers.append(ConvBlock(in_channels, out_channels, stride=1 if index == 0 else 2, dropout=dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], max(width * 2, 64)),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(width * 2, 64), 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_model(worker_id):
    variants = [
        {"width": 64, "depth": 4, "dropout": 0.12},
        {"width": 72, "depth": 4, "dropout": 0.18},
        {"width": 80, "depth": 4, "dropout": 0.15},
        {"width": 64, "depth": 5, "dropout": 0.20},
        {"width": 96, "depth": 4, "dropout": 0.16},
        {"width": 72, "depth": 5, "dropout": 0.22},
        {"width": 104, "depth": 4, "dropout": 0.18},
        {"width": 80, "depth": 5, "dropout": 0.20},
    ]
    return HistologyNet(**variants[worker_id % len(variants)])


def split_indices(labels, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    positives = indices[labels >= 0.5]
    negatives = indices[labels < 0.5]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    valid_pos = positives[: max(1, int(len(positives) * VALID_FRACTION))]
    valid_neg = negatives[: max(1, int(len(negatives) * VALID_FRACTION))]
    valid_idx = np.concatenate([valid_pos, valid_neg])
    valid_set = set(valid_idx.tolist())
    train_idx = np.asarray([idx for idx in indices if idx not in valid_set], dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
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
        prefetch_factor=2 if DATALOADER_WORKERS > 0 else None,
        generator=generator,
    )


def binary_auc(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    positives = targets >= 0.5
    positive_count = int(positives.sum())
    negative_count = int(len(targets) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(predictions)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(predictions) + 1)
    positive_rank_sum = ranks[positives].sum()
    return float((positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count))


def train_one_model(worker_id, train_paths, labels, test_paths):
    seed = BASE_SEED + worker_id * 10007
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(device)
        stream = torch.cuda.Stream(device)
    else:
        stream = None

    train_idx, valid_idx = split_indices(labels, seed)
    expanded_train_idx = np.tile(train_idx, TRAIN_REPEAT)
    train_dataset = HistologyDataset(
        [train_paths[i] for i in expanded_train_idx],
        labels[expanded_train_idx],
        augment=True,
    )
    valid_dataset = HistologyDataset([train_paths[i] for i in valid_idx], labels[valid_idx], augment=False)
    test_dataset = HistologyDataset(test_paths, augment=False)

    train_loader = make_loader(train_dataset, BATCH_SIZE, True, seed)
    valid_loader = make_loader(valid_dataset, BATCH_SIZE * 2, False, seed)
    test_loader = make_loader(test_dataset, BATCH_SIZE * 2, False, seed)

    model = make_model(worker_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if stream is None:
                context = nullcontext()
            else:
                context = torch.cuda.stream(stream)
            with context:
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device):
                    logits = model(images)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if stream is not None:
                stream.synchronize()
            running_loss += float(loss.detach().cpu())
            batch_count += 1
        if stream is not None:
            stream.synchronize()
        print(f"worker={worker_id} epoch={epoch + 1}/{EPOCHS} loss={running_loss / max(1, batch_count):.5f}")

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
    auc = binary_auc(valid_targets_np, valid_predictions_np)
    elapsed = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    print(
        f"worker={worker_id} done seconds={elapsed:.2f} "
        f"valid_accuracy={accuracy:.5f} valid_auc={auc:.5f} peak_cuda_mb={peak_mb:.1f}"
    )
    return np.asarray(test_predictions, dtype=np.float32), {
        "accuracy": accuracy,
        "auc": auc,
        "seconds": elapsed,
        "peak_cuda_mb": peak_mb,
    }


def load_problem():
    train_csv = find_first(["**/train_labels.csv", "**/train.csv"])
    sample_submission = find_first(["**/sample_submission.csv", "**/sample_submission*.csv"])
    train_df = pd.read_csv(train_csv)
    sample_df = pd.read_csv(sample_submission)

    id_col = "id" if "id" in train_df.columns else train_df.columns[0]
    label_col = "label" if "label" in train_df.columns else [col for col in train_df.columns if col != id_col][0]
    sample_id_col = "id" if "id" in sample_df.columns else sample_df.columns[0]
    pred_col = "label" if "label" in sample_df.columns else [col for col in sample_df.columns if col != sample_id_col][0]

    by_name = image_index()
    train_paths = [resolve_image_path(image_id, by_name) for image_id in train_df[id_col].tolist()]
    test_paths = [resolve_image_path(image_id, by_name) for image_id in sample_df[sample_id_col].tolist()]
    labels = train_df[label_col].astype(float).to_numpy()
    return sample_df, sample_id_col, pred_col, train_paths, labels, test_paths


def main():
    print(
        "histopathologic workload "
        f"parallel_models={PARALLEL_MODELS} epochs={EPOCHS} batch_size={BATCH_SIZE} "
        f"image_size={IMAGE_SIZE} dataloader_workers={DATALOADER_WORKERS} "
        f"requested_dataloader_workers={REQUESTED_DATALOADER_WORKERS} "
        f"shm_available_mb={SHM_AVAILABLE_MB:.1f} train_repeat={TRAIN_REPEAT}"
    )
    sample_df, sample_id_col, pred_col, train_paths, labels, test_paths = load_problem()
    print(
        f"loaded train_images={len(train_paths)} test_images={len(test_paths)} "
        f"positive_rate={labels.mean():.5f}"
    )

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
    submission[pred_col] = np.clip(ensemble_predictions, 0.0, 1.0)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH} rows={len(submission)}")
    print(f"worker_metrics={metrics}")


main()
