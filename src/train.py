"""
Reusable CellCNN training.

    python -m src.train
    python -m src.train --epochs 30 --lr 5e-4 --exp-name baseline

notebooks/04_training_demo.ipynb calls the same functions defined here for
interactive exploration — patch extraction, the training loop, checkpointing,
and metrics all live in this one place instead of being duplicated in a
notebook.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from src import config
from src.logging_utils import setup_logging
from src.seed import seed_everything
from src.dataset import BioHubDataset, PatchDataset, extract_patches, build_patch_arrays
from src.detector import CellDetector
from src.model import CellCNN
from src.losses import get_criterion
from src.experiment import Experiment
from src.metrics import classification_report, plot_confusion_matrix

logger = setup_logging()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="train", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images).squeeze(1)  # raw logits
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = (outputs >= 0).float()  # logit >= 0  <=>  sigmoid(logit) >= 0.5
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader), correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="val", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images).squeeze(1)  # raw logits
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        predictions = (outputs >= 0).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader), correct / total


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run the model over a loader and return (y_true, y_prob) numpy arrays —
    probabilities, not logits, since classification_report expects [0, 1]."""
    model.eval()

    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_probs)


def build_dataset(sample_name=None, frame_idx=0, seed=None):
    """Detect candidate centers on one frame's max-projection and extract
    labeled positive/negative patches around them."""
    dataset = BioHubDataset(config.DATASET_PATH)
    sample = sample_name or dataset.train_samples[0]

    volume = np.asarray(dataset.load_volume(sample))
    projection = volume[frame_idx].max(axis=0)

    detector = CellDetector(
        sigma=config.GAUSSIAN_SIGMA,
        threshold_abs=config.DETECTION_THRESHOLD,
        min_distance=config.CELL_RADIUS,
    )
    centers = detector.detect(projection)

    positive_patches, negative_patches = extract_patches(projection, centers)
    X, y = build_patch_arrays(
        positive_patches, negative_patches,
        seed=seed if seed is not None else config.RANDOM_SEED,
    )

    logger.info(
        f"Sample={sample} frame={frame_idx}: {len(positive_patches)} positive / "
        f"{len(negative_patches)} negative patches"
    )
    return X, y


def run_training(argv=None):
    parser = argparse.ArgumentParser(description="Train CellCNN on extracted patches.")
    parser.add_argument("--sample", default=None, help="Training sample name (default: first available)")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--exp-name", default=None, help="experiments/<exp-name>/ (default: auto exp001, exp002, ...)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    logger.info(f"Seeded run with seed={args.seed} (see src/seed.py)")

    exp = Experiment(name=args.exp_name)
    exp.save_config(extra=vars(args))
    logger.info(f"Experiment directory: {exp.dir}")

    X, y = build_dataset(args.sample, args.frame, seed=args.seed)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.VAL_SPLIT, random_state=args.seed, stratify=y,
    )

    # Normalize using TRAIN statistics only (see notebooks/04_training_demo.ipynb
    # for why this specific step is the difference between a model that learns
    # and one that predicts a constant class).
    train_mean, train_std = float(X_train.mean()), float(X_train.std())
    X_train = (X_train - train_mean) / (train_std + 1e-8)
    X_val = (X_val - train_mean) / (train_std + 1e-8)
    np.savez(exp.dir / "norm_stats.npz", mean=train_mean, std=train_std)

    train_loader = DataLoader(
        PatchDataset(X_train, y_train, augment=config.USE_AUGMENTATION),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        PatchDataset(X_val, y_val, augment=False),
        batch_size=args.batch_size, shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = CellCNN().to(device)
    criterion = get_criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
        min_lr=config.LR_SCHEDULER_MIN_LR,
    )

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in tqdm(range(args.epochs), desc="epochs"):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.3f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.3f} | LR={lr_now:.2e}"
        )
        exp.log_epoch(epoch + 1, train_loss, train_acc, val_loss, val_acc, lr_now)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                exp.dir / "model.pth",
            )
            logger.info(f"  New best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            logger.info("Validation stopped improving. Stopping...")
            break

    exp.save_metrics_csv()
    exp.save_plots(train_losses, val_losses, train_accs, val_accs)

    # Reload the BEST checkpoint (not necessarily the last epoch) for the
    # final report, since the last epoch may already be slightly overfit.
    state = torch.load(exp.dir / "model.pth", map_location=device)
    model.load_state_dict(state["model_state_dict"])

    y_true, y_prob = collect_predictions(model, val_loader, device)
    report = classification_report(y_true, y_prob)
    logger.info(
        f"Final validation metrics — Precision={report['precision']:.3f} "
        f"Recall={report['recall']:.3f} F1={report['f1']:.3f} "
        f"ROC-AUC={report['roc_auc']:.3f}"
    )
    exp.save_report(report)
    plot_confusion_matrix(report["confusion_matrix"], exp.dir / "confusion_matrix.png")

    exp.promote_to_default()
    logger.info(f"Experiment complete: {exp.dir}")

    return exp


if __name__ == "__main__":
    run_training()
