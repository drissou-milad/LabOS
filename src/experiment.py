"""
Lightweight experiment tracking — no external service required.

Each call to Experiment() creates a new experiments/expNNN/ folder:

    experiments/
        exp001/
            config.json           # every setting used for this run
            metrics.csv            # per-epoch train/val loss & accuracy
            loss.png                 # loss curves
            accuracy.png               # accuracy curves
            confusion_matrix.png         # final validation confusion matrix
            report.json                   # final precision/recall/f1/roc_auc
            model.pth                       # best checkpoint for this run
        exp002/
            ...

This makes runs reproducible and comparable — you can always answer
"what config produced this loss curve?" by reading one folder.
"""

import csv
import json
from pathlib import Path

from src import config


class Experiment:

    def __init__(self, base_dir=None, name=None):
        self.base_dir = Path(base_dir or "experiments")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if name is None:
            name = self._next_name()

        self.dir = self.base_dir / name
        self.dir.mkdir(parents=True, exist_ok=True)

        self._metrics_rows = []

    def _next_name(self):
        existing = [p for p in self.base_dir.glob("exp*") if p.is_dir()]
        next_id = len(existing) + 1
        return f"exp{next_id:03d}"

    # ----------------------------------------------------------------
    # Config
    # ----------------------------------------------------------------

    def save_config(self, extra=None):
        """Dump every UPPERCASE constant in config.py, plus any run-specific
        overrides (e.g. CLI args), to config.json."""
        cfg = {k: str(v) for k, v in vars(config).items() if k.isupper()}
        if extra:
            cfg.update({k: str(v) for k, v in extra.items()})

        with open(self.dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    # ----------------------------------------------------------------
    # Per-epoch metrics
    # ----------------------------------------------------------------

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        self._metrics_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
        })

    def save_metrics_csv(self):
        path = self.dir / "metrics.csv"
        fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._metrics_rows)

        return path

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------

    def save_plots(self, train_losses, val_losses, train_accs, val_accs):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.dir / "loss.png")
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(train_accs, label="train")
        plt.plot(val_accs, label="val")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.dir / "accuracy.png")
        plt.close()

    # ----------------------------------------------------------------
    # Final report
    # ----------------------------------------------------------------

    def save_report(self, report):
        with open(self.dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)

    # ----------------------------------------------------------------
    # Convenience: also update the "current best" pointer used by
    # src/predict.py's default --checkpoint, so predict.py doesn't need to
    # know which experiment folder was most recent.
    # ----------------------------------------------------------------

    def promote_to_default(self, model_filename="model.pth", norm_stats_filename="norm_stats.npz"):
        import shutil

        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.dir / model_filename, config.BEST_MODEL_PATH)

        # Also promote this run's normalization stats alongside the checkpoint.
        # Without this, models/best_model.pth exists but nothing in models/
        # tells inference what mean/std to normalize incoming patches with —
        # src/predict.py and notebooks/05_inference_submission.ipynb both
        # need this file to actually match what the promoted checkpoint was
        # trained on, not just some other run's statistics.
        norm_stats_src = self.dir / norm_stats_filename
        if norm_stats_src.exists():
            shutil.copy(norm_stats_src, config.MODEL_DIR / norm_stats_filename)
