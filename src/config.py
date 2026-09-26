import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# The hardcoded default below only exists on whichever Windows machine this was originally
# developed on — running anywhere else (Linux, Mac, CI, a teammate's machine) without setting
# BIOHUB_DATASET_PATH will point at a path that can't exist there. Prefer the environment
# variable; keep the original hardcoded value only as a last-resort fallback so existing
# workflows on that one machine don't break.
DATASET_PATH = Path(os.environ.get(
    "BIOHUB_DATASET_PATH",
    r"D:\Datasets\BioHub\biohub-cell-tracking-during-development",
))
TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"

OUTPUT_PATH = Path("outputs")
MODEL_DIR = Path("models")
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
LAST_MODEL_PATH = MODEL_DIR / "last_model.pth"
BEST_NORM_STATS_PATH = MODEL_DIR / "norm_stats.npz"

LOG_FILE = OUTPUT_PATH / "training.log"

# --------------------------------------------------------------------------
# Detection (used by src/detector.py)
# --------------------------------------------------------------------------
DETECTION_THRESHOLD = 10 # peak_local_max threshold_abs
GAUSSIAN_SIGMA = 2
CELL_RADIUS = 5            # peak_local_max min_distance

# --------------------------------------------------------------------------
# Patch extraction (used by notebooks/04_training.ipynb, src/predict.py)
# --------------------------------------------------------------------------
PATCH_SIZE = 32
NEGATIVE_EXCLUSION_RADIUS = 20  # min distance from a positive center for a valid negative sample

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# Loss: "bce" (nn.BCEWithLogitsLoss) or "focal" (src/losses.py FocalLoss)
LOSS_FN = "bce"
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# LR scheduler (ReduceLROnPlateau, monitors val_loss)
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_MIN_LR = 1e-6

# Early stopping (monitors val_loss)
EARLY_STOPPING_PATIENCE = 6

# Augmentation (applied to training patches only, see src/augmentations.py)
USE_AUGMENTATION = True
BRIGHTNESS_MAX_DELTA = 0.2
GAUSSIAN_NOISE_SIGMA = 0.05

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_LEVEL = "INFO"

# --------------------------------------------------------------------------
# Tracking / lineage (used by src/tracker.py, src/lineage.py)
# --------------------------------------------------------------------------
TRACKING_MAX_DISTANCE = 25  # HungarianTracker max_distance: normal frame-to-frame linking

# LineageBuilder's secondary search radius for division candidates (daughter
# cells can end up a bit further from the parent's last position than normal
# frame-to-frame motion, so this is intentionally looser than
# TRACKING_MAX_DISTANCE, not equal to it).
DIVISION_MAX_DISTANCE = 30

# --------------------------------------------------------------------------
# Physical calibration (used by src/analytics.py for real-unit speed/velocity)
# --------------------------------------------------------------------------
# Both None by default, ON PURPOSE. Every distance/speed number this pipeline computes
# internally is in pixels and frames — turning that into "12 um/min" requires knowing this
# specific microscope's pixel size and this specific acquisition's frame interval, neither of
# which can be guessed or defaulted to something plausible-looking without being dishonest
# about it. src/analytics.py reports pixels/frame whenever these are None, and only switches to
# real units once you set them here for your actual data.
PIXEL_SIZE_UM = None      # e.g. 0.65 -- microns per pixel, from your microscope's calibration
FRAME_INTERVAL_MIN = None  # e.g. 5.0 -- minutes between consecutive frames in the time-lapse

