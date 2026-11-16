# config.py
import os

# Paths (adjust for Kaggle / local)
KAGGLE_DATASET_PATH = None  # will be set by kagglehub if used
ROOT_DIR = None             # set after download

# Training
NUM_CLASSES = 24
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
PATIENCE = 3
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
# TEST_RATIO = remaining

# Model
IMAGE_SIZE = 224
DROPBLOCK_P2 = 0.1
DROPBLOCK_BLOCKSIZE2 = 5
DROPBLOCK_P4 = 0.2
DROPBLOCK_BLOCKSIZE4 = 7

# Checkpoint paths (Kaggle style)
CHECKPOINT_PATH = "/kaggle/working/checkpoint_v2.pth"
BEST_MODEL_PATH = "/kaggle/working/best_model_v2.pth"
HISTORY_FILE = "/kaggle/working/training_history_v2.json"

# Exclude
EXCLUDE_TYPES = ["oral cancer"]