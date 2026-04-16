from __future__ import annotations
from pathlib import Path

# -----------------------------
# Data settings
# -----------------------------
DATA_DIR = Path("data")
TRAIN_SUBSET = 12000
VAL_SUBSET = 3000
EVAL_SUBSET = 3000
QUANTILE = 0.90
SEED = 42

# -----------------------------
# Random Forest
# -----------------------------
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 18
RF_MIN_SAMPLES_LEAF = 2

# -----------------------------
# XGBoost
# -----------------------------
XGB_N_ESTIMATORS = 400
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.90
XGB_COLSAMPLE_BYTREE = 0.90

# -----------------------------
# Transformer
# -----------------------------
TRANSFORMER_EPOCHS = 12
TRANSFORMER_BATCH_SIZE = 128
TRANSFORMER_LEARNING_RATE = 1e-3
TRANSFORMER_WEIGHT_DECAY = 1e-4
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 128
TRANSFORMER_DROPOUT = 0.10
