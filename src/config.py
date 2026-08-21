"""Global configuration for the uncertainty-aware cervical cytology framework."""
import os, random
import numpy as np
import torch


class Config:
    EXPERIMENT_NAME = "CervicalCancer_MultiModal_"
    SEED = 42
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    NUM_EPOCHS = 30
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10
    NUM_CLASSES = 2
    DROPOUT_RATE = 0.4
    MC_DROPOUT_SAMPLES = 20
    MODALITY_DROPOUT_PROB = 0.25
    USE_AMP = True
    TEST_SIZE = 0.15
    VAL_SIZE = 0.18
    N_GENES = 80
    N_MUTATIONS = 20
    N_METHYLATION = 40
    GENOMIC_NOISE_LEVEL = 0.7
    DIRS = ["data", "data/sipakmed", "data/herlev", "data/clinical",
            "models", "results", "figures"]


config = Config()


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def make_dirs() -> None:
    for d in config.DIRS:
        os.makedirs(d, exist_ok=True)
