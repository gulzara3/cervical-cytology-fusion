"""
Training entry point (contract) for the uncertainty-aware fusion framework.

The concrete, fully executable end-to-end run : data download, feature building,
training, evaluation, and every analysis reported in the paper : is
notebooks/Reproducible_Pipeline.ipynb (cells 1-15, then R1-R9). It imports the
same modules in src/, uses the global seed 42, and reproduces the shipped
results/ and figures/ on a single GPU (reference run: NVIDIA Tesla T4).

This script documents the training contract for users adapting the pipeline to
local (non-Colab) environments.
"""
from src.config import config, set_seed, make_dirs


def main():
    make_dirs()
    set_seed(config.SEED)
    raise SystemExit(
        "Run scripts/download_data.py, then follow notebooks/"
        "Reproducible_Pipeline.ipynb for the executable end-to-end pipeline."
    )


if __name__ == "__main__":
    main()
