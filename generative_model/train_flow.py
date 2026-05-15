"""
Train a conditional normalizing flow on muon data loaded from an HDF5 file.

    python train_flow.py
    python train_flow.py --device cuda --n_epochs 100 --n_samples_per_group 100000
"""

import argparse
import os

from src.data_utils import load_data, get_eval_bins_from_file
from src.trainer import train
from src.evaluate import evaluate_and_plot

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "data_file": os.path.join(_SCRIPT_DIR, "../data/muon_data_energy_loss_sens_G4_Fe.h5"),
    "n_samples_per_group": 50_000,
    "val_fraction": 0.1,
    "seed": 42,

    "context_hidden_dims": [64, 64],
    "context_out_dim": 32,

    "n_transforms": 8,
    "transform_hidden_dims": [128, 128],
    "count_bins": 16,
    "spline_bound": 6.0,

    "batch_size": 4096,
    "lr": 3e-4,
    "n_epochs": 50,
    "grad_clip": 1.0,

    "checkpoint_dir": os.path.join(_SCRIPT_DIR, "checkpoints"),
    "plot_dir": os.path.join(_SCRIPT_DIR, "plots"),
    "save_every_n_epochs": 5,
    "device": "cuda",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional normalizing flow for muon physics")
    for key, val in CONFIG.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", action="store_true", default=val)
        elif isinstance(val, list):
            parser.add_argument(f"--{key}", nargs="+", type=type(val[0]) if val else int, default=val)
        else:
            parser.add_argument(f"--{key}", type=type(val), default=val)
    return vars(parser.parse_args())


if __name__ == "__main__":
    config = parse_args()
    model, history, norm_stats = train(config, load_data)
    print("\nTraining complete. Running evaluation...")
    eval_bins = get_eval_bins_from_file(config)
    evaluate_and_plot(model, history, norm_stats, config, eval_bins)
