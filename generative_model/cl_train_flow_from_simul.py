"""
Train a conditional normalizing flow with on-the-fly Geant4 simulation.

    python cl_train_flow_from_simul.py
    python cl_train_flow_from_simul.py --device cuda --n_epochs 100 --n_cores 32
"""

import argparse
import os

from src.simulation import simulate_data, get_eval_bins_from_simul
from src.trainer import train
from src.evaluate import evaluate_and_plot

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # Simulation
    "material": "G4_Fe",
    "n_cores": 8,
    "n_samples_per_group": 50_000,
    "n_momentum_bins": 30,
    "p_min": 0.18,
    "p_max": 400.0,

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
    parser = argparse.ArgumentParser(
        description="Train conditional normalizing flow with on-the-fly Geant4 simulation"
    )
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
    model, history, norm_stats = train(config, simulate_data)
    print("\nTraining complete. Running evaluation...")
    eval_bins = get_eval_bins_from_simul(config)
    evaluate_and_plot(model, history, norm_stats, config, eval_bins)
