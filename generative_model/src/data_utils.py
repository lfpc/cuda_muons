import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def to_cond_target(p0, px, py, pz, step):
    """Filter invalid rows and compute (cond, target) from raw physics arrays.

    Returns:
        cond:   (N, 2) float32 — [log(p0), log(step)]
        target: (N, 2) float32 — [log(pt/p0), log(-delta_pz/p0)]
    """
    pt = np.sqrt(px**2 + py**2)
    delta_pz = pz - p0
    valid = (pt > 0) & (delta_pz < 0) & (p0 > 0) & (step > 0)
    p0, pt, delta_pz, step = p0[valid], pt[valid], delta_pz[valid], step[valid]
    log_dPt = np.log(pt / p0)
    log_dPz = np.log(-delta_pz / p0)
    cond = np.stack([np.log(p0), np.log(step)], axis=1)
    target = np.stack([log_dPt, log_dPz], axis=1)
    return cond, target


def build_loaders(inputs, targets, config):
    """Split into train/val, compute normalization stats, return DataLoaders."""
    n = len(inputs)
    n_val = int(n * config["val_fraction"])
    rng = np.random.default_rng(config["seed"] + 1)
    perm = rng.permutation(n)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    in_mean = inputs[train_idx].mean(axis=0).astype(np.float32)
    in_std = inputs[train_idx].std(axis=0).astype(np.float32)
    tgt_mean = targets[train_idx].mean(axis=0).astype(np.float32)
    tgt_std = targets[train_idx].std(axis=0).astype(np.float32)

    norm_stats = {
        "in_mean": torch.tensor(in_mean),
        "in_std": torch.tensor(in_std),
        "tgt_mean": torch.tensor(tgt_mean),
        "tgt_std": torch.tensor(tgt_std),
    }

    def normalize(inp, tgt):
        inp_n = (inp - in_mean) / in_std
        tgt_n = (tgt - tgt_mean) / tgt_std
        return torch.tensor(inp_n), torch.tensor(tgt_n)

    in_tr, tgt_tr = normalize(inputs[train_idx], targets[train_idx])
    in_val, tgt_val = normalize(inputs[val_idx], targets[val_idx])

    bound = config["spline_bound"]
    pct_out = (tgt_tr.abs() > bound).float().mean().item() * 100
    if pct_out > 0.1:
        print(f"  WARNING: {pct_out:.2f}% of normalized targets exceed spline bound={bound:.1f}. "
              "Consider increasing spline_bound.")
    else:
        print(f"  Spline bound check OK: <0.1% of targets exceed {bound:.1f}")

    bs = config["batch_size"]
    train_loader = DataLoader(
        TensorDataset(in_tr, tgt_tr), batch_size=bs, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(in_val, tgt_val), batch_size=bs * 2, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, norm_stats


def load_data(config):
    """Load training data from HDF5 file."""
    data_file = config["data_file"]
    n_per_group = config["n_samples_per_group"]
    rng = np.random.default_rng(config["seed"])

    all_inputs, all_targets = [], []
    print(f"Loading data from {data_file} ({n_per_group} samples/group)...")
    t0 = time.time()

    with h5py.File(data_file, "r") as f:
        keys = sorted(f.keys(), key=lambda k: float(k.strip("()").split(",")[0]))
        for key in tqdm(keys, desc="Loading groups", unit="group"):
            grp = f[key]
            n_avail = grp["initial_momenta"].shape[0]
            n = min(n_per_group, n_avail)
            idx = rng.choice(n_avail, size=n, replace=False)
            idx.sort()

            p0   = grp["initial_momenta"][idx].astype(np.float32)
            px   = grp["px"][idx].astype(np.float32)
            py   = grp["py"][idx].astype(np.float32)
            pz   = grp["pz"][idx].astype(np.float32)
            step = grp["step_length"][idx].astype(np.float32)

            cond, target = to_cond_target(p0, px, py, pz, step)
            all_inputs.append(cond)
            all_targets.append(target)

    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    mask = np.isfinite(inputs).all(axis=1) & np.isfinite(targets).all(axis=1)
    inputs, targets = inputs[mask], targets[mask]

    print(f"  Loaded {len(inputs):,} samples in {time.time()-t0:.1f}s")
    print(f"  Input  range: log_P [{inputs[:,0].min():.2f}, {inputs[:,0].max():.2f}], "
          f"log_step [{inputs[:,1].min():.2f}, {inputs[:,1].max():.2f}]")
    print(f"  Target range: log_dPt [{targets[:,0].min():.2f}, {targets[:,0].max():.2f}], "
          f"log_dPz [{targets[:,1].min():.2f}, {targets[:,1].max():.2f}]")

    return inputs, targets


def get_eval_bins_from_file(config, n_eval=50_000, n_bins_plot=6):
    """Load raw arrays for n_bins_plot representative momentum bins from HDF5.

    Returns list of (bin_label, p0, px, py, pz, step) tuples.
    """
    data_file = config["data_file"]
    rng = np.random.default_rng(0)

    with h5py.File(data_file, "r") as f:
        keys = sorted(f.keys(), key=lambda k: float(k.strip("()").split(",")[0]))

    bin_indices = np.linspace(0, len(keys) - 1, n_bins_plot, dtype=int)
    eval_keys = [keys[i] for i in bin_indices]

    result = []
    for key in tqdm(eval_keys, desc="Loading eval bins", unit="bin"):
        with h5py.File(data_file, "r") as f:
            grp = f[key]
            n_avail = grp["initial_momenta"].shape[0]
            n = min(n_eval, n_avail)
            idx = rng.choice(n_avail, size=n, replace=False)
            idx.sort()
            p0   = grp["initial_momenta"][idx].astype(np.float32)
            px   = grp["px"][idx].astype(np.float32)
            py   = grp["py"][idx].astype(np.float32)
            pz   = grp["pz"][idx].astype(np.float32)
            step = grp["step_length"][idx].astype(np.float32)

        p_low  = float(key.strip("()").split(",")[0])
        p_high = float(key.strip("()").split(",")[1])
        result.append((f"P ∈ [{p_low:.2f}, {p_high:.2f}] GeV", p0, px, py, pz, step))

    return result
