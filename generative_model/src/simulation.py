import functools
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../utils_cuda_muons"))
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from collect_geant_data import simulate_muon_batch
from get_geometry import get_sphere_design

from src.data_utils import to_cond_target


def build_detector(material):
    detector = get_sphere_design(mag_field=[0., 0., 0.], material=material)
    detector["store_primary"] = True
    detector["store_all"] = False
    return detector


def simulate_bin(n_samples, bounds, detector, n_cores):
    """Run parallel Geant4 simulation for one momentum bin; return raw arrays."""
    chunk_size = n_samples // n_cores
    remainder = n_samples % n_cores
    chunks = [chunk_size + (1 if j < remainder else 0) for j in range(n_cores)]

    sim_fn = functools.partial(
        simulate_muon_batch, detector=detector, step_size=None, initial_momenta_bounds=bounds
    )

    p0_list, px_list, py_list, pz_list, step_list = [], [], [], [], []
    with mp.Pool(processes=n_cores) as pool:
        for batch in pool.imap_unordered(sim_fn, chunks):
            p0_list.extend(batch["initial_momenta"])
            px_list.extend(batch["px"])
            py_list.extend(batch["py"])
            pz_list.extend(batch["pz"])
            step_list.extend(batch["step_length"])

    return (
        np.array(p0_list,   dtype=np.float32),
        np.array(px_list,   dtype=np.float32),
        np.array(py_list,   dtype=np.float32),
        np.array(pz_list,   dtype=np.float32),
        np.array(step_list, dtype=np.float32),
    )


def simulate_data(config):
    """Simulate training data across logspace momentum bins."""
    n_bins    = config["n_momentum_bins"]
    p_min     = max(config["p_min"], 0.18)
    p_max     = config["p_max"]
    n_per_bin = config["n_samples_per_group"]
    n_cores   = config["n_cores"]

    detector = build_detector(config["material"])
    momenta_points = np.logspace(np.log10(p_min), np.log10(p_max), num=n_bins + 1)

    all_inputs, all_targets = [], []
    print(f"Simulating {n_bins} bins × {n_per_bin:,} samples on {n_cores} cores...")
    t0 = time.time()

    for i in tqdm(range(n_bins), desc="Simulating bins", unit="bin"):
        bounds = (float(momenta_points[i]), float(momenta_points[i + 1]))
        p0, px, py, pz, step = simulate_bin(n_per_bin, bounds, detector, n_cores)
        cond, target = to_cond_target(p0, px, py, pz, step)
        all_inputs.append(cond)
        all_targets.append(target)

    inputs  = np.concatenate(all_inputs,  axis=0)
    targets = np.concatenate(all_targets, axis=0)
    mask = np.isfinite(inputs).all(axis=1) & np.isfinite(targets).all(axis=1)
    inputs, targets = inputs[mask], targets[mask]

    print(f"  Total: {len(inputs):,} samples in {time.time()-t0:.1f}s")
    print(f"  Input  range: log_P [{inputs[:,0].min():.2f}, {inputs[:,0].max():.2f}], "
          f"log_step [{inputs[:,1].min():.2f}, {inputs[:,1].max():.2f}]")
    print(f"  Target range: log_dPt [{targets[:,0].min():.2f}, {targets[:,0].max():.2f}], "
          f"log_dPz [{targets[:,1].min():.2f}, {targets[:,1].max():.2f}]")

    return inputs, targets


def get_eval_bins_from_simul(config, n_eval=50_000, n_bins_plot=6):
    """Simulate fresh evaluation data for n_bins_plot representative momentum bins.

    Returns list of (bin_label, p0, px, py, pz, step) tuples.
    """
    n_bins  = config["n_momentum_bins"]
    p_min   = max(config["p_min"], 0.18)
    p_max   = config["p_max"]
    n_cores = config["n_cores"]

    detector = build_detector(config["material"])
    momenta_points = np.logspace(np.log10(p_min), np.log10(p_max), num=n_bins + 1)
    bin_indices = np.linspace(0, n_bins - 1, n_bins_plot, dtype=int)

    result = []
    for bin_idx in tqdm(bin_indices, desc="Simulating eval bins", unit="bin"):
        m0 = float(momenta_points[bin_idx])
        m1 = float(momenta_points[bin_idx + 1])
        p0, px, py, pz, step = simulate_bin(n_eval, (m0, m1), detector, n_cores)
        result.append((f"P ∈ [{m0:.2f}, {m1:.2f}] GeV", p0, px, py, pz, step))

    return result
