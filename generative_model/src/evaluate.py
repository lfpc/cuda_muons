import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from src.data_utils import to_cond_target


def plot_loss_curve(history, plot_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train"], color="steelblue", label="Train", lw=1.5)
    ax.plot(history["val"],   color="firebrick",  label="Val",   lw=1.5)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("NLL",   fontsize=14)
    ax.set_title("Training Loss", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "loss_curve.png"), dpi=150)
    plt.close()
    print("Saved loss_curve.png")


def evaluate_and_plot(model, history, norm_stats, config, eval_bins):
    """Plot diagnostics and KS statistics.

    eval_bins: list of (bin_label, p0, px, py, pz, step) tuples — one per
               representative momentum bin.  Use get_eval_bins_from_file or
               get_eval_bins_from_simul to build this list.
    """
    device = next(model.parameters()).device
    os.makedirs(config["plot_dir"], exist_ok=True)

    plot_loss_curve(history, config["plot_dir"])

    in_mean  = norm_stats["in_mean"]
    in_std   = norm_stats["in_std"]
    tgt_mean = norm_stats["tgt_mean"]
    tgt_std  = norm_stats["tgt_std"]

    n_plot = len(eval_bins)
    fig_marg, axs_marg = plt.subplots(n_plot, 2, figsize=(14, 4 * n_plot))
    fig_joint, axs_joint = plt.subplots(3, 2, figsize=(14, 18))
    ks_lines = ["bin, log_dPt KS, log_dPt p-val, log_dPz KS, log_dPz p-val"]

    model.eval()
    with torch.no_grad():
        for plot_i, (bin_label, p0, px, py, pz, step) in enumerate(
            tqdm(eval_bins, desc="Evaluating bins", unit="bin")
        ):
            cond, target = to_cond_target(p0, px, py, pz, step)
            true_log_dPt = target[:, 0]
            true_log_dPz = target[:, 1]

            cond_norm = ((torch.tensor(cond) - in_mean) / in_std).to(device)
            samples_norm = model.sample(cond_norm).cpu().numpy()
            samples = samples_norm * tgt_std.numpy() + tgt_mean.numpy()
            gen_log_dPt = samples[:, 0]
            gen_log_dPz = samples[:, 1]

            for col, (true_vals, gen_vals, var_name) in enumerate([
                (true_log_dPt, gen_log_dPt, r"$\log(\delta P_t / P_0)$"),
                (true_log_dPz, gen_log_dPz, r"$\log(-\delta P_z / P_0)$"),
            ]):
                ax = axs_marg[plot_i, col]
                lo = min(np.percentile(true_vals, 0.1),  np.percentile(gen_vals, 0.1))
                hi = max(np.percentile(true_vals, 99.9), np.percentile(gen_vals, 99.9))
                bins = np.linspace(lo, hi, 80)
                ax.hist(true_vals, bins=bins, density=True, histtype="step",
                        color="firebrick",  lw=1.5, label="Geant4")
                ax.hist(gen_vals,  bins=bins, density=True, histtype="step",
                        color="steelblue", lw=1.5, label="Flow")
                ax.set_yscale("log")
                ax.set_title(f"{var_name}\n{bin_label}", fontsize=11)
                ax.legend(fontsize=9)
                ax.grid(True, linestyle="--", alpha=0.6)

            ks_pt = stats.ks_2samp(true_log_dPt, gen_log_dPt)
            ks_pz = stats.ks_2samp(true_log_dPz, gen_log_dPz)
            ks_lines.append(
                f"{bin_label}, {ks_pt.statistic:.4f}, {ks_pt.pvalue:.4f}, "
                f"{ks_pz.statistic:.4f}, {ks_pz.pvalue:.4f}"
            )

            if plot_i < 3:
                axs_joint[plot_i, 0].hist2d(true_log_dPt, true_log_dPz, bins=60,
                                             cmap="Reds", density=True)
                axs_joint[plot_i, 0].set_title(f"Geant4: {bin_label}", fontsize=10)
                axs_joint[plot_i, 0].set_xlabel(r"$\log(\delta P_t/P_0)$")
                axs_joint[plot_i, 0].set_ylabel(r"$\log(-\delta P_z/P_0)$")
                axs_joint[plot_i, 1].hist2d(gen_log_dPt, gen_log_dPz, bins=60,
                                             cmap="Blues", density=True)
                axs_joint[plot_i, 1].set_title(f"Flow: {bin_label}", fontsize=10)
                axs_joint[plot_i, 1].set_xlabel(r"$\log(\delta P_t/P_0)$")
                axs_joint[plot_i, 1].set_ylabel(r"$\log(-\delta P_z/P_0)$")

    fig_marg.suptitle("1D Marginals: Geant4 vs Flow", fontsize=16, y=1.001)
    fig_marg.tight_layout()
    fig_marg.savefig(os.path.join(config["plot_dir"], "marginals_comparison.png"),
                     dpi=150, bbox_inches="tight")
    plt.close(fig_marg)
    print("Saved marginals_comparison.png")

    fig_joint.suptitle("Joint Distribution: Geant4 (red) vs Flow (blue)", fontsize=14)
    fig_joint.tight_layout()
    fig_joint.savefig(os.path.join(config["plot_dir"], "joint_distribution.png"),
                      dpi=150, bbox_inches="tight")
    plt.close(fig_joint)
    print("Saved joint_distribution.png")

    ks_path = os.path.join(config["plot_dir"], "ks_stats.txt")
    with open(ks_path, "w") as f:
        f.write("\n".join(ks_lines) + "\n")
    print("Saved ks_stats.txt")
    print("\nKS statistics (lower = better):")
    for line in ks_lines:
        print(" ", line)
