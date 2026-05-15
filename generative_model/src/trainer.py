import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.model import ConditionalFlowModel
from src.data_utils import build_loaders


def save_checkpoint(model, optimizer, epoch, val_loss, norm_stats, config, tag):
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    path = os.path.join(config["checkpoint_dir"], f"{tag}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "norm_stats": norm_stats,
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)


def train(config, data_fn):
    """Train the flow model.

    data_fn(config) -> (inputs, targets) — either load_data or simulate_data.
    Returns (model, history, norm_stats).
    """
    device = torch.device(config["device"])
    print(f"Using device: {device}")
    torch.manual_seed(config["seed"])

    inputs, targets = data_fn(config)
    train_loader, val_loader, norm_stats = build_loaders(inputs, targets, config)
    del inputs, targets

    model = ConditionalFlowModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["n_epochs"])

    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    epoch_bar = tqdm(range(1, config["n_epochs"] + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_losses = []
        for x_cond, x_target in tqdm(train_loader, desc="  Train", leave=False, unit="batch"):
            x_cond   = x_cond.to(device)
            x_target = x_target.to(device)
            optimizer.zero_grad()
            log_prob = model.log_prob(x_target, x_cond)
            finite_mask = torch.isfinite(log_prob)
            if finite_mask.any():
                loss = -log_prob[finite_mask].mean()
            else:
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            model.clear_cache()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_cond, x_target in tqdm(val_loader, desc="    Val", leave=False, unit="batch"):
                x_cond   = x_cond.to(device)
                x_target = x_target.to(device)
                lp = model.log_prob(x_target, x_cond)
                finite_mask = torch.isfinite(lp)
                if finite_mask.any():
                    val_losses.append(-lp[finite_mask].mean().item())
                model.clear_cache()

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses) if val_losses else float("nan")
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        scheduler.step()

        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, norm_stats, config, "best")

        if epoch % config["save_every_n_epochs"] == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, norm_stats, config, f"epoch_{epoch:03d}")

    return model, history, norm_stats
