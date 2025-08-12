import torch
from torch import Tensor, nn

"""
This file contains the training pipeline for the Navier-Stokes model.

The shape requirements are as follows:
model  # (B, T*Hp*Wp, C+P) -> (B, T*Hp*Wp, C+P)
encoder  # (B, T, H, W, Q) -> (B, T, Hp, Wp, C)
decoder  # (B, T, Hp, Wp, C) -> (B, T, H, W, Q)
positional_encoding  # (B, T, Hp, Wp, C) -> (B, T, Hp, Wp, C+P)
positional_unencoding  # (B, T, Hp, Wp, C+P) -> (B, T, Hp, Wp, C)
"""

def get_prediction(initial_conditions: Tensor,
                   trajectory: Tensor,
                   model: nn.Module,
                   encoder: nn.Module,
                   decoder: nn.Module,
                   kind: str,
                   decode_initial_conditions: bool) -> Tensor:
    if kind == "one_step":
        input_trajectory = torch.cat([initial_conditions, trajectory], dim=1)
        q = encoder(input_trajectory)
        y, z = model.trajectory_to_trajectory(q, kind=kind)
    else:
        q = encoder(initial_conditions)
        y, z = model.initial_conditions_to_trajectory(q, trajectory.shape[1], kind=kind)
    z = decoder(z)
    if decode_initial_conditions:
        y = decoder(y)
    else:
        y = initial_conditions
    return y, z

def training_epoch(loader, model, encoder, decoder, kind, loss_fn, optim, scheduler=None, grad_clip_norm=1.0, device=None, compute_initial_conditions_loss=False):
    running_loss = 0.0
    n_batches = 0
    for batch in loader:
        initial_conditions, trajectory = batch["a"], batch["u"]
        if device:
            initial_conditions = initial_conditions.to(device)
            trajectory = trajectory.to(device)
        optim.zero_grad()

        initial_conditions_pred, trajectory_preds = get_prediction(initial_conditions, trajectory, model, encoder, decoder, kind, decode_initial_conditions=compute_initial_conditions_loss)
        if compute_initial_conditions_loss:
            preds = torch.cat(initial_conditions_pred, trajectory_preds, dim=1)
            targets = torch.cat(initial_conditions, trajectory, dim=1)
        else:
            preds = trajectory_preds
            targets = trajectory
        loss = loss_fn(preds, targets)
        loss.backward()
        if grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(
                [p for p in optim.param_groups[0]['params'] if p.grad is not None],
                grad_clip_norm
            )
        optim.step()
        if scheduler:
            scheduler.step()
        running_loss += loss.item()
        model.clear_kv_cache()
        n_batches += 1
    return running_loss / n_batches

def evaluation_epoch(loader, model, encoder, decoder, kind, loss_fn, device=None, compute_initial_conditions_loss=False):
    running_loss = 0.0
    n_batches = 0
    for batch in loader:
        initial_conditions, trajectory = batch["a"], batch["u"]
        if device:
            initial_conditions = initial_conditions.to(device)
            trajectory = trajectory.to(device)
        initial_conditions_pred, trajectory_preds = get_prediction(initial_conditions, trajectory, model, encoder, decoder, kind, decode_initial_conditions=compute_initial_conditions_loss)
        if compute_initial_conditions_loss:
            preds = torch.cat(initial_conditions_pred, trajectory_preds, dim=1)
            targets = torch.cat(initial_conditions, trajectory, dim=1)
        else:
            preds = trajectory_preds
            targets = trajectory
        loss = loss_fn(preds, targets)
        running_loss += loss.item()
        model.clear_kv_cache()
        n_batches += 1
    return running_loss / n_batches