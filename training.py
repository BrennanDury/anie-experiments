from re import I
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
        y, z = model.trajectory_to_trajectory(encoder(input_trajectory), kind=kind)
    else:
        y, z = model.initial_conditions_to_trajectory(encoder(initial_conditions), trajectory.shape[1], kind=kind)
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

class Pipeline(nn.Module):
    def __init__(self, model, encoder, decoder, positional_encoding, positional_unencoding, residual=False):
        super().__init__()
        self.model = model  # (B, T*Hp*Wp, C) -> (B, T*Hp*Wp, C)
        self.encoder = encoder  # (B, T, H, W, Q) -> (B, T, Hp, Wp, C)
        self.decoder = decoder  # (B, T, Hp, Wp, C+P) -> (B, T, H, W, Q)
        self.positional_encoding = positional_encoding  # (B, T, Hp, Wp, C) -> (B, T, Hp, Wp, C+P)
        self.positional_unencoding = positional_unencoding  # (B, T, Hp, Wp, C+P) -> (B, T, Hp, Wp, C)
        self.residual = residual

    def forward(self, x, t=0):
        """
        x : (B, T, H, W, Q)
        t : int
        return : (B, T, H, W, Q)
        """

        y = self.encoder(x)
        y = self.positional_encoding(y, t)
        projection = lambda z: self.positional_encoding(self.positional_unencoding(z.reshape_as(y), t), t).reshape_as(z)
        y = self.model(y.flatten(1, 3), projection).reshape_as(y)
        y = self.positional_unencoding(y, t)
        y = self.decoder(y)
        if self.residual:
            y = x + y
        return y

    def clear_kv_cache(self):
        self.model.clear_kv_cache()