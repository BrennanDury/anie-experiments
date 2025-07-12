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

Masking must be handled by the model.
"""

def broadcast_initial_conditions(x: Tensor, length: int) -> Tensor:
    """
    x : (B, 1, H, W, Q)
    return : (B, length, H, W, Q)
    """
    return x.repeat(1, length, 1, 1, 1)

def acausal_prediction(x: Tensor, model: nn.Module, T: int) -> Tensor:
    """
    x : (B, 1, H, W, Q)
    model : (B, T+1, H, W, Q) -> (B, T+1, H, W, Q)
    return : (B, T, H, W, Q)
    """
    y = broadcast_initial_conditions(x, T+1)
    return model(y)[:, 1:]

def generate_prediction(x: Tensor, model: nn.Module, T: int) -> Tensor:
    """
    x : (B, 1, H, W, Q)
    model : (B, 1, H, W, Q), t -> (B, 1, H, W, Q)
    T : int
    return : (B, T, H, W, Q)
    """
    sequence = [x]
    for t in range(T):
        sequence.append(model(sequence[-1], t))
    return torch.cat(sequence, dim=1)[:, 1:]

def one_step_prediction(x: Tensor, model: nn.Module) -> Tensor:
    """
    x : (B, T, H, W, Q)
    model : (B, T, H, W, Q) -> (B, T, H, W, Q)
    return : (B, T, H, W, Q)
    """
    return model(x)

def get_prediction(initial_conditions: Tensor, trajectory: Tensor, model: nn.Module, kind: str) -> Tensor:
    """
    initial_conditions : (B, 1, H, W, Q)
    trajectory : (B, T, H, W, Q)
    model : X -> Y
    kind : str
    return : (B, T, H, W, Q)
    """
    if kind == "acausal":
        return acausal_prediction(initial_conditions, model, trajectory.shape[1])
    elif kind == "generate":
        return generate_prediction(initial_conditions, model, trajectory.shape[1])
    elif kind == "one_step":
        full_trajectory = torch.cat([initial_conditions, trajectory], dim=1)
        return one_step_prediction(full_trajectory[:, :-1], model)
    else:
        raise ValueError(f"Invalid kind: {kind}")

def training_epoch(loader, model, kind, loss_fn, optim, clip_grad_norm=1.0):
    running_loss = 0.0
    for initial_conditions, trajectory in loader:
        optim.zero_grad()
        preds = get_prediction(initial_conditions, trajectory, model, kind)
        loss = loss_fn(preds, trajectory)
        loss.backward()
        if clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                [p for p in optim.param_groups[0]['params'] if p.grad is not None],
                clip_grad_norm
            )
        optim.step()
        running_loss += loss.item()
        model.clear_kv_cache()
    return running_loss / len(loader)

def evaluation_epoch(loader, model, kind, loss_fn):
    running_loss = 0.0
    for init_cond, traj in loader:
        preds = get_prediction(init_cond, traj, model, kind)
        loss = loss_fn(preds, traj)
        running_loss += loss.item()
    return running_loss / len(loader)

class Pipeline(nn.Module):
    def __init__(self, model, encoder, decoder, positional_encoding, positional_unencoding):
        super().__init__()
        self.model = model  # (B, T*Hp*Wp, C) -> (B, T*Hp*Wp, C)
        self.encoder = encoder  # (B, T, H, W, Q) -> (B, T, Hp, Wp, C)
        self.decoder = decoder  # (B, T, Hp, Wp, C+P) -> (B, T, H, W, Q)
        self.positional_encoding = positional_encoding  # (B, T, Hp, Wp, C) -> (B, T, Hp, Wp, C+P)
        self.positional_unencoding = positional_unencoding  # (B, T, Hp, Wp, C+P) -> (B, T, Hp, Wp, C)

    def forward(self, x, t=0):
        """
        x : (B, T, H, W, Q)
        t : int
        return : (B, T, H, W, Q)
        """
        x = self.encoder(x)
        x = self.positional_encoding(x, t)
        x = self.model(x.flatten(1, 3)).reshape_as(x)
        x = self.positional_unencoding(x)
        x = self.decoder(x)
        return x

    def clear_kv_cache(self):
        self.model.clear_kv_cache()