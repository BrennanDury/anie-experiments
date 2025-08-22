import numpy as np
from scipy.io import loadmat
import torch
from typing import Optional, Tuple
import h5py

def subsample(
    x: torch.Tensor, n: Optional[int] = None, axis: int = 0
) -> torch.Tensor:
    if n is None or n >= x.shape[axis]:
        return x
    idx = torch.linspace(0, x.shape[axis] - 1, n, dtype=torch.long)
    slicers = [slice(None)] * x.ndim
    slicers[axis] = idx
    return x[tuple(slicers)]

def load_navier_stokes_data(
    data_path: str,
    n_timesteps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data = loadmat(data_path)
    u_np = data["u"]
    a_np = data["a"]

    u = torch.from_numpy(u_np)[..., None].permute(0, 3, 1, 2, 4)
    a = torch.from_numpy(a_np)[..., None, None].permute(0, 3, 1, 2, 4)

    u = subsample(u, n_timesteps, axis=1)
    return a, u

def load_burgers_data(
    data_path: str,
    n_timesteps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(data_path).to(torch.float32)
    data = data.permute(0, 2, 1)[:, 1:, :, None, None]
    a = data[:, :1]
    u = data[:, 1:]
    u = subsample(u, n_timesteps, axis=1)
    return a, u

def load_swe_data(
    data_path: str,
    n_timesteps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(data_path, 'r') as f:
        data_array = np.stack([np.array(f[str(i).zfill(4)]['data']) for i in range(1000)])
    data_tensor = torch.tensor(data_array)
    a = data_tensor[:, :1]
    u = data_tensor[:, 1:]
    u = subsample(u, n_timesteps, axis=1)
    return a, u