"""Autonomous neural vector field, with optional affine state normalization.

The model learns dz/dt where z=(x-mean)/std. The physical field is
 dx/dt=std*f(z). The buffers are included in the state_dict.
"""
import torch
from torch import nn
from torchdiffeq import odeint


class VectorField(nn.Module):
    def __init__(self, dim, hidden=64, mean=None, std=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        self.register_buffer('state_mean', torch.zeros(dim) if mean is None else torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer('state_std', torch.ones(dim) if std is None else torch.as_tensor(std, dtype=torch.float32).clone())

    def forward(self, t, z):
        """Normalized-time derivative dz/dt (physical time units)."""
        return self.net(z)

    def trajectory(self, z0, t, rtol=1e-4, atol=1e-5):
        return odeint(self, z0, t, method='dopri5', rtol=rtol, atol=atol).transpose(0, 1)

    def physical_field(self, x):
        z = (x - self.state_mean) / self.state_std
        return self.net(z) * self.state_std

    def physical_trajectory(self, x0, t):
        z0 = (x0 - self.state_mean) / self.state_std
        return self.trajectory(z0, t) * self.state_std + self.state_mean
